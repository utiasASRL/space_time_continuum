#ifndef TWOSIDEDINTERPOLATIONFACTOR_HPP
#define TWOSIDEDINTERPOLATIONFACTOR_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/InterpolatedNode.hpp>

namespace Spacetime::Priors
{
    /**
     * @class TwoSidedInterpolationFactor
     * @brief Evaluates an inner gtsam::NoiseModelFactor on two interpolated nodes at once --
     *        typically of two different robots (see Joints::Joint) -- chaining its
     *        Jacobians through both interpolations to both sets of bordering nodes.
     *
     * The inner factor is defined on the two nodes' inner keys (see InterpolatedNode),
     * which must not overlap: give one side renamed placeholders. Any other inner key is a
     * genuine variable (e.g. a sensor bias) and is passed through unchanged, as in
     * Types::NearestNodeFactor: read straight from the outer values, with the inner
     * Jacobian as its own and no conditional covariance. Its covariance becomes
     *
     *     Sigma_inner + G_A Pcheck_A G_A^T + G_B Pcheck_B G_B^T + extra(inner values),
     *
     * where G_X is the inner Jacobian w.r.t. side X's interpolated node and Pcheck_X that
     * interpolation's conditional covariance (the two GP interpolations are independent),
     * and `extra` an optional state-dependent term (e.g. a joint offset's uncertainty
     * mapped into a velocity residual). Like InterpolationFactor, the noise model is
     * rebuilt at every linearization, and error() reuses the latest one when
     * `cache_noise_model` is set.
     *
     * @tparam InterpolatorA Side A's interpolator.
     * @tparam InterpolatorB Side B's interpolator.
     */
    template <typename InterpolatorA, typename InterpolatorB>
    class TwoSidedInterpolationFactor : public gtsam::NoiseModelFactor
    {
    public:
        using Base = gtsam::NoiseModelFactor;
        using This = TwoSidedInterpolationFactor<InterpolatorA, InterpolatorB>;
        using NodeA = InterpolatedNode<InterpolatorA>;
        using NodeB = InterpolatedNode<InterpolatorB>;
        /// Extra error-space covariance, evaluated on the inner factor's values.
        using ExtraCovarianceFn = std::function<Eigen::MatrixXd(const gtsam::Values &inner_values)>;

        using Base::error;

        /**
         * @param inner_factor Factor on (a subset of) the two nodes' inner keys, with a
         *        (non-constrained) Gaussian noise model.
         * @param node_a Side A's interpolated node.
         * @param node_b Side B's interpolated node; its inner keys must differ from side A's.
         * @param extra_covariance Optional extra covariance (see the class description).
         * @param cache_noise_model See InterpolationFactor.
         * @throws std::invalid_argument for a null inner factor, overlapping inner keys, a
         *         passed-through inner key that is also a bordering node's key, or a noise
         *         model that is not a non-constrained noiseModel::Gaussian.
         */
        TwoSidedInterpolationFactor(const gtsam::NoiseModelFactor::shared_ptr &inner_factor, const NodeA &node_a, const NodeB &node_b,
                                    ExtraCovarianceFn extra_covariance = nullptr, bool cache_noise_model = true)
            : Base(inner_factor ? inner_factor->noiseModel() : gtsam::SharedNoiseModel()),
              m_inner_factor(inner_factor), m_node_a(node_a), m_node_b(node_b),
              m_extra_covariance(std::move(extra_covariance)), m_cache_noise_model(cache_noise_model)
        {
            if (!inner_factor)
                throw std::invalid_argument("TwoSidedInterpolationFactor requires a non-null inner factor.");
            for (const gtsam::Key key : node_a.innerKeys())
                if (node_b.ownsInnerKey(key))
                    throw std::invalid_argument("TwoSidedInterpolationFactor: both sides use the same inner key; rename one side's placeholders.");
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(inner_factor->noiseModel());
            if (!gaussian || gaussian->isConstrained())
                throw std::invalid_argument("TwoSidedInterpolationFactor: the inner factor's noise model must be a non-constrained noiseModel::Gaussian.");
            m_base_covariance = gaussian->covariance();

            std::unordered_set<gtsam::Key> key_set;
            for (const auto &corner_keys : {node_a.cornerKeys(), node_b.cornerKeys()})
                for (const gtsam::Key key : corner_keys)
                    if (key_set.insert(key).second)
                        this->keys_.push_back(key);
            for (const gtsam::Key key : inner_factor->keys())
                if (!node_a.ownsInnerKey(key) && !node_b.ownsInnerKey(key))
                {
                    if (!key_set.insert(key).second)
                        throw std::invalid_argument("TwoSidedInterpolationFactor: a passed-through inner key is also a bordering node's key.");
                    this->keys_.push_back(key);
                }
            for (size_t i = 0; i < this->keys_.size(); ++i)
                m_key_to_index[this->keys_[i]] = i;
        }

        void print(const std::string &s = "", const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
        {
            std::cout << s << "TwoSidedInterpolationFactor between (t=" << m_node_a.time() << ", s=" << m_node_a.arclength()
                      << ") and (t=" << m_node_b.time() << ", s=" << m_node_b.arclength() << ") on ";
            for (const auto &k : this->keys())
                std::cout << keyFormatter(k) << " ";
            std::cout << std::endl;
            m_inner_factor->print("Inner Factor: ", keyFormatter);
        }

        /// Same type, keys and base noise model; the inner factor and query points are not compared.
        bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override
        {
            const This *e = dynamic_cast<const This *>(&expected);
            return e != nullptr && Base::equals(*e, tol);
        }

        using Base::unwhitenedError;
        /// The inner factor's unwhitened error on both interpolated nodes, with Jacobians w.r.t. this factor's keys.
        gtsam::Vector unwhitenedError(const gtsam::Values &values, gtsam::OptionalMatrixVecType H = nullptr) const override
        {
            return evaluate(values, H, nullptr);
        }

        /// The full (augmented) error-space covariance at `values` (see the class description).
        gtsam::Matrix covarianceAt(const gtsam::Values &values) const
        {
            gtsam::Matrix covariance;
            evaluate(values, nullptr, &covariance);
            return covariance;
        }

        std::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &x) const override
        {
            if (!active(x))
                return std::shared_ptr<gtsam::JacobianFactor>();

            std::vector<gtsam::Matrix> A(size());
            gtsam::Matrix covariance;
            gtsam::Vector b = -evaluate(x, &A, &covariance);
            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(covariance);
            if (m_cache_noise_model)
            {
                std::lock_guard<std::mutex> lock(m_cached_noise_model_mutex);
                m_cached_noise_model = noise_model;
            }
            noise_model->WhitenSystem(A, b);

            std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms(size());
            for (size_t j = 0; j < size(); ++j)
            {
                terms[j].first = keys()[j];
                terms[j].second.swap(A[j]);
            }
            return std::make_shared<gtsam::JacobianFactor>(terms, b);
        }

        double error(const gtsam::Values &c) const override
        {
            if (!active(c))
                return 0.0;

            gtsam::noiseModel::Gaussian::shared_ptr noise_model;
            if (m_cache_noise_model)
            {
                std::lock_guard<std::mutex> lock(m_cached_noise_model_mutex);
                noise_model = m_cached_noise_model;
            }
            gtsam::Vector e;
            if (noise_model)
                e = evaluate(c, nullptr, nullptr);
            else
            {
                gtsam::Matrix covariance;
                e = evaluate(c, nullptr, &covariance);
                noise_model = gtsam::noiseModel::Gaussian::Covariance(covariance);
            }
            return noise_model->loss(noise_model->squaredMahalanobisDistance(e));
        }

    private:
        /**
         * @brief Interpolates both nodes, evaluates the inner factor on them, and
         *        backpropagates its Jacobians to both sides' bordering keys.
         * @param covariance Optional output: the augmented error-space covariance.
         */
        gtsam::Vector evaluate(const gtsam::Values &values, gtsam::OptionalMatrixVecType H, gtsam::Matrix *covariance) const
        {
            const bool need_jacobians = H != nullptr || covariance != nullptr;
            typename NodeA::Jacobians jacobians_a;
            typename NodeB::Jacobians jacobians_b;
            Eigen::MatrixXd P_a, P_b;
            const auto interpolated_a = m_node_a.evaluate(values, H ? &jacobians_a : nullptr, covariance ? &P_a : nullptr);
            const auto interpolated_b = m_node_b.evaluate(values, H ? &jacobians_b : nullptr, covariance ? &P_b : nullptr);

            const gtsam::KeyVector &inner_keys = m_inner_factor->keys();
            gtsam::Values inner_values;
            for (const gtsam::Key key : inner_keys)
            {
                if (m_node_a.ownsInnerKey(key))
                    m_node_a.insert(key, *interpolated_a, inner_values);
                else if (m_node_b.ownsInnerKey(key))
                    m_node_b.insert(key, *interpolated_b, inner_values);
                else
                    inner_values.insert(key, values.at(key));
            }

            std::vector<gtsam::Matrix> H_inner(inner_keys.size());
            const gtsam::Vector error = need_jacobians ? m_inner_factor->unwhitenedError(inner_values, &H_inner)
                                                       : m_inner_factor->unwhitenedError(inner_values);
            if (!need_jacobians)
                return error;

            const int error_dim = static_cast<int>(error.size());
            Eigen::MatrixXd G_a, G_b;
            if (covariance)
            {
                G_a = Eigen::MatrixXd::Zero(error_dim, NodeA::node_dim);
                G_b = Eigen::MatrixXd::Zero(error_dim, NodeB::node_dim);
            }
            if (H)
                H->assign(size(), Eigen::MatrixXd());

            const auto index_of = [this](gtsam::Key key) { return m_key_to_index.at(key); };
            for (size_t i = 0; i < inner_keys.size(); ++i)
            {
                if (m_node_a.ownsInnerKey(inner_keys[i]))
                    m_node_a.chain(inner_keys[i], H_inner[i], jacobians_a, index_of, H, covariance ? &G_a : nullptr);
                else if (m_node_b.ownsInnerKey(inner_keys[i]))
                    m_node_b.chain(inner_keys[i], H_inner[i], jacobians_b, index_of, H, covariance ? &G_b : nullptr);
                else if (H)
                    (*H)[index_of(inner_keys[i])] = H_inner[i];
            }
            if (H)
            {
                NodeA::fillUntouched(jacobians_a, index_of, error_dim, *H);
                NodeB::fillUntouched(jacobians_b, index_of, error_dim, *H);
            }

            if (covariance)
            {
                *covariance = m_base_covariance + G_a * P_a * G_a.transpose() + G_b * P_b * G_b.transpose();
                if (m_extra_covariance)
                    *covariance += m_extra_covariance(inner_values);
            }
            return error;
        }

        gtsam::NoiseModelFactor::shared_ptr m_inner_factor;
        NodeA m_node_a;
        NodeB m_node_b;
        ExtraCovarianceFn m_extra_covariance;
        bool m_cache_noise_model;
        gtsam::Matrix m_base_covariance;
        /// The augmented noise model of the latest linearize() (see cache_noise_model).
        mutable gtsam::noiseModel::Gaussian::shared_ptr m_cached_noise_model;
        mutable std::mutex m_cached_noise_model_mutex;
        std::unordered_map<gtsam::Key, size_t> m_key_to_index;
    };
} // namespace Spacetime::Priors

#endif // TWOSIDEDINTERPOLATIONFACTOR_HPP
