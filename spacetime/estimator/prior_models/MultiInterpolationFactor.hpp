#ifndef MULTIINTERPOLATIONFACTOR_HPP
#define MULTIINTERPOLATIONFACTOR_HPP

#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
     * @class MultiInterpolationFactor
     * @brief Evaluates an inner gtsam::NoiseModelFactor on any number of interpolated nodes
     *        of one grid prior at once (e.g. every routing disk of a tendon, see
     *        Actuators::KinematicTendon), chaining its Jacobians through every
     *        interpolation to the bordering nodes. The N-node counterpart of
     *        TwoSidedInterpolationFactor.
     *
     * The inner factor is defined on the nodes' inner keys (see InterpolatedNode), which
     * must not overlap between nodes: give each node its own renamed placeholders. Nodes
     * may share bordering nodes; their Jacobian contributions are summed. Any other inner
     * key is a genuine variable (e.g. an actuator input) and is passed through unchanged:
     * read straight from the outer values, with the inner Jacobian as its own. The
     * covariance becomes
     *
     *     Sigma_inner + sum_i G_i Pcheck_i G_i^T,
     *
     * where G_i is the inner Jacobian w.r.t. node i and Pcheck_i that interpolation's
     * conditional covariance. The interpolations are treated as independent, which is an
     * approximation when nodes share bordering nodes. Like InterpolationFactor, the noise
     * model is rebuilt at every linearization, and error() reuses the latest one when
     * `cache_noise_model` is set.
     *
     * @tparam InterpolatorT The prior's interpolator.
     */
    template <typename InterpolatorT>
    class MultiInterpolationFactor : public gtsam::NoiseModelFactor
    {
    public:
        using Base = gtsam::NoiseModelFactor;
        using This = MultiInterpolationFactor<InterpolatorT>;
        using Node = InterpolatedNode<InterpolatorT>; ///< One interpolated node.

        using Base::error;

        /**
         * @param inner_factor Factor on (a subset of) the nodes' inner keys, with a
         *        (non-constrained) Gaussian noise model.
         * @param nodes The interpolated nodes, at least one; their inner keys must differ.
         * @param cache_noise_model See InterpolationFactor.
         * @throws std::invalid_argument for a null inner factor, no nodes, overlapping inner
         *         keys, a passed-through inner key that is also a bordering node's key, or a
         *         noise model that is not a non-constrained noiseModel::Gaussian.
         */
        MultiInterpolationFactor(const gtsam::NoiseModelFactor::shared_ptr &inner_factor, std::vector<Node> nodes, bool cache_noise_model = true)
            : Base(inner_factor ? inner_factor->noiseModel() : gtsam::SharedNoiseModel()),
              m_inner_factor(inner_factor), m_nodes(std::move(nodes)), m_cache_noise_model(cache_noise_model)
        {
            if (!inner_factor)
                throw std::invalid_argument("MultiInterpolationFactor requires a non-null inner factor.");
            if (m_nodes.empty())
                throw std::invalid_argument("MultiInterpolationFactor requires at least one interpolated node.");
            for (size_t i = 0; i < m_nodes.size(); ++i)
                for (const gtsam::Key key : m_nodes[i].innerKeys())
                    for (size_t j = i + 1; j < m_nodes.size(); ++j)
                        if (m_nodes[j].ownsInnerKey(key))
                            throw std::invalid_argument("MultiInterpolationFactor: two nodes use the same inner key; rename their placeholders.");
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(inner_factor->noiseModel());
            if (!gaussian || gaussian->isConstrained())
                throw std::invalid_argument("MultiInterpolationFactor: the inner factor's noise model must be a non-constrained noiseModel::Gaussian.");
            m_base_covariance = gaussian->covariance();

            std::unordered_set<gtsam::Key> key_set;
            for (const Node &node : m_nodes)
                for (const gtsam::Key key : node.cornerKeys())
                    if (key_set.insert(key).second)
                        this->keys_.push_back(key);
            for (const gtsam::Key key : inner_factor->keys())
                if (ownerOf(key) < 0)
                {
                    if (!key_set.insert(key).second)
                        throw std::invalid_argument("MultiInterpolationFactor: a passed-through inner key is also a bordering node's key.");
                    this->keys_.push_back(key);
                }
            for (size_t i = 0; i < this->keys_.size(); ++i)
                m_key_to_index[this->keys_[i]] = i;
        }

        /// The interpolated nodes.
        const std::vector<Node> &nodes() const { return m_nodes; }

        /// The inner factor.
        const gtsam::NoiseModelFactor::shared_ptr &innerFactor() const { return m_inner_factor; }

        /// Print the query points, this factor's keys and the inner factor.
        void print(const std::string &s = "", const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
        {
            std::cout << s << "MultiInterpolationFactor at";
            for (const Node &node : m_nodes)
                std::cout << " (t=" << node.time() << ", s=" << node.arclength() << ")";
            std::cout << " on ";
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
        /// The inner factor's unwhitened error on the interpolated nodes, with Jacobians w.r.t. this factor's keys.
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

        /// Linearize, whitening with the augmented covariance at `x` (cached for error() when cache_noise_model is set).
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

        /// Noise-model loss of the error, using the cached augmented noise model when available.
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
        /// Index of the node owning inner key `key`, or -1 for a passed-through key.
        int ownerOf(gtsam::Key key) const
        {
            for (size_t i = 0; i < m_nodes.size(); ++i)
                if (m_nodes[i].ownsInnerKey(key))
                    return static_cast<int>(i);
            return -1;
        }

        /**
         * @brief Interpolates every node, evaluates the inner factor on them, and
         *        backpropagates its Jacobians to the bordering keys.
         * @param covariance Optional output: the augmented error-space covariance.
         */
        gtsam::Vector evaluate(const gtsam::Values &values, gtsam::OptionalMatrixVecType H, gtsam::Matrix *covariance) const
        {
            const bool need_jacobians = H != nullptr || covariance != nullptr;
            const size_t n = m_nodes.size();
            std::vector<typename Node::Jacobians> jacobians(n);
            std::vector<Eigen::MatrixXd> P(n);
            std::vector<std::shared_ptr<typename Node::ValuesType>> interpolated(n);
            for (size_t i = 0; i < n; ++i)
                interpolated[i] = m_nodes[i].evaluate(values, H ? &jacobians[i] : nullptr, covariance ? &P[i] : nullptr);

            const gtsam::KeyVector &inner_keys = m_inner_factor->keys();
            std::vector<int> owners(inner_keys.size());
            gtsam::Values inner_values;
            for (size_t k = 0; k < inner_keys.size(); ++k)
            {
                owners[k] = ownerOf(inner_keys[k]);
                if (owners[k] >= 0)
                    m_nodes[owners[k]].insert(inner_keys[k], *interpolated[owners[k]], inner_values);
                else
                    inner_values.insert(inner_keys[k], values.at(inner_keys[k]));
            }

            std::vector<gtsam::Matrix> H_inner(inner_keys.size());
            const gtsam::Vector error = need_jacobians ? m_inner_factor->unwhitenedError(inner_values, &H_inner)
                                                       : m_inner_factor->unwhitenedError(inner_values);
            if (!need_jacobians)
                return error;

            const int error_dim = static_cast<int>(error.size());
            std::vector<Eigen::MatrixXd> G;
            if (covariance)
                G.assign(n, Eigen::MatrixXd::Zero(error_dim, Node::node_dim));
            if (H)
                H->assign(size(), Eigen::MatrixXd());

            const auto index_of = [this](gtsam::Key key) { return m_key_to_index.at(key); };
            for (size_t k = 0; k < inner_keys.size(); ++k)
            {
                if (owners[k] >= 0)
                    m_nodes[owners[k]].chain(inner_keys[k], H_inner[k], jacobians[owners[k]], index_of, H, covariance ? &G[owners[k]] : nullptr);
                else if (H)
                    (*H)[index_of(inner_keys[k])] = H_inner[k];
            }
            if (H)
                for (size_t i = 0; i < n; ++i)
                    Node::fillUntouched(jacobians[i], index_of, error_dim, *H);

            if (covariance)
            {
                *covariance = m_base_covariance;
                for (size_t i = 0; i < n; ++i)
                    if (P[i].size() > 0)
                        *covariance += G[i] * P[i] * G[i].transpose();
            }
            return error;
        }

        gtsam::NoiseModelFactor::shared_ptr m_inner_factor;
        std::vector<Node> m_nodes;
        bool m_cache_noise_model;
        gtsam::Matrix m_base_covariance;
        /// The augmented noise model of the latest linearize() (see cache_noise_model).
        mutable gtsam::noiseModel::Gaussian::shared_ptr m_cached_noise_model;
        mutable std::mutex m_cached_noise_model_mutex;
        std::unordered_map<gtsam::Key, size_t> m_key_to_index;
    };
} // namespace Spacetime::Priors

#endif // MULTIINTERPOLATIONFACTOR_HPP
