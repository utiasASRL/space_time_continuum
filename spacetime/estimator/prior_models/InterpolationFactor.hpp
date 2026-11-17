#ifndef INTERPOLATIONFACTOR_HPP
#define INTERPOLATIONFACTOR_HPP

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtsam/inference/Key.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/States.hpp>
#include <spacetime/estimator/prior_models/InterpolatedNode.hpp>

namespace Spacetime::Priors
{
    /**
     * @class InterpolationFactor
     * @brief Prior-agnostic wrapper that evaluates an inner gtsam::NoiseModelFactor on a
     *        node interpolated (in arclength and/or time) from up to four bordering
     *        estimated nodes -- gtsam::WnoaInterpFactor generalized from its fixed
     *        (pose, velocity) node to any prior-defined set of interpolated sub-states.
     *
     * The caller (typically a sensor, see Types::SensorBase::getFactor) supplies an inner
     * factor defined on the well-known "interpolated" keys of
     * `InterpolatorT::InterpolatedKeys` (see Types::InterpolatedKeyMap), plus optionally any
     * genuinely-outer keys it depends on directly, which are passed through unchanged. This
     * factor evaluates the inner factor on the interpolated node, chains its Jacobians
     * through the interpolation to the bordering (outer) keys, and, unless
     * `fixed_noise_model` is set, adds the interpolation's conditional covariance to the
     * inner noise model (mirroring WnoaInterpFactor::computeInterpolatedError /
     * WnoaInterpFactor::getInterpolatedNoiseModel).
     *
     * All prior-specific math is delegated to `InterpolatorT`; this class never names a
     * sub-state (pose, strain, velocity, ...). `InterpolatedKeys` tells it which keys are
     * interpolated (`keys()`), how to insert a component into a gtsam::Values (`insert`),
     * and which rows of the node_dim-wide per-corner-key Jacobian blocks belong to each key
     * (`rowRange`). `InterpolatorT` must expose `NodeType`, `ValuesType`, `node_dim`,
     * `InterpolatedKeys`, `Plan`, `makePlan(time, arclength, node00, node01, node10, node11)`
     * and `interpolate(plan, values, node00, node01, node10, node11, jacobians,
     * conditionalCovariance)` with Priors::GridInterpolator's nullptr-corner conventions;
     * every GridInterpolator provides them. WNOA's concrete (pose, strain, velocity) layout
     * lives in WNOA::InterpolatedKeys, not here.
     *
     * @tparam InterpolatorT Prior-specific interpolator type (e.g. WNOA::Interpolator<Pose>).
     */
    template <typename InterpolatorT>
    class InterpolationFactor : public gtsam::NoiseModelFactor
    {
    public:
        using NodeType = typename InterpolatorT::NodeType;                   ///< Grid node type of the prior's state.
        using ValuesType = typename InterpolatorT::ValuesType;               ///< Interpolated node values.
        using InterpolatedKeysT = typename InterpolatorT::InterpolatedKeys; ///< Placeholder keys the inner factor is defined on.
        using Base = gtsam::NoiseModelFactor;
        using This = InterpolationFactor<InterpolatorT>;
        using Node = InterpolatedNode<InterpolatorT>; ///< The interpolated query point.

        static_assert(Types::InterpolatedKeyMap<InterpolatedKeysT, ValuesType>,
                      "InterpolatorT::InterpolatedKeys must satisfy Types::InterpolatedKeyMap.");

        /// Tangent dimension of the interpolated node.
        static constexpr int node_dim = InterpolatorT::node_dim;

        // Tell the compiler to import the base class's version of error (see
        // WnoaInterpFactor.h for the identical rationale: without this, the base
        // class's error(const HybridValues&) is hidden by the
        // error(const Values&) defined below).
        using Base::error;

        /**
         * @brief Construct an interpolation wrapper factor.
         *
         * @param inner_factor Factor defined on (a subset of) InterpolatedKeysT::keys(),
         *        and/or any other genuinely-outer keys (looked up directly in the
         *        outer Values passed to this factor). Its noise model becomes this
         *        factor's base noise model, mirroring gtsam::WnoaInterpFactor's
         *        `Base(inner_factor->noiseModel())`.
         * @param interpolator Prior-specific interpolator instance.
         * @param time Query time.
         * @param arclength Query arclength.
         * @param node00 Base bordering node (always required).
         * @param node01 Next-temporal bordering node from node00, or nullptr when `time`
         *        coincides with node00's time or extrapolates past it.
         * @param node10 Next-spatial bordering node from node00, or nullptr when
         *        `arclength` coincides with node00's arclength or extrapolates past it.
         * @param node11 Next-spatial bordering node from node01 (required when the query
         *        interpolates in both axes; see GridInterpolator::interpolate).
         * @param fixed_noise_model If true, do not augment the noise model with the
         *        interpolator's conditional covariance.
         * @param cache_noise_model If true (and the noise model is augmented), error()
         *        reuses the augmented noise model built by the latest linearize() instead
         *        of rebuilding it -- which needs the interpolation Jacobians -- at every
         *        evaluation. The noise model then follows the estimate one linearization
         *        behind, and an error() before any linearize() still builds it. Ignored
         *        when fixed_noise_model is set.
         * @throws std::runtime_error for a null inner_factor or node00, or corners that
         *         cannot bracket the query (see GridInterpolator::makePlan).
         */
        InterpolationFactor(
            const gtsam::NoiseModelFactor::shared_ptr &inner_factor,
            const InterpolatorT &interpolator,
            double time, double arclength,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10,
            const std::shared_ptr<NodeType> &node11,
            bool fixed_noise_model = false,
            bool cache_noise_model = true)
            : Base(inner_factor ? inner_factor->noiseModel() : gtsam::SharedNoiseModel()),
              m_inner_factor(inner_factor),
              m_node(makeNode(interpolator, time, arclength, node00, node01, node10, node11)),
              m_fixed_noise_model(fixed_noise_model),
              m_cache_noise_model(cache_noise_model && !fixed_noise_model)
        {
            if (!inner_factor)
                throw std::runtime_error("InterpolationFactor requires a non-null inner factor.");

            this->keys_ = m_node.cornerKeys();
            std::unordered_set<gtsam::Key> key_set(this->keys_.begin(), this->keys_.end());
            // The inner factor's pass-through keys (e.g. an actuator's input) are this factor's too.
            for (const gtsam::Key key : inner_factor->keys())
                if (!m_node.ownsInnerKey(key) && key_set.insert(key).second)
                    this->keys_.push_back(key);
            for (size_t i = 0; i < this->keys_.size(); ++i)
                m_outer_key_to_index[this->keys_[i]] = i;
        }

        ~InterpolationFactor() override = default;

        /// Print the query point, this factor's keys and the inner factor.
        void print(
            const std::string &s = "",
            const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
        {
            std::cout << s << "InterpolationFactor at (t=" << m_node.time() << ", s=" << m_node.arclength() << ") on ";
            for (const auto &k : this->keys())
                std::cout << keyFormatter(k) << " ";
            std::cout << std::endl;
            m_inner_factor->print("Inner Factor: ", keyFormatter);
        }

        /// Same type, keys and base noise model; the inner factor, query point and corners are not compared.
        bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override
        {
            const This *e = dynamic_cast<const This *>(&expected);
            return e != nullptr && Base::equals(*e, tol);
        }

        using Base::unwhitenedError;
        /**
         * @brief The inner factor's unwhitened error at the interpolated node, with Jacobians w.r.t. this factor's keys.
         * @throws std::runtime_error if the inner factor references a pass-through key absent from `values`.
         */
        gtsam::Vector unwhitenedError(const gtsam::Values &values, gtsam::OptionalMatrixVecType H = nullptr) const override
        {
            return computeInterpolatedError(values, H, nullptr);
        }

        /**
         * @brief Linearize, whitening with the (possibly augmented) noise model and caching it when cache_noise_model is set.
         * @throws std::runtime_error if the noise model must be augmented and the inner
         *         one is not a noiseModel::Gaussian (optionally inside a noiseModel::Robust).
         */
        std::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &x) const override
        {
            if (!active(x))
                return std::shared_ptr<gtsam::JacobianFactor>();

            std::vector<gtsam::Matrix> A(size());
            gtsam::Vector b;
            auto noise_model = eval(x, b, &A);
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

            using gtsam::noiseModel::Constrained;
            if (noiseModel_ && noiseModel_->isConstrained())
                return std::make_shared<gtsam::JacobianFactor>(
                    terms, b, std::static_pointer_cast<Constrained>(noiseModel_)->unit());
            return std::make_shared<gtsam::JacobianFactor>(terms, b);
        }

        /// Noise-model loss of the error, reusing the cached augmented noise model when available.
        double error(const gtsam::Values &c) const override
        {
            if (!active(c))
                return 0.0;

            gtsam::Vector b;
            gtsam::noiseModel::Base::shared_ptr noise_model;
            if (m_cache_noise_model)
            {
                std::lock_guard<std::mutex> lock(m_cached_noise_model_mutex);
                noise_model = m_cached_noise_model;
            }
            if (noise_model)
                b = -computeInterpolatedError(c, nullptr, nullptr); // the mean alone: no Jacobians
            else
                noise_model = eval(c, b, nullptr);
            // The vector overload keeps each component, so a Scalar-reweighted robust
            // model sums its per-row losses; for a Gaussian model it is the usual 0.5 d^2.
            if (noise_model)
                return noise_model->loss(gtsam::Vector(-b));
            return 0.5 * b.squaredNorm();
        }

        /// The factor evaluated on the interpolated node; its noise model is this factor's base noise model.
        const gtsam::NoiseModelFactor::shared_ptr &innerFactor() const { return m_inner_factor; }

        /// Query time of the interpolated node.
        double time() const { return m_node.time(); }

        /// Query arclength of the interpolated node.
        double arclength() const { return m_node.arclength(); }

        /// Whether the noise model is the inner one, without the interpolation's conditional covariance.
        bool fixedNoiseModel() const { return m_fixed_noise_model; }

        /**
         * @brief G, the inner factor's error Jacobian w.r.t. the interpolated node
         *        (error_dim x node_dim, in the interpolator's node perturbation frame).
         *        Unless fixedNoiseModel(), linearize() whitens with the inner covariance
         *        plus G Pcheck G^T, Pcheck being the node's conditional covariance.
         */
        Eigen::MatrixXd innerJacobian(const gtsam::Values &values) const
        {
            Eigen::MatrixXd G;
            computeInterpolatedError(values, nullptr, nullptr, &G);
            return G;
        }

    private:
        /**
         * @brief Core routine mirroring WnoaInterpFactor::computeInterpolatedError:
         *        builds the interpolated node, evaluates the inner factor on it (plus
         *        any pass-through outer keys it references directly), and
         *        backpropagates its Jacobians to this factor's outer (bordering-node)
         *        keys via the interpolator's own per-corner-key Jacobian blocks.
         *
         * @param H Optional output Jacobians, one per key in keys() order.
         * @param conditionalCovarianceInnerSpace Optional output: the interpolation's
         *        conditional covariance mapped into the inner factor's error space,
         *        G Pcheck G^T with G the inner Jacobian w.r.t. the interpolated node.
         * @param innerJacobian Optional output: that G (error_dim x node_dim).
         */
        gtsam::Vector computeInterpolatedError(
            const gtsam::Values &values, gtsam::OptionalMatrixVecType H,
            Eigen::MatrixXd *conditionalCovarianceInnerSpace, Eigen::MatrixXd *innerJacobian = nullptr) const
        {
            const bool need_G = (conditionalCovarianceInnerSpace != nullptr) || (innerJacobian != nullptr);
            const bool need_interp_jac = (H != nullptr) || need_G;

            typename Node::Jacobians jacobians; // corner key -> node_dim x (that key's own dim)
            Eigen::MatrixXd conditionalCovarianceNode; // node_dim x node_dim

            const std::shared_ptr<ValuesType> interpolated = m_node.evaluate(
                values, need_interp_jac ? &jacobians : nullptr,
                conditionalCovarianceInnerSpace ? &conditionalCovarianceNode : nullptr);

            // Build the Values the inner factor is evaluated on: the interpolated
            // node's components at their inner keys (via InterpolatedKeysT's own
            // insert(), which alone knows what those components are), plus (for any
            // other key the inner factor references) a direct pass-through copy from
            // the outer values -- mirroring WnoaInterpFactor::computeInterpolatedError's
            // values_inner construction.
            const gtsam::KeyVector &inner_keys = m_inner_factor->keys();
            gtsam::Values values_inner;
            for (const gtsam::Key key : inner_keys)
            {
                if (m_node.ownsInnerKey(key))
                    m_node.insert(key, *interpolated, values_inner);
                else
                {
                    auto it = values.find(key);
                    if (it == values.end())
                        throw std::runtime_error(
                            "InterpolationFactor: inner factor references a key that is neither "
                            "an interpolated key nor present in the outer values.");
                    values_inner.insert(key, it->value);
                }
            }

            std::vector<gtsam::Matrix> H_inner_storage;
            gtsam::OptionalMatrixVecType H_inner = nullptr;
            if (need_interp_jac)
            {
                H_inner_storage.resize(inner_keys.size());
                H_inner = &H_inner_storage;
            }

            gtsam::Vector error = H_inner
                                       ? m_inner_factor->unwhitenedError(values_inner, H_inner)
                                       : m_inner_factor->unwhitenedError(values_inner);

            if (!need_interp_jac)
                return error;

            const int error_dim = static_cast<int>(error.size());
            Eigen::MatrixXd G; // error_dim x node_dim, assembled only if it or the inner-space covariance is requested
            if (need_G)
                G = Eigen::MatrixXd::Zero(error_dim, node_dim);

            if (H)
                H->assign(size(), Eigen::MatrixXd());

            const auto index_of = [this](gtsam::Key key) { return m_outer_key_to_index.at(key); };

            // Loop through inner keys and, for each, either (a) chain-rule its
            // Jacobian block through the interpolator's per-corner-key Jacobians
            // (interpolated keys, see InterpolatedNode::chain), or (b) map it
            // directly onto the matching outer key (pass-through keys) -- mirroring
            // WnoaInterpFactor's inner-key mapping loop, generalized to an arbitrary
            // number of interpolated sub-keys of arbitrary width.
            for (size_t i = 0; i < inner_keys.size(); ++i)
            {
                const gtsam::Key key = inner_keys[i];
                const Eigen::MatrixXd &Jinner = (*H_inner)[i];

                if (m_node.ownsInnerKey(key))
                    m_node.chain(key, Jinner, jacobians, index_of, H, need_G ? &G : nullptr);
                else if (H)
                {
                    auto it = m_outer_key_to_index.find(key);
                    if (it == m_outer_key_to_index.end())
                        throw std::runtime_error(
                            "InterpolationFactor: inner factor references an outer key not tracked "
                            "by this factor's keys().");
                    if ((*H)[it->second].size() == 0)
                        (*H)[it->second] = Jinner;
                    else
                        (*H)[it->second] += Jinner;
                }
            }

            // Any outer (corner) key the inner factor never touched still needs a
            // properly-sized zero Jacobian block.
            if (H)
                Node::fillUntouched(jacobians, index_of, error_dim, *H);

            if (conditionalCovarianceInnerSpace)
                *conditionalCovarianceInnerSpace = G * conditionalCovarianceNode * G.transpose();
            if (innerJacobian)
                *innerJacobian = std::move(G);

            return error;
        }

        /// Compute residual/noise model, augmenting the base noise model with the
        /// interpolation conditional covariance unless `m_fixed_noise_model` is set.
        /// A noiseModel::Robust is unwrapped, its Gaussian augmented, and rewrapped with
        /// the same robust loss.
        gtsam::noiseModel::Base::shared_ptr eval(const gtsam::Values &values, gtsam::Vector &b,
                                                 std::vector<gtsam::Matrix> *A) const
        {
            if (m_fixed_noise_model)
            {
                b = -computeInterpolatedError(values, A, nullptr);
                return Base::noiseModel();
            }

            Eigen::MatrixXd conditionalCovarianceInnerSpace;
            b = -computeInterpolatedError(values, A, &conditionalCovarianceInnerSpace);

            auto robust_noise_model = std::dynamic_pointer_cast<gtsam::noiseModel::Robust>(Base::noiseModel());
            auto base_noise_model = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
                robust_noise_model ? robust_noise_model->noise() : Base::noiseModel());
            if (!base_noise_model)
                throw std::runtime_error("InterpolationFactor: the inner factor's noise model must be noiseModel::Gaussian "
                                         "(or a derivative), optionally wrapped in a noiseModel::Robust.");

            auto augmented = gtsam::noiseModel::Gaussian::Covariance(base_noise_model->covariance() + conditionalCovarianceInnerSpace);
            if (robust_noise_model)
                return gtsam::noiseModel::Robust::Create(robust_noise_model->robust(), augmented);
            return augmented;
        }

        /// Validates node00 with InterpolationFactor's own (runtime_error) message, then builds the node.
        static Node makeNode(const InterpolatorT &interpolator, double time, double arclength,
                             const std::shared_ptr<NodeType> &node00, const std::shared_ptr<NodeType> &node01,
                             const std::shared_ptr<NodeType> &node10, const std::shared_ptr<NodeType> &node11)
        {
            if (!node00)
                throw std::runtime_error("InterpolationFactor requires node00 to be non-null.");
            return Node(interpolator, time, arclength, node00, node01, node10, node11);
        }

        gtsam::NoiseModelFactor::shared_ptr m_inner_factor;
        Node m_node;
        bool m_fixed_noise_model;
        bool m_cache_noise_model;
        /// The augmented noise model of the latest linearize() (see cache_noise_model).
        mutable gtsam::noiseModel::Base::shared_ptr m_cached_noise_model;
        mutable std::mutex m_cached_noise_model_mutex;
        std::unordered_map<gtsam::Key, size_t> m_outer_key_to_index;
    };
} // namespace Spacetime::Priors

#endif // INTERPOLATIONFACTOR_HPP
