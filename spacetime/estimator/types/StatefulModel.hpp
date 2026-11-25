#ifndef STATEFULMODEL_HPP
#define STATEFULMODEL_HPP

#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Model.hpp>
#include <spacetime/estimator/types/NearestNodeFactor.hpp>
#include <spacetime/estimator/types/States.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime::Types
{
    template <StateType StateT = State>
    class StatefulModel;

    /**
     * @brief A model that owns part of the estimated state (e.g. a prior): its extendState
     *        adds the state's nodes (one row per call) and the factors linking them. It also
     *        resolves its state at an arbitrary (time, arclength), both as a factor wrapping
     *        another model's inner factor (getOuterFactor, e.g. for a sensor's measurement)
     *        and as values (query). Only one stateful model may extend a given state (see
     *        Estimator::addModel).
     *
     *        StatefulModel<> is the common base sensors and actuators hold. A stateful model
     *        derives from StatefulModel<StateT> instead, whose state() is typed.
     */
    template <>
    class StatefulModel<State> : public Model
    {
    public:
        /**
         * @param state The state this model extends.
         * @throws std::invalid_argument for a null state.
         */
        explicit StatefulModel(std::shared_ptr<State> state) : mp_state(std::move(state))
        {
            if (!mp_state)
                throw std::invalid_argument("StatefulModel requires a non-null state.");
        }
        StatefulModel(const StatefulModel &) = delete;
        StatefulModel &operator=(const StatefulModel &) = delete;

        /// The state this model extends.
        const std::shared_ptr<State> &state() const { return mp_state; }

        /// Variables of this model's state that are held fixed (see Types::State::lockedValues).
        const gtsam::Values &lockedValues() const override { return mp_state->lockedValues(); }

        /// Locks this model's state's variables that were locked without a value (see Types::State::resolveLocks).
        void resolveLocks(const gtsam::Values &new_values) override { mp_state->resolveLocks(new_values); }

        /**
         * @brief Phase 1 (see Types::Model::extendState): adds the state's new variables up to
         *        `time`, with their initial guesses in `new_values`, and returns the factors
         *        that come with them.
         */
        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override = 0;

        /**
         * @brief Build the outer factor that evaluates `inner_factor` -- a fully-formed gtsam
         *        factor built by the caller (typically a sensor, see Types::SensorBase::getFactor)
         *        on whichever well-known "interpolated" keys this model's node type defines (see
         *        Types::InterpolatedKeyMap and, for WNOA specifically, WNOA::InterpolatedKeys) --
         *        against the state at (time, arclength), mirroring gtsam::WnoaInterpFactor's
         *        factor-passing convention.
         *
         * The default snaps the query to the nearest node (see State::getNearestNode: nearest
         * arclength first, then nearest time) and evaluates `inner_factor` on that node
         * directly (see Types::NearestNodeFactor), so its Jacobians are the identity onto it.
         * Models with a real interpolation scheme override this (e.g. WNOA::Prior, through
         * Priors::InterpolationFactor), and should override query() to match.
         *
         * @return nullptr if the query cannot be resolved: by the default, when it lies outside
         *         the estimated state's extent or the node type does not define interpolated
         *         keys (see NodeBase::interpolatedKeys).
         */
        virtual std::shared_ptr<gtsam::NonlinearFactor> getOuterFactor(
            double time,
            double arclength,
            const gtsam::NoiseModelFactor::shared_ptr &inner_factor) const
        {
            const std::shared_ptr<NodeBase> node = nearestNode(time, arclength);
            if (!node)
            {
                warning() << "StatefulModel: cannot resolve (t=" << time << ", s=" << arclength
                          << "): no nearest node supporting interpolated keys within the estimated state's extent.";
                return nullptr;
            }
            return std::make_shared<NearestNodeFactor>(inner_factor, node->interpolatedKeys());
        }

        /**
         * @brief Wrap an inner factor built on point-indexed interpolated keys (see
         *        Types::InterpolatedKeyAt, e.g. WNOA::InterpolatedKeys::Pose(i)), where point
         *        i is (time, arclengths[i]), plus any other key, which is passed through.
         *
         * The default resolves a factor on a single point: it renames that point's keys
         * back to the plain interpolated keys and wraps the result with getOuterFactor at
         * that point, so it matches a sensor's factor there exactly. A factor spanning
         * several points needs a model that interpolates them jointly (e.g.
         * Priors::GridPrior); the default drops it with a warning. A factor on no point is
         * returned unchanged.
         *
         * @return nullptr if the factor cannot be resolved.
         * @throws std::out_of_range for a point index past `arclengths`.
         */
        virtual std::shared_ptr<gtsam::NonlinearFactor> getMultiPointOuterFactor(
            double time,
            const std::vector<double> &arclengths,
            const gtsam::NoiseModelFactor::shared_ptr &inner_factor) const
        {
            const std::vector<std::size_t> points = pointsOf(*inner_factor, arclengths.size());
            if (points.empty())
                return inner_factor;
            if (points.size() > 1)
            {
                warning() << "StatefulModel: cannot resolve a factor spanning " << points.size()
                          << " arclengths at t=" << time << ": this model only interpolates one point at a time.";
                return nullptr;
            }
            return getOuterFactor(time, arclengths[points.front()], singlePointFactor(inner_factor));
        }

        /**
         * @brief `inner_factor`, whose point-indexed keys all belong to one point, as a factor
         *        on the plain interpolated keys (see placeholderOf), for getOuterFactor.
         */
        static gtsam::NoiseModelFactor::shared_ptr singlePointFactor(const gtsam::NoiseModelFactor::shared_ptr &inner_factor)
        {
            std::vector<std::pair<gtsam::Key, gtsam::Key>> to_plain;
            for (const gtsam::Key key : inner_factor->keys())
                if (isPointKey(key))
                    to_plain.emplace_back(key, placeholderOf(key));
            return std::make_shared<NearestNodeFactor>(inner_factor, to_plain);
        }

        /**
         * @brief Resolve this model's node state at an arbitrary (time, arclength),
         *        interpolating or extrapolating as the estimated state's current extent
         *        requires, so callers never have to track where that extent ends.
         *
         * This is the read-only counterpart to getOuterFactor: the same node selection and
         * the same interpolation/extrapolation math, but evaluated against a caller-supplied
         * gtsam::Values instead of being wrapped into a factor.
         *
         * The default matches the default getOuterFactor: it returns the nearest node's
         * values, with a zero conditional covariance (snapping adds no uncertainty) and that
         * node's own posterior marginal.
         *
         * @param values Values to read the bordering nodes from (e.g.
         *        Estimator::getCurrentValues()).
         * @param conditionalCovariance Optional output, node-sized: the uncertainty the
         *        interpolation/extrapolation itself introduces, treating the bordering
         *        estimated nodes as exact, in the queried node's state perturbation frame.
         * @param posterior Optional posterior covariance of the estimated state (e.g.
         *        EstimatorSnapshot::covariance), from which the bordering nodes' joint
         *        marginal is read.
         * @param posteriorCovariance Optional output, node-sized: the full posterior
         *        covariance at the query point -- the conditional covariance plus the
         *        bordering nodes' joint marginal propagated through the interpolation.
         *        Set to an empty (0 x 0) matrix when `posterior` is null or lacks a
         *        required block.
         * @return nullptr if the query cannot be resolved (for the default, see
         *         getOuterFactor), or when a required bordering node is absent from `values`
         *         (e.g. already marginalized out by a fixed-lag smoother).
         */
        virtual std::shared_ptr<NodeValuesBase> query(
            double time,
            double arclength,
            const gtsam::Values &values,
            Eigen::MatrixXd *conditionalCovariance = nullptr,
            const PosteriorCovariance *posterior = nullptr,
            Eigen::MatrixXd *posteriorCovariance = nullptr) const
        {
            const std::shared_ptr<NodeBase> node = nearestNode(time, arclength);
            if (!node)
                return nullptr;

            // Node keys in tangent-vector order, so the covariances are laid out like the node.
            gtsam::KeyVector keys;
            size_t node_dim = 0;
            for (const auto &[interpolated_key, key] : node->interpolatedKeys())
            {
                if (!values.exists(key))
                    return nullptr;
                keys.push_back(key);
                node_dim += values.at(key).dim();
            }

            std::shared_ptr<NodeValuesBase> result = node->getValuesBase(values);
            if (!result)
                return nullptr;

            if (conditionalCovariance)
                *conditionalCovariance = Eigen::MatrixXd::Zero(node_dim, node_dim);
            if (posteriorCovariance && (!posterior || !posterior->joint(keys, *posteriorCovariance)))
                posteriorCovariance->resize(0, 0);
            return result;
        }

    protected:
        std::shared_ptr<State> mp_state; ///< The state this model extends; never null.

    private:
        /// The node the default getOuterFactor/query snap (time, arclength) to, or nullptr
        /// if there is none or its node type does not define interpolated keys.
        std::shared_ptr<NodeBase> nearestNode(double time, double arclength) const
        {
            std::shared_ptr<NodeBase> node = mp_state->getNearestNode(arclength, time);
            if (!node || node->interpolatedKeys().empty())
                return nullptr;
            return node;
        }
    };

    /**
     * @brief A stateful model whose state is a `StateT` (see StatefulModel<State>).
     * @tparam StateT The state type this model extends (e.g. a ContinuumRobotState).
     */
    template <StateType StateT>
    class StatefulModel : public StatefulModel<State>
    {
    public:
        /**
         * @param state The state this model extends.
         * @throws std::invalid_argument for a null state.
         */
        explicit StatefulModel(std::shared_ptr<StateT> state) : StatefulModel<State>(std::move(state)) {}

        /// The state this model extends.
        std::shared_ptr<StateT> state() const { return std::static_pointer_cast<StateT>(mp_state); } // only ever set to a StateT, above
    };

    /// Types derived from StatefulModel: models that store and extend a state. @tparam T The candidate model type.
    template <typename T>
    concept StatefulType = std::derived_from<T, StatefulModel<>>;

    /// Shared pointer to the untyped StatefulModel base, as held by sensors and actuators.
    using pStatefulType = std::shared_ptr<StatefulModel<>>;
} // namespace Spacetime::Types
#endif // STATEFULMODEL_HPP
