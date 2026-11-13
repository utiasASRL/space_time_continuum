#ifndef GRIDINTERPOLATOR_HPP
#define GRIDINTERPOLATOR_HPP

#include <cmath>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/functions.hpp>

namespace Spacetime::Priors
{
    /**
     * @class GridInterpolator
     * @brief Prior-agnostic scaffolding for interpolating a node on the
     *        ContinuumRobotState space-time grid from up to four bordering nodes, shared by
     *        every prior whose interpolation factors per axis (e.g. WNOA::Interpolator,
     *        WNOV::Interpolator).
     *
     * This class owns everything that depends only on the grid's topology and on the
     * node type's key layout: the corner dispatch (exact-match, one-axis and two-stage
     * arclength-then-time cases), per-axis mode classification, splitting the monolithic
     * interpolation Jacobian into per-key blocks, and propagating the corners' posterior
     * marginal. The node's key layout is read generically through
     * NodeBase::interpolatedKeys (tangent order) and `InterpolatedKeysT::rowRange` (see
     * Types::InterpolatedKeyMap), so no sub-state is ever named here.
     *
     * A `Derived` prior supplies only the math along ONE axis, through hooks that this
     * class calls once it has classified the axis (see resolveAxis). The work that
     * depends only on the axis offsets (not on the node values) is split out into an
     * `AxisPlanT`, computed once per query and reusable across evaluations (see Plan):
     *
     *   AxisPlanT planInterpolate(bool space, double delta_i, double delta) const;
     *   AxisPlanT planExtrapolate(bool space, double delta_i) const;
     *   ValuesType resolveInterpolate(const AxisPlanT &plan, const ValuesType &n0,
     *                                 const ValuesType &n1, AxisJacobian *jac, MatrixNode *cov) const;
     *   ValuesType resolveExtrapolate(const AxisPlanT &plan, const ValuesType &n0,
     *                                 AxisJacobian *jac, MatrixNode *cov) const;
     *
     * `delta_i` is the query's offset past n0 along the axis, `delta` the offset of n1.
     * `jac` must be written as node_dim x 2*node_dim, ordered [d/d(n0) | d/d(n1)], of the
     * resolved node's state perturbation with respect to perturbations of the two nodes
     * (the right half zero when extrapolating). `cov` is the conditional covariance
     * assuming the bordering nodes are exact, in the resolved node's state perturbation
     * frame.
     *
     * Dispatch is driven by the QUERY COORDINATES relative to node00's, not by which
     * corners happen to be null: per axis, the query either coincides with node00
     * (offset below TOLERANCE), interpolates (a bordering node past it was supplied), or
     * extrapolates (none was). A caller may therefore supply a corner that turns out to
     * be irrelevant -- it simply contributes a zero Jacobian block (see
     * backfillMissingJacobianKeys, which Priors::InterpolationFactor depends on, since it
     * derives its keys() from every corner it was constructed with).
     *
     * Extrapolation is defined FORWARD only (delta >= 0): the anchor node is always the
     * last one at or before the query (see GridPrior::selectCorners), and a backwards
     * query with no bordering node throws (see classifyAxis).
     *
     * @tparam Derived The prior-specific interpolator (CRTP), providing the two hooks.
     * @tparam NodeT The prior's node type, providing `dim`, `ValuesType`,
     *         `getValues(const gtsam::Values &)` and `interpolatedKeys()`.
     * @tparam InterpolatedKeysT The node type's interpolated-key layout (see
     *         Types::InterpolatedKeyMap).
     * @tparam AxisPlanT Derived's per-axis plan (default-constructible), see above.
     */
    template <typename Derived, typename NodeT, typename InterpolatedKeysT, typename AxisPlanT>
    class GridInterpolator
    {
    public:
        using NodeType = NodeT;
        using ValuesType = typename NodeT::ValuesType;
        /// Exposed for Priors::InterpolationFactor (see Types::InterpolatedKeyMap).
        using InterpolatedKeys = InterpolatedKeysT;

        static constexpr int node_dim = NodeT::dim;
        using MatrixNode = Eigen::Matrix<double, node_dim, node_dim>;
        /// Jacobian of one axis stage, [d/d(n0) | d/d(n1)] (see resolveAxis).
        using AxisJacobian = Eigen::Matrix<double, node_dim, 2 * node_dim>;

        static_assert(Types::InterpolatedKeyMap<InterpolatedKeysT, ValuesType>,
                      "InterpolatedKeysT must satisfy Types::InterpolatedKeyMap.");

        /**
         * @brief How a query coordinate relates to its anchor node along one axis.
         *  - Coincide: the query already lies on the anchor's coordinate.
         *  - Interpolate: a bordering node past the query exists on this axis.
         *  - Extrapolate: no bordering node exists; the query is past the last one.
         */
        enum class AxisMode
        {
            Coincide,
            Interpolate,
            Extrapolate
        };

        /// One axis stage of a planned query: how it resolves and, unless it coincides, Derived's plan for it.
        struct AxisStage
        {
            AxisMode mode = AxisMode::Coincide;
            AxisPlanT plan{};
        };

        /**
         * @brief Everything about a query at (time, arclength) against a given set of
         *        corners that does not depend on the corners' values: each axis's mode
         *        and the plan of every axis stage the query resolves through (see
         *        makePlan). Computing it once lets a caller that evaluates the same query
         *        repeatedly (e.g. Priors::InterpolationFactor, at every linearization)
         *        skip that work.
         *
         * Stages, by shape (see interpolateCorners): a one-axis query uses `space0` or
         * `time`; a query interpolating in both axes uses `space0` (between node00 and
         * node10), `space1` (between node01 and node11) and `time`; the other two-axis
         * shapes use `space0` and `time`.
         */
        struct Plan
        {
            AxisMode space_mode = AxisMode::Coincide;
            AxisMode time_mode = AxisMode::Coincide;
            AxisStage space0, space1, time;
        };

        /**
         * @brief Interpolate a single node in time between two bordering nodes at the same
         *        arclength.
         *
         * @param t_k Left border time.
         * @param t_kp1 Right border time.
         * @param time Query time, t_k <= time <= t_kp1.
         * @param node_k Left border node values.
         * @param node_kp1 Right border node values.
         * @param jac Optional output Jacobian (node_dim x 2*node_dim), ordered
         *        [d/d(node_k) | d/d(node_kp1)].
         * @param conditionalCov Optional output conditional covariance (node_dim x
         *        node_dim), assuming node_k/node_kp1 are known exactly, in the
         *        interpolated node's state perturbation frame.
         */
        ValuesType interpolateTime(
            double t_k, double t_kp1, double time,
            const ValuesType &node_k, const ValuesType &node_kp1,
            gtsam::OptionalMatrixType jac = nullptr,
            MatrixNode *conditionalCov = nullptr) const
        {
            AxisJacobian axis_jac;
            ValuesType result = resolveAxis(/*space=*/false, time - t_k, t_kp1 - t_k,
                                            node_k, &node_kp1, jac ? &axis_jac : nullptr, conditionalCov);
            if (jac)
                *jac = axis_jac;
            return result;
        }

        /**
         * @brief Interpolate a single node in arclength between two bordering nodes at
         *        the same time. Symmetric counterpart to interpolateTime.
         */
        ValuesType interpolateSpace(
            double s_n, double s_np1, double arclength,
            const ValuesType &node_n, const ValuesType &node_np1,
            gtsam::OptionalMatrixType jac = nullptr,
            MatrixNode *conditionalCov = nullptr) const
        {
            AxisJacobian axis_jac;
            ValuesType result = resolveAxis(/*space=*/true, arclength - s_n, s_np1 - s_n,
                                            node_n, &node_np1, jac ? &axis_jac : nullptr, conditionalCov);
            if (jac)
                *jac = axis_jac;
            return result;
        }

        /**
         * @brief Extrapolate a single node forward in time past the last bordering node at
         *        the same arclength, conditioned on that anchor node alone.
         *
         * @param t_anchor Anchor (last bordering) node's time.
         * @param time Query time; must satisfy time >= t_anchor -- extrapolation is
         *        defined forward only.
         * @param node_anchor Anchor node values.
         * @param jac Optional output Jacobian (node_dim x node_dim -- ONE node, unlike
         *        interpolateTime's node_dim x 2*node_dim) with respect to a perturbation
         *        of the anchor.
         * @param conditionalCov Optional output conditional covariance (node_dim x
         *        node_dim), assuming node_anchor is known exactly, in the extrapolated
         *        node's state perturbation frame.
         */
        ValuesType extrapolateTime(
            double t_anchor, double time, const ValuesType &node_anchor,
            gtsam::OptionalMatrixType jac = nullptr,
            MatrixNode *conditionalCov = nullptr) const
        {
            return extrapolateAxis(/*space=*/false, time - t_anchor, node_anchor, jac, conditionalCov);
        }

        /**
         * @brief Extrapolate a single node forward in arclength past the last bordering
         *        node at the same time. Symmetric counterpart to extrapolateTime.
         */
        ValuesType extrapolateSpace(
            double s_anchor, double arclength, const ValuesType &node_anchor,
            gtsam::OptionalMatrixType jac = nullptr,
            MatrixNode *conditionalCov = nullptr) const
        {
            return extrapolateAxis(/*space=*/true, arclength - s_anchor, node_anchor, jac, conditionalCov);
        }

        /**
         * @brief Top-level interpolation entry point, dispatching between the
         *        exact-match, arclength-only, time-only, and full 2D
         *        (arclength-then-time) cases based on the query coordinates and which
         *        corner nodes are supplied (see GridPrior::selectCorners).
         *
         * Preconditions (matching GridPrior::selectCorners, not re-validated here):
         *  - node00 is always required.
         *  - node01 (the next temporal node from node00) must be supplied whenever
         *    `time` does not already coincide with node00's time, unless the query
         *    extrapolates in time.
         *  - node10 (the next spatial node from node00) must be supplied whenever
         *    `arclength` does not already coincide with node00's arclength, unless the
         *    query extrapolates in arclength.
         *  - node11 (the next spatial node from node01, equivalently the next
         *    temporal node from node10) must be supplied whenever both node01 and
         *    node10 are supplied.
         *
         * @param jacobians Optional output map from each present corner node's
         *        individual keys to the corresponding node_dim x (that key's dim)
         *        Jacobian block of the interpolated node with respect to a perturbation
         *        of that key.
         * @param conditionalCovariance Optional output conditional covariance
         *        (node_dim x node_dim) of the interpolated node, assuming all supplied
         *        corner nodes are known exactly, in its state perturbation frame.
         * @param posterior Optional posterior covariance of the estimated state, from
         *        which the corners' joint marginal P_c is read (see
         *        EstimatorSnapshot::covariance).
         * @param posteriorCovariance Optional output posterior covariance (node_dim x
         *        node_dim) of the interpolated node, Phat = Pcheck + J P_c J^T, where J
         *        stacks the per-key Jacobians of every corner the query depends on
         *        (paper eqs. (36) and (38)). Set to an empty (0 x 0) matrix when
         *        `posterior` is null or lacks a required block (e.g. a marginalized
         *        corner); the mean is returned regardless.
         */
        std::shared_ptr<Types::NodeValues> interpolate(
            double time, double arclength, const gtsam::Values &values,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10,
            const std::shared_ptr<NodeType> &node11,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> *jacobians = nullptr,
            Eigen::MatrixXd *conditionalCovariance = nullptr,
            const PosteriorCovariance *posterior = nullptr,
            Eigen::MatrixXd *posteriorCovariance = nullptr) const
        {
            return interpolate(makePlan(time, arclength, node00, node01, node10, node11), values,
                               node00, node01, node10, node11, jacobians, conditionalCovariance,
                               posterior, posteriorCovariance);
        }

        /**
         * @brief The value-independent part of interpolate()'s query at (time, arclength)
         *        against these corners (see Plan), validating the corners as interpolate()
         *        does. Only the corners' coordinates are read, not their values.
         */
        Plan makePlan(
            double time, double arclength,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10,
            const std::shared_ptr<NodeType> &node11) const
        {
            if (!node00)
                throw std::runtime_error("Interpolator::interpolate requires node00 to be non-null.");
            if (node10 && node10->arclength <= node00->arclength + TOLERANCE)
                throw std::runtime_error(
                    "Interpolator::interpolate: node10 must lie strictly past node00 in arclength.");
            if (node01 && node01->time <= node00->time + TOLERANCE)
                throw std::runtime_error(
                    "Interpolator::interpolate: node01 must lie strictly past node00 in time.");

            Plan plan;
            plan.space_mode = classifyAxis(arclength - node00->arclength, node10 != nullptr);
            plan.time_mode = classifyAxis(time - node00->time, node01 != nullptr);
            const bool space_active = plan.space_mode != AxisMode::Coincide;
            const bool time_active = plan.time_mode != AxisMode::Coincide;

            if (space_active && !time_active)
                plan.space0 = makeStage(/*space=*/true, arclength - node00->arclength,
                                        node10 ? node10->arclength - node00->arclength : 0.0, node10 != nullptr);
            else if (time_active && !space_active)
                plan.time = makeStage(/*space=*/false, time - node00->time,
                                      node01 ? node01->time - node00->time : 0.0, node01 != nullptr);
            else if (plan.space_mode == AxisMode::Interpolate && plan.time_mode == AxisMode::Interpolate)
            {
                if (!node11)
                    throw std::runtime_error(
                        "Interpolator::interpolate requires node11 when interpolating in both arclength and time.");
                plan.space0 = makeStage(/*space=*/true, arclength - node00->arclength,
                                        node10->arclength - node00->arclength, /*has_border=*/true);
                plan.space1 = makeStage(/*space=*/true, arclength - node01->arclength,
                                        node11->arclength - node01->arclength, /*has_border=*/true);
                plan.time = makeStage(/*space=*/false, time - node00->time,
                                      node01->time - node00->time, /*has_border=*/true);
            }
            else if (plan.time_mode == AxisMode::Interpolate) // arclength extrapolated
            {
                plan.time = makeStage(/*space=*/false, time - node00->time,
                                      node01->time - node00->time, /*has_border=*/true);
                plan.space0 = makeStage(/*space=*/true, arclength - node00->arclength, 0.0, /*has_border=*/false);
            }
            else if (time_active) // time extrapolated
            {
                plan.space0 = makeStage(/*space=*/true, arclength - node00->arclength,
                                        node10 ? node10->arclength - node00->arclength : 0.0, node10 != nullptr);
                plan.time = makeStage(/*space=*/false, time - node00->time, 0.0, /*has_border=*/false);
            }
            return plan;
        }

        /**
         * @brief interpolate() for a query already planned with makePlan, against the
         *        same corners.
         */
        std::shared_ptr<Types::NodeValues> interpolate(
            const Plan &plan, const gtsam::Values &values,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10,
            const std::shared_ptr<NodeType> &node11,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> *jacobians = nullptr,
            Eigen::MatrixXd *conditionalCovariance = nullptr,
            const PosteriorCovariance *posterior = nullptr,
            Eigen::MatrixXd *posteriorCovariance = nullptr) const
        {
            if (posteriorCovariance && !posterior)
                posteriorCovariance->resize(0, 0);
            if (!posteriorCovariance || !posterior)
                return interpolateCorners(plan, values, node00, node01, node10, node11,
                                          jacobians, conditionalCovariance);

            // The posterior needs both the per-key Jacobians and the conditional covariance,
            // whether or not the caller asked for them.
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> local_jacobians;
            Eigen::MatrixXd local_conditional;
            auto &jac_out = jacobians ? *jacobians : local_jacobians;
            auto &cond_out = conditionalCovariance ? *conditionalCovariance : local_conditional;

            auto result = interpolateCorners(plan, values, node00, node01, node10, node11,
                                             &jac_out, &cond_out);
            propagatePosterior({node00, node01, node10, node11}, jac_out, cond_out, *posterior, *posteriorCovariance);
            return result;
        }

    protected:
        /// Only constructible as the base of a Derived interpolator (CRTP).
        GridInterpolator() = default;

        /// Reject a backwards offset. Extrapolation is forward-only; see classifyAxis.
        static void throwIfBackwards(double delta)
        {
            if (delta < -TOLERANCE)
                throw std::runtime_error(
                    "Interpolator: extrapolation is defined forward only; got a negative offset.");
        }

        /**
         * @brief Classify one axis from the query's offset past the anchor and whether a
         *        bordering node is available past it.
         *
         * Throws for a negative offset: extrapolation in this framework is defined FORWARD
         * only, and a bordering node is always AHEAD of the anchor (validated in
         * makePlan), so it cannot bracket a query that lies behind it either.
         * Callers wanting the state before the earliest node must reject the query instead
         * (see GridPrior::getOuterFactor).
         */
        static AxisMode classifyAxis(double delta_i, bool has_border_node)
        {
            if (std::abs(delta_i) < TOLERANCE)
                return AxisMode::Coincide;
            if (delta_i < 0.0)
                throw std::runtime_error(
                    "Interpolator: query lies before its anchor node. Extrapolation is "
                    "defined forward only, and a bordering node ahead of the anchor cannot "
                    "bracket it.");
            if (has_border_node)
                return AxisMode::Interpolate;
            return AxisMode::Extrapolate;
        }

        /**
         * @brief The single per-axis resolver behind interpolateTime/interpolateSpace and
         *        extrapolateTime/extrapolateSpace: plans the axis (see makeStage) and
         *        resolves it (see resolveStage).
         *
         * `n1 == nullptr` selects extrapolation, and `delta` is then unused.
         *
         * @param jac Optional output Jacobian, always node_dim x 2*node_dim; its right half
         *        is identically zero in the Coincide and Extrapolate modes.
         */
        ValuesType resolveAxis(
            bool space, double delta_i, double delta,
            const ValuesType &n0, const ValuesType *n1,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            return resolveStage(makeStage(space, delta_i, delta, n1 != nullptr), n0, n1, jac, cov);
        }

        /**
         * @brief Classify one axis stage and, unless it coincides, have Derived plan it.
         *
         * Interpolation formulas are derived for a query BETWEEN the two bordering nodes.
         * Past the far one they silently return something that is not the correct
         * conditional, so it is rejected rather than answered wrongly -- the caller wants
         * extrapolation off the far node instead (which is what GridPrior::selectCorners
         * produces, by anchoring on the last node at or before the query and leaving the
         * border null).
         */
        AxisStage makeStage(bool space, double delta_i, double delta, bool has_border) const
        {
            AxisStage stage;
            stage.mode = classifyAxis(delta_i, has_border);
            if (stage.mode == AxisMode::Interpolate && delta_i > delta + TOLERANCE)
                throw std::runtime_error(
                    "Interpolator: query lies past the far bordering node; interpolation "
                    "is only defined between the two, and extrapolation must be anchored on the "
                    "far node with no border supplied.");
            if (stage.mode == AxisMode::Interpolate)
                stage.plan = derived().planInterpolate(space, delta_i, delta);
            else if (stage.mode == AxisMode::Extrapolate)
                stage.plan = derived().planExtrapolate(space, delta_i);
            return stage;
        }

        /**
         * @brief Resolve one planned axis stage from its bordering node values: handles
         *        the Coincide case itself and hands the other two to Derived's hooks.
         *        `n1` is required when the stage interpolates and ignored otherwise.
         */
        ValuesType resolveStage(
            const AxisStage &stage,
            const ValuesType &n0, const ValuesType *n1,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            if (stage.mode == AxisMode::Coincide)
            {
                if (jac)
                {
                    jac->setZero();
                    jac->template leftCols<node_dim>().setIdentity();
                }
                if (cov)
                    cov->setZero();
                return n0;
            }

            if (stage.mode == AxisMode::Extrapolate)
                return derived().resolveExtrapolate(stage.plan, n0, jac, cov);
            return derived().resolveInterpolate(stage.plan, n0, *n1, jac, cov);
        }

    private:
        /// CRTP downcast to the prior-specific interpolator providing the hooks.
        const Derived &derived() const { return static_cast<const Derived &>(*this); }

        /**
         * @brief Shared body of extrapolateTime/extrapolateSpace: resolveAxis with no
         *        bordering node, narrowing the emitted Jacobian from the internal
         *        node_dim x 2*node_dim form to the single-node node_dim x node_dim one
         *        (the discarded right half is identically zero).
         */
        ValuesType extrapolateAxis(
            bool space, double delta_i, const ValuesType &node_anchor,
            gtsam::OptionalMatrixType jac,
            MatrixNode *conditionalCov) const
        {
            AxisJacobian wide_jac;
            ValuesType result = resolveAxis(space, delta_i, /*delta=*/0.0, node_anchor, nullptr,
                                            jac ? &wide_jac : nullptr, conditionalCov);
            if (jac)
                *jac = wide_jac.template leftCols<node_dim>();
            return result;
        }

        /// The corner dispatch behind interpolate(), producing the mean, the per-key
        /// Jacobians and the conditional covariance from a query planned by makePlan
        /// against the same corners. See interpolate() for the contract.
        std::shared_ptr<Types::NodeValues> interpolateCorners(
            const Plan &plan, const gtsam::Values &values,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10,
            const std::shared_ptr<NodeType> &node11,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> *jacobians,
            Eigen::MatrixXd *conditionalCovariance) const
        {
            if (!node00)
                throw std::runtime_error("Interpolator::interpolate requires node00 to be non-null.");
            const AxisMode space_mode = plan.space_mode;
            const AxisMode time_mode = plan.time_mode;

            // The covariance combination in the two-stage cases needs the stage Jacobians to
            // propagate stage 1's conditional covariance through stage 2, even when the caller
            // did not ask for the per-key Jacobian map -- so they must be computed whenever
            // EITHER output is requested.
            const bool need_jac = (jacobians != nullptr) || (conditionalCovariance != nullptr);

            const ValuesType v00 = node00->getValues(values);

            // --- Shape 1: the query already sits on node00 in both axes. Corners may still
            // have been supplied (a caller need not know the query would land exactly on a
            // node); they simply contribute nothing, and backfill gives them zero blocks.
            if (space_mode == AxisMode::Coincide && time_mode == AxisMode::Coincide)
            {
                if (jacobians)
                {
                    assignIdentityJacobian(node00, *jacobians);
                    backfillMissingJacobianKeys({node01, node10, node11}, *jacobians);
                }
                if (conditionalCovariance)
                    conditionalCovariance->setZero(node_dim, node_dim);
                return std::make_shared<ValuesType>(v00);
            }

            // --- Shape 2: exactly one axis is active -> a single stage, which interpolates
            // or extrapolates along it depending on whether that axis's bordering node was
            // supplied.
            if (space_mode == AxisMode::Coincide || time_mode == AxisMode::Coincide)
            {
                const bool space = (time_mode == AxisMode::Coincide);
                const std::shared_ptr<NodeType> &border = space ? node10 : node01;

                ValuesType border_values;
                if (border)
                    border_values = border->getValues(values);

                AxisJacobian jac;
                MatrixNode condCov;
                ValuesType result = resolveStage(
                    space ? plan.space0 : plan.time, v00, border ? &border_values : nullptr,
                    jacobians ? &jac : nullptr, conditionalCovariance ? &condCov : nullptr);

                if (jacobians)
                {
                    splitJacobianIntoKeys(jac, {node00, border}, *jacobians);
                    backfillMissingJacobianKeys({node00, node01, node10, node11}, *jacobians);
                }
                if (conditionalCovariance)
                    *conditionalCovariance = condCov;
                return std::make_shared<ValuesType>(result);
            }

            // --- Shape 3: both axes active -> two stages.
            //
            // Ordering rule: resolve the INTERPOLATED axis first and the EXTRAPOLATED axis
            // last, so at most one stage is ever an extrapolation. This matters for the
            // covariance: the two-stage combination below treats the stage-1 results as
            // independent, which is defensible when each is pinned between two estimated
            // nodes, but false for two simultaneous extrapolations off the same boundary at
            // neighbouring coordinates (they share the same unmodelled extension and are
            // strongly correlated, so dropping the cross term would make the result
            // over-confident). Running only one extrapolation, last, avoids the question:
            // its covariance is a single sandwich of stage 1's, with no independence
            // assumption at all.

            // Both interpolate: arclength at each bordering time, then time. This is the
            // original Section II-F.3 / Fig. 4 path from the accompanying paper.
            if (space_mode == AxisMode::Interpolate && time_mode == AxisMode::Interpolate)
            {
                const ValuesType v01 = node01->getValues(values);
                const ValuesType v10 = node10->getValues(values);
                const ValuesType v11 = node11->getValues(values);

                AxisJacobian jac_s0, jac_s1;
                MatrixNode condCov_s0, condCov_s1;
                ValuesType interp_t0 = resolveStage(
                    plan.space0, v00, &v10,
                    need_jac ? &jac_s0 : nullptr, conditionalCovariance ? &condCov_s0 : nullptr);
                ValuesType interp_t1 = resolveStage(
                    plan.space1, v01, &v11,
                    need_jac ? &jac_s1 : nullptr, conditionalCovariance ? &condCov_s1 : nullptr);

                AxisJacobian jac_t;
                MatrixNode condCov_t;
                ValuesType result = resolveStage(
                    plan.time, interp_t0, &interp_t1,
                    need_jac ? &jac_t : nullptr, conditionalCovariance ? &condCov_t : nullptr);

                const MatrixNode H_t0 = jac_t.template leftCols<node_dim>();
                const MatrixNode H_t1 = jac_t.template rightCols<node_dim>();
                if (jacobians)
                {
                    splitJacobianIntoKeys(AxisJacobian(H_t0.lazyProduct(jac_s0)), {node00, node10}, *jacobians);
                    splitJacobianIntoKeys(AxisJacobian(H_t1.lazyProduct(jac_s1)), {node01, node11}, *jacobians);
                }
                if (conditionalCovariance)
                    *conditionalCovariance = condCov_t + sandwich(H_t0, condCov_s0) + sandwich(H_t1, condCov_s1);
                return std::make_shared<ValuesType>(result);
            }

            // Arclength extrapolated, time interpolated: interpolate in time along the
            // bordering arclength first (node00 -> node01), then extrapolate that
            // intermediate node out in arclength.
            if (time_mode == AxisMode::Interpolate)
            {
                const ValuesType v01 = node01->getValues(values);

                AxisJacobian jac_t;
                MatrixNode condCov_t;
                ValuesType mid = resolveStage(
                    plan.time, v00, &v01,
                    need_jac ? &jac_t : nullptr, conditionalCovariance ? &condCov_t : nullptr);

                AxisJacobian jac_s_wide;
                MatrixNode condCov_s;
                ValuesType result = resolveStage(
                    plan.space0, mid, nullptr,
                    need_jac ? &jac_s_wide : nullptr, conditionalCovariance ? &condCov_s : nullptr);
                const MatrixNode jac_s = jac_s_wide.template leftCols<node_dim>();

                if (jacobians)
                {
                    splitJacobianIntoKeys(AxisJacobian(jac_s.lazyProduct(jac_t)), {node00, node01}, *jacobians);
                    backfillMissingJacobianKeys({node00, node01, node10, node11}, *jacobians);
                }
                if (conditionalCovariance)
                    *conditionalCovariance = condCov_s + sandwich(jac_s, condCov_t);
                return std::make_shared<ValuesType>(result);
            }

            // Time extrapolated: resolve the arclength axis first (interpolating between
            // node00 and node10 when node10 exists, otherwise extrapolating off node00),
            // then extrapolate the intermediate node forward in time. This covers both the
            // (space-interpolate, time-extrapolate) and (extrapolate, extrapolate) cells.
            {
                ValuesType v10;
                if (node10)
                    v10 = node10->getValues(values);

                AxisJacobian jac_s;
                MatrixNode condCov_s;
                ValuesType mid = resolveStage(
                    plan.space0, v00, node10 ? &v10 : nullptr,
                    need_jac ? &jac_s : nullptr, conditionalCovariance ? &condCov_s : nullptr);

                AxisJacobian jac_t_wide;
                MatrixNode condCov_t;
                ValuesType result = resolveStage(
                    plan.time, mid, nullptr,
                    need_jac ? &jac_t_wide : nullptr, conditionalCovariance ? &condCov_t : nullptr);
                const MatrixNode jac_t = jac_t_wide.template leftCols<node_dim>();

                if (jacobians)
                {
                    splitJacobianIntoKeys(AxisJacobian(jac_t.lazyProduct(jac_s)), {node00, node10}, *jacobians);
                    backfillMissingJacobianKeys({node00, node01, node10, node11}, *jacobians);
                }
                if (conditionalCovariance)
                    *conditionalCovariance = condCov_t + sandwich(jac_t, condCov_s);
                return std::make_shared<ValuesType>(result);
            }
        }

        /// H C H^T for node_dim x node_dim blocks, with coefficient-based (lazy) products:
        /// at node_dim scale Eigen's blocked GEMM costs more in packing than it saves.
        static MatrixNode sandwich(const MatrixNode &H, const MatrixNode &C)
        {
            const MatrixNode HC = H.lazyProduct(C);
            return HC.lazyProduct(H.transpose());
        }

        /**
         * @brief Phat = Pcheck + J P_c J^T: propagate the corners' joint posterior marginal
         *        P_c through the interpolation Jacobian J and add the conditional covariance.
         *
         * Only keys the query actually depends on (a nonzero Jacobian block) enter J, so a
         * corner that was supplied but does not affect the result (e.g. a query coinciding
         * with node00) does not require its blocks to have been extracted.
         */
        void propagatePosterior(
            std::initializer_list<std::shared_ptr<NodeType>> corners,
            const std::unordered_map<gtsam::Key, Eigen::MatrixXd> &jacobians,
            const Eigen::MatrixXd &conditionalCovariance,
            const PosteriorCovariance &posterior,
            Eigen::MatrixXd &posteriorCovariance) const
        {
            posteriorCovariance.resize(0, 0);

            gtsam::KeyVector keys;
            Eigen::Index total_dim = 0;
            for (const auto &corner : corners)
            {
                if (!corner)
                    continue;
                for (const auto &[interpolated_key, key] : corner->interpolatedKeys())
                {
                    (void)interpolated_key;
                    auto it = jacobians.find(key);
                    if (it != jacobians.end() && !it->second.isZero(0.0))
                    {
                        keys.push_back(key);
                        total_dim += it->second.cols();
                    }
                }
            }

            Eigen::MatrixXd corner_covariance;
            if (!posterior.joint(keys, corner_covariance))
                return;

            Eigen::MatrixXd J(node_dim, total_dim);
            Eigen::Index offset = 0;
            for (const gtsam::Key key : keys)
            {
                const Eigen::MatrixXd &block = jacobians.at(key);
                J.middleCols(offset, block.cols()) = block;
                offset += block.cols();
            }

            const Eigen::MatrixXd P = conditionalCovariance + J * corner_covariance * J.transpose();
            posteriorCovariance = 0.5 * (P + P.transpose());
        }

        /// Assign a node_dim x node_dim identity Jacobian, split per-key, for a query that matches `node` exactly.
        static void assignIdentityJacobian(const std::shared_ptr<NodeType> &node,
                                           std::unordered_map<gtsam::Key, Eigen::MatrixXd> &jacobians)
        {
            for (const auto &[interpolated_key, key] : node->interpolatedKeys())
            {
                const auto [offset, width] = InterpolatedKeysT::rowRange(interpolated_key);
                Eigen::MatrixXd block = Eigen::MatrixXd::Zero(node_dim, width);
                block.block(offset, 0, width, width).setIdentity();
                jacobians[key] = block;
            }
        }

        /**
         * @brief Split a monolithic node_dim x (corners.size()*node_dim) Jacobian (as
         *        produced by interpolateTime/interpolateSpace, or their composition) into
         *        per-key blocks and insert them into the output map, keyed by each corner's
         *        own keys.
         *
         * A nullptr corner is skipped but still consumes its column slot, so callers can
         * pass a fixed corner list (e.g. {node00, node10}) regardless of whether the
         * second corner exists -- in the extrapolation and coincide modes its columns are
         * identically zero anyway (see resolveAxis).
         */
        template <typename JacobianT>
        static void splitJacobianIntoKeys(
            const Eigen::MatrixBase<JacobianT> &jac,
            std::initializer_list<std::shared_ptr<NodeType>> corners,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> &jacobians)
        {
            Eigen::Index i = 0;
            for (const auto &corner : corners)
            {
                if (corner)
                {
                    for (const auto &[interpolated_key, key] : corner->interpolatedKeys())
                    {
                        const auto [offset, width] = InterpolatedKeysT::rowRange(interpolated_key);
                        jacobians[key] = jac.block(0, i * node_dim + offset, node_dim, width);
                    }
                }
                ++i;
            }
        }

        /**
         * @brief Give every supplied corner node an entry in the Jacobian map, filling in
         *        a zero block for any that the resolved query turned out not to depend on.
         *
         * Priors::InterpolationFactor derives its keys() from EVERY non-null corner it was
         * constructed with, and its Jacobian-assembly loop iterates this map -- so a corner
         * key present in keys() but absent here would leave an empty matrix in (*H)[i] and
         * crash linearize(). That happens whenever a corner is supplied but the query turns
         * out to coincide with, or extrapolate from, the other side of its axis.
         */
        static void backfillMissingJacobianKeys(
            std::initializer_list<std::shared_ptr<NodeType>> corners,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> &jacobians)
        {
            for (const auto &corner : corners)
            {
                if (!corner)
                    continue;
                for (const auto &[interpolated_key, key] : corner->interpolatedKeys())
                    if (!jacobians.count(key))
                        jacobians[key] = Eigen::MatrixXd::Zero(node_dim, InterpolatedKeysT::rowRange(interpolated_key).second);
            }
        }
    };
} // namespace Spacetime::Priors

#endif // GRIDINTERPOLATOR_HPP
