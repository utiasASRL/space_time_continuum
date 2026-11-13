#ifndef WNOA_INTERPOLATOR_HPP
#define WNOA_INTERPOLATOR_HPP

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Eigen/Dense>
#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>
#include <spacetime/estimator/prior_models/WNOA/InterpolatedKeys.hpp>
#include <spacetime/estimator/prior_models/WNOA/functions.hpp>
#include <spacetime/estimator/prior_models/GridInterpolator.hpp>

namespace Spacetime::Priors::WNOA
{
    /**
     * @class Interpolator
     * @brief WNOA-specific (prior-specific) interpolator for the continuum robot
     *        node state (pose, generalized strain, generalized velocity), following
     *        the same structural role as gtsam::Interpolator<PoseType> but extended
     *        to interpolation across BOTH arclength and time using the two-part
     *        (arclength-then-time) scheme described in Section II-F of the
     *        accompanying paper ("A Stochastic Framework for Continuous-Time State
     *        Estimation of Continuum Robots").
     *
     * The class exposes, per axis, the same Lambda/Psi interpolation-matrix and
     * conditional-covariance primitives as gtsam::Interpolator. The per-axis entry
     * points (interpolateTime/Space, extrapolateTime/Space) and the `interpolate`
     * entry point that dispatches between the exact-match, 1D (arclength-only or
     * time-only), and full 2D cases are inherited from Priors::GridInterpolator; this
     * class supplies only the WNOA math along one axis (resolveInterpolate /
     * resolveExtrapolate).
     *
     * Corner dispatch (coincide / interpolate / extrapolate per axis, driven by the query
     * coordinates) is Priors::GridInterpolator's; see its class documentation.
     *
     * Extrapolation
     * -------------
     * Conditioned on the anchor node alone, the GP conditional is just interpolation
     * with Lambda = Phi(delta) and Psi = 0, so the mean reduces to the WNOA
     * constant-velocity / constant-strain prediction
     *   T(t_K + dt) = T_K * Exp((dt * varpi)^),  eps = J(dt varpi) eps_K,  varpi = varpi_K
     * (written in THIS FILE's T_ib / right-composition convention, see localToGlobal; the
     * paper and assets/extrapolation.tex use the mirrored T_bi / left-composition form)
     * (the last equality because J(xi) xi == xi), matching Barfoot, "State Estimation
     * for Robotics" (2nd ed.) eq. (11.51) with the strain state added. Losing the Psi
     * term also removes the second term of the conditional covariance, leaving the full
     * accumulated process noise Pcheck(delta) = Q(delta) -- and removing any need for
     * Q(delta)^-1, so extrapolation is better conditioned than interpolation.
     *
     * Extrapolation is defined FORWARD only (delta >= 0). The WNOA prior is an
     * integrated Wiener process, so its reversal is not the same process and Q(delta)
     * has no valid delta < 0 branch; a backwards query with no bordering node throws
     * (see GridInterpolator::classifyAxis).
     *
     * For a query active in both axes, the stages are ordered so the INTERPOLATED axis
     * resolves first and the EXTRAPOLATED axis last -- see the ordering note in
     * GridInterpolator::interpolateCorners for why (it keeps the two-stage covariance free of the "stage-1
     * results are independent" assumption). Note also that Phi_t and Phi_s commute, so
     * the doubly-extrapolated case is order-independent in local coordinates.
     *
     * Accuracy caveat specific to extrapolation: the prior factors this graph actually
     * contains (gtsam::WnoaMotionFactor plus a zero-mean BetweenFactor, matching
     * Legacy::BinaryTimeFactor::Factor::getZeroErrorNode) define the zero-process-noise manifold as
     * eps_1 = eps_0, whereas this class's chart gives eps_1 = J(xi) eps_0. The two agree
     * to O(|xi| |eps_0|). For interpolation |xi| is bounded by the node spacing, but for
     * extrapolation xi = delta * varpi_0 grows linearly in delta -- one reason
     * GridPrior::getOuterFactor warns whenever it builds an extrapolated
     * measurement factor.
     *
     * Conditional-covariance note: the covariance formulas here treat the
     * bordering (estimated) nodes as known exactly and report only the
     * interpolation/process-noise-induced uncertainty (matching
     * gtsam::WnoaInterpFactor's `fixed_noise_model_ = false` mode). The bordering nodes'
     * posterior marginal is propagated separately, by GridInterpolator::interpolate's
     * `posteriorCovariance` output. The per-axis formula,
     *   Pcheck(tau) = Q(tau - t_k) - Psi(tau) * Q(t_{k+1} - t_k) * Psi(tau)^T,
     * is derived directly from Barfoot, "State Estimation for Robotics" (2nd ed.),
     * Ch. 3.4.1, eq. (3.199)-(3.209), specialized to the case where the bordering
     * states carry zero uncertainty. Pcheck itself lives in
     * the local chart gamma of the left bordering node (Lambda/Psi/Q are functions
     * of the time/arclength difference only), but every per-query output of this
     * class -- interpolateTime/Space, extrapolateTime/Space and interpolate --
     * reports it in the interpolated node's STATE perturbation frame,
     * G * Pcheck * G^T, where G = d(state)/d(gamma) is chartToStateJacobian. That is
     * the frame every consumer (InterpolationFactor's noise model, the posterior
     * propagation, Prior::query) uses; adding the chart-frame Pcheck directly, as
     * the paper's eq. (36) does, under-reports e.g. the lever-arm position variance
     * of an arclength-interpolated node (see assets/covariance.tex). Only the
     * computeConditionalCov* helpers return the raw chart-frame Pcheck. For the
     * two-stage (arclength-then-time) case, the stage-1
     * (arclength) conditional covariances of the two intermediate nodes are
     * independent of one another, so they are
     * propagated into the final result through the SAME manifold Jacobian used
     * for the mean/Jacobian computation, matching the sandwich-form covariance
     * propagation pattern used throughout this project (see
     * WnoaInterpFactor::getInterpolatedNoiseModel for the analogous 1D case).
     *
     * Small-angle Jacobian approximation: the interpolated node's Jacobian with
     * respect to a bordering node's pose has a component (in globalToLocal's
     * eps/varpi rows, and chartToStateJacobian's 1/2 ad(.) blocks) of the
     * form `-1/2 * Jr(xi)^-1 * ad(v) * Jr(xi)^-1` (equivalently `+1/2 ad(v)` in
     * the composed form), derived from the standard first-order relation
     * `Jr(xi+dxi) ~= Jr(xi) - 1/2 dxi^ Jr(xi)` (see e.g. Forster et al., "IMU
     * Preintegration on Manifold", or Sola et al., "A micro Lie theory"). This is
     * a small-angle (small `xi`) approximation, not an exact closed form -- exact
     * differentiation of `Jr(xi)^-1` has no simple closed form. It was verified
     * (see WNOAInterpolationTests.cpp) to match numerical
     * differentiation to ~1e-4 relative accuracy for `xi` on the order of a few
     * hundredths of a radian, degrading to ~10-50% relative error for `xi`
     * approaching order 1 rad. This mirrors the accompanying paper's own stated
     * caveat that "issues will arise when rotations between two neighbouring
     * estimation nodes approach pi/2 rad" -- i.e., accuracy here assumes
     * estimation nodes are spaced closely enough in time/arclength that the
     * local rotation between them stays small, which callers are responsible for
     * ensuring (e.g. via the estimation period and the spatial discretization).
     *
     * Extrapolation inherits this approximation and concentrates it. The residual is
     * flat in the extrapolation distance -- it tracks the anchor node's own
     * strain/velocity magnitude, not `delta` -- but a purely extrapolated query has only
     * one bordering node, so Lambda = Phi puts the whole sensitivity on it instead of
     * splitting it across two or four. Measured against numerical differentiation at a
     * 0.02-scale operating point, the largest absolute Jacobian error is 0.0026-0.0049
     * for interpolation and 0.0108-0.0122 for single-corner extrapolation -- about 2.4x,
     * matching that concentration. It still vanishes linearly with the operating point
     * (see WNOAInterpolationTests' ExtrapolationJacobianErrorVanishesWithTheOperatingPoint).
     *
     * @tparam Pose Pose type for states (e.g., Pose2, Pose3).
     */
    /**
     * @brief The node-value-independent part of interpolating (or extrapolating) along one
     *        WNOA axis (see Priors::GridInterpolator's AxisPlanT): the (Lambda, Psi)
     *        interpolation matrices and the chart-frame conditional covariance Pcheck.
     *        Extrapolation has Psi = 0.
     */
    template <typename Pose>
    struct AxisPlan
    {
        using MatrixNode = Eigen::Matrix<double, Node<Pose>::dim, Node<Pose>::dim>;
        MatrixNode lambda = MatrixNode::Identity();
        MatrixNode psi = MatrixNode::Zero();
        MatrixNode chart_cov = MatrixNode::Zero();
    };

    template <typename Pose>
    class Interpolator : public GridInterpolator<Interpolator<Pose>, Node<Pose>, WNOA::InterpolatedKeys<Pose>, WNOA::AxisPlan<Pose>>
    {
        using Base = GridInterpolator<Interpolator<Pose>, Node<Pose>, WNOA::InterpolatedKeys<Pose>, WNOA::AxisPlan<Pose>>;

        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        static_assert(std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                      gtsam::lie_group_tag> ||
                          std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                         gtsam::vector_space_tag>,
                      "Pose type must be either a Lie group or vector space");
        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

    public:
        using NodeType = Node<Pose>;
        using ValuesType = typename NodeType::ValuesType;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        /// The WNOA-specific interpolated-key layout (see WNOA::InterpolatedKeys's
        /// own documentation), exposed generically for Priors::InterpolationFactor
        /// (see Types::InterpolatedKeyMap) -- this is the ONE place in the class
        /// hierarchy above WNOAState.hpp that is allowed to know WNOA's node is
        /// made of (pose, strain, velocity); InterpolationFactor itself never does.
        using InterpolatedKeys = WNOA::InterpolatedKeys<Pose>;

        static constexpr int dim = gtsam::traits<Pose>::dimension;
        static constexpr int node_dim = NodeType::dim;

        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using VectorNode = Eigen::Matrix<double, node_dim, 1>;
        using MatrixNode = Eigen::Matrix<double, node_dim, node_dim>;
        using AxisJacobian = typename Base::AxisJacobian;
        /// d(local chart coordinate)/d(one dim-sized component of a node), see globalToLocal.
        using ChartComponentJacobian = Eigen::Matrix<double, node_dim, dim>;
        /// Pair of (Lambda, Psi) interpolation matrices, mirroring gtsam::Interpolator::LambdaPsiMats.
        using LambdaPsi = std::pair<MatrixNode, MatrixNode>;

        /**
         * @brief Construct a WNOA interpolator from the four WNOA process-noise PSD
         *        matrices (matching WNOA::Prior's own Q1..Q4 members).
         * @param Q1 Time-axis (pose, velocity) WNOA process PSD.
         * @param Q2 Arclength-axis (pose, strain) WNOA process PSD.
         * @param Q3 Strain time-derivative (random-walk-in-time) process PSD.
         * @param Q4 Velocity arclength-derivative (random-walk-in-space) process PSD.
         */
        Interpolator(const MatrixN &Q1, const MatrixN &Q2, const MatrixN &Q3, const MatrixN &Q4)
            : m_Q1(Q1), m_Q2(Q2), m_Q3(Q3), m_Q4(Q4) {}

        /**
         * @brief Compute the (Lambda, Psi) interpolation matrices for the time axis.
         *        Mirrors gtsam::Interpolator::getLambdaPsi, but for the full
         *        node_dim (pose, strain, velocity) WNOA-in-time transition.
         */
        LambdaPsi getLambdaPsiTime(double t_k, double t_kp1, double t_tau) const
        {
            return getLambdaPsi(t_tau - t_k, t_kp1 - t_k, m_Q1, m_Q3, /*space=*/false);
        }

        /**
         * @brief Compute the (Lambda, Psi) interpolation matrices for the arclength
         *        axis. Mirrors gtsam::Interpolator::getLambdaPsi, but for the full
         *        node_dim (pose, strain, velocity) WNOA-in-arclength transition.
         */
        LambdaPsi getLambdaPsiSpace(double s_n, double s_np1, double s_tau) const
        {
            return getLambdaPsi(s_tau - s_n, s_np1 - s_n, m_Q2, m_Q4, /*space=*/true);
        }

        /**
         * @brief Conditional covariance of the time-interpolated node, assuming the
         *        two bordering nodes are known exactly. Mirrors
         *        gtsam::Interpolator::computeConditionalCov.
         *
         * This is the CHART-frame Pcheck (in the left node's local coordinates), a
         * building block independent of the node values. The state-frame conditional
         * covariance of an actual query comes from interpolateTime's conditionalCov.
         */
        MatrixNode computeConditionalCovTime(double t_k, double t_kp1, double t_tau) const
        {
            return computeConditionalCov(t_tau - t_k, t_kp1 - t_k, m_Q1, m_Q3, /*space=*/false);
        }

        /**
         * @brief Conditional covariance of the arclength-interpolated node, assuming
         *        the two bordering nodes are known exactly. Mirrors
         *        gtsam::Interpolator::computeConditionalCov.
         *
         * CHART-frame, like computeConditionalCovTime; the state-frame value of an
         * actual query comes from interpolateSpace's conditionalCov.
         */
        MatrixNode computeConditionalCovSpace(double s_n, double s_np1, double s_tau) const
        {
            return computeConditionalCov(s_tau - s_n, s_np1 - s_n, m_Q2, m_Q4, /*space=*/true);
        }

        /**
         * @brief Extrapolation counterpart of getLambdaPsiTime: conditioned on the
         *        anchor node alone (no bordering node past the query), the GP
         *        conditional degenerates to Lambda = Phi_t(delta), Psi = 0. See the
         *        class-level documentation's extrapolation section.
         * @param delta Forward time offset past the anchor; must be non-negative.
         */
        MatrixNode getLambdaExtrapTime(double delta) const
        {
            Base::throwIfBackwards(delta);
            if (delta < TOLERANCE)
                return MatrixNode::Identity();
            return Functions<Pose>::getPhiT(delta);
        }

        /// Arclength counterpart of getLambdaExtrapTime (Lambda = Phi_s(delta), Psi = 0); delta must be non-negative.
        MatrixNode getLambdaExtrapSpace(double delta) const
        {
            Base::throwIfBackwards(delta);
            if (delta < TOLERANCE)
                return MatrixNode::Identity();
            return Functions<Pose>::getPhiS(delta);
        }

        /**
         * @brief Conditional covariance of a time-extrapolated node, assuming the anchor
         *        node is known exactly. With no bordering node to condition on, the
         *        interpolation formula Q(delta_i) - Psi Q(delta) Psi^T loses its second
         *        term (Psi = 0), leaving the full accumulated process noise Q_t(delta).
         *
         * CHART-frame (the anchor's local coordinates); the state-frame value of an
         * actual query comes from extrapolateTime's conditionalCov.
         */
        MatrixNode computeConditionalCovExtrapTime(double delta) const
        {
            Base::throwIfBackwards(delta);
            if (delta < TOLERANCE)
                return MatrixNode::Zero();
            return Functions<Pose>::getQt(delta, m_Q1, m_Q3);
        }

        /// Arclength counterpart of computeConditionalCovExtrapTime (Q_s(delta)); also chart-frame.
        MatrixNode computeConditionalCovExtrapSpace(double delta) const
        {
            Base::throwIfBackwards(delta);
            if (delta < TOLERANCE)
                return MatrixNode::Zero();
            return Functions<Pose>::getQs(delta, m_Q2, m_Q4);
        }

    private:
        friend Base;

        /**
         * @brief GridInterpolator hook: plan interpolating along one axis between two
         *        bordering nodes -- the WNOA (Lambda, Psi) pair of that axis (Q1/Q3 for
         *        time, Q2/Q4 for arclength) and its chart-frame Pcheck.
         */
        AxisPlan<Pose> planInterpolate(bool space, double delta_i, double delta) const
        {
            const MatrixN &Qa = space ? m_Q2 : m_Q1;
            const MatrixN &Qb = space ? m_Q4 : m_Q3;
            AxisPlan<Pose> plan;
            std::tie(plan.lambda, plan.psi) = getLambdaPsi(delta_i, delta, Qa, Qb, space);
            plan.chart_cov = conditionalCovFromPsi(delta_i, delta, plan.psi, Qa, Qb, space);
            return plan;
        }

        /// GridInterpolator hook: plan extrapolating along one axis off the anchor node
        /// alone, (Lambda, Psi) = (Phi(delta_i), 0) with Pcheck the full Q(delta_i).
        AxisPlan<Pose> planExtrapolate(bool space, double delta_i) const
        {
            AxisPlan<Pose> plan;
            plan.lambda = space ? getLambdaExtrapSpace(delta_i) : getLambdaExtrapTime(delta_i);
            plan.chart_cov = space ? computeConditionalCovExtrapSpace(delta_i) : computeConditionalCovExtrapTime(delta_i);
            return plan;
        }

        /// GridInterpolator hook: interpolate along one planned axis between the two bordering
        /// nodes; `cov` is reported in the resolved node's state frame.
        ValuesType resolveInterpolate(
            const AxisPlan<Pose> &plan,
            const ValuesType &n0, const ValuesType &n1,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            MatrixNode G;
            ValuesType result = interpolateFromLambdaPsi(plan.lambda, plan.psi, n0, n1, jac, cov ? &G : nullptr);
            if (cov)
                *cov = chartToStateCovariance(G, plan.chart_cov);
            return result;
        }

        /**
         * @brief GridInterpolator hook: extrapolate along one axis off the anchor node
         *        alone, i.e. (Lambda, Psi) = (Phi(delta_i), 0) with conditional covariance
         *        the full Q(delta_i).
         *
         * Because Psi is exactly zero, interpolateFromLambdaPsi's dependence on its second
         * node vanishes structurally (the right half of `lambda_dgdx`, and hence of every
         * dxst_dx* block, is identically zero), so `n0` is passed in that role: it is a
         * guaranteed-valid node and globalToLocal(n0, n0) is the best-conditioned point of
         * the whole chart (T01 = I exactly, so gamma = [0; eps0; varpi0] with no error).
         */
        ValuesType resolveExtrapolate(
            const AxisPlan<Pose> &plan, const ValuesType &n0,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            MatrixNode G;
            ValuesType result = interpolateFromLambdaPsi(plan.lambda, plan.psi, n0, n0, jac, cov ? &G : nullptr);
            if (cov)
                *cov = chartToStateCovariance(G, plan.chart_cov);
            return result;
        }

        /// Pcheck lives in the left node's local chart; report it in the resolved node's
        /// state perturbation frame, where every consumer uses it (see assets/covariance.tex).
        static MatrixNode chartToStateCovariance(const MatrixNode &G, const MatrixNode &chart_cov)
        {
            const MatrixNode G_cov = G.lazyProduct(chart_cov);
            const MatrixNode state_cov = G_cov.lazyProduct(G.transpose());
            return 0.5 * (state_cov + state_cov.transpose());
        }

        /**
         * @brief Shared (Lambda, Psi) computation for a single WNOA axis. `space`
         *        selects which pair of process-noise PSD matrices couples with the
         *        pose (Q1/Q3 for time, Q2/Q4 for arclength) via
         *        Functions<Pose>::getPhiT/getQt or getPhiS/getQs.
         */
        LambdaPsi getLambdaPsi(double delta_i, double delta, const MatrixN &Qa, const MatrixN &Qb, bool space) const
        {
            MatrixNode lambda, psi;
            if (delta_i == 0.0 || delta == 0.0)
            {
                lambda = MatrixNode::Identity();
                psi = MatrixNode::Zero();
            }
            else if ((delta - delta_i) == 0.0)
            {
                lambda = MatrixNode::Zero();
                psi = MatrixNode::Identity();
            }
            else
            {
                const MatrixNode Q_inv = space ? Functions<Pose>::getQsInv(delta, Qa, Qb) : Functions<Pose>::getQtInv(delta, Qa, Qb);
                const MatrixNode Q_i = space ? Functions<Pose>::getQs(delta_i, Qa, Qb) : Functions<Pose>::getQt(delta_i, Qa, Qb);
                const MatrixNode Phi_i = space ? Functions<Pose>::getPhiS(delta_i) : Functions<Pose>::getPhiT(delta_i);
                const MatrixNode Phi_i_end = space ? Functions<Pose>::getPhiS(delta - delta_i) : Functions<Pose>::getPhiT(delta - delta_i);
                const MatrixNode Phi_end = space ? Functions<Pose>::getPhiS(delta) : Functions<Pose>::getPhiT(delta);
                psi = Q_i * Phi_i_end.transpose() * Q_inv;
                lambda = Phi_i - psi * Phi_end;
            }
            return {lambda, psi};
        }

        /**
         * @brief Shared conditional-covariance computation for a single WNOA axis,
         *        assuming the two bordering nodes are known exactly:
         *          Pcheck(tau) = Q(delta_i) - Psi(tau) * Q(delta) * Psi(tau)^T.
         *        See the class-level documentation for the derivation reference.
         */
        MatrixNode computeConditionalCov(double delta_i, double delta, const MatrixN &Qa, const MatrixN &Qb, bool space) const
        {
            return conditionalCovFromPsi(delta_i, delta, getLambdaPsi(delta_i, delta, Qa, Qb, space).second, Qa, Qb, space);
        }

        /// computeConditionalCov for an axis whose Psi is already known (see planInterpolate).
        MatrixNode conditionalCovFromPsi(double delta_i, double delta, const MatrixNode &psi,
                                         const MatrixN &Qa, const MatrixN &Qb, bool space) const
        {
            if (delta_i == 0.0 || (delta - delta_i) == 0.0)
                return MatrixNode::Zero();

            const MatrixNode Q_i = space ? Functions<Pose>::getQs(delta_i, Qa, Qb) : Functions<Pose>::getQt(delta_i, Qa, Qb);
            const MatrixNode Q_total = space ? Functions<Pose>::getQs(delta, Qa, Qb) : Functions<Pose>::getQt(delta, Qa, Qb);
            return Q_i - psi * Q_total * psi.transpose();
        }

        /**
         * @brief Interpolate a node's mean and (optionally) its analytic Jacobian
         *        from a precomputed (Lambda, Psi) pair and two bordering node values:
         *        gamma_st = Lambda * gamma(node0) + Psi * gamma(node1), in node0's chart.
         *        Descended from the legacy WNOA::Prior::interpolateMean1D, re-derived in
         *        the T_ib convention (see globalToLocal).
         *
         * @param jac Optional output node_dim x 2*node_dim Jacobian [d/d(node0) | d/d(node1)].
         * @param chartToState Optional output: the chart-to-state Jacobian G at the
         *        interpolated point (see chartToStateJacobian), needed to express a
         *        chart-frame covariance in the interpolated node's state frame.
         */
        ValuesType interpolateFromLambdaPsi(
            const MatrixNode &lambda, const MatrixNode &psi,
            const ValuesType &node0, const ValuesType &node1,
            AxisJacobian *jac,
            MatrixNode *chartToState = nullptr) const
        {
            VectorNode g_hat0, g_hat1;
            // d(g_hat_j)/d(node_k); d(g_hat0)/d(node1) is identically zero (see below).
            MatrixNode dg0dx0, dg1dx0, dg1dx1;
            if (jac)
            {
                // g_hat0 = globalToLocal(node0, node0) depends only on node0, which
                // fills BOTH the "node" and "node_nk" argument roles -- so its total
                // sensitivity to node0's pose is the SUM of the two roles' pose
                // Jacobians (H_nk_pose and H_node_pose are two partial derivatives of
                // the same total derivative), and its sensitivity to node1 is exactly
                // zero (node1 never appears in this call at all).
                ChartComponentJacobian dg0_self_nk_pose, dg0_self_node_pose, dg0_self_node_strain, dg0_self_node_velocity;
                globalToLocal(node0, node0, g_hat0, &dg0_self_nk_pose, &dg0_self_node_pose,
                              &dg0_self_node_strain, &dg0_self_node_velocity);
                dg0dx0 << (dg0_self_nk_pose + dg0_self_node_pose), dg0_self_node_strain, dg0_self_node_velocity;

                ChartComponentJacobian dg1_nk_pose, dg1_node_pose, dg1_node_strain, dg1_node_velocity;
                globalToLocal(node1, node0, g_hat1, &dg1_nk_pose, &dg1_node_pose,
                              &dg1_node_strain, &dg1_node_velocity);
                // node0 (the "node_nk" role here) only ever contributes through its
                // own pose -- its strain/velocity columns are always zero.
                dg1dx0 << dg1_nk_pose, ChartComponentJacobian::Zero(), ChartComponentJacobian::Zero();
                dg1dx1 << dg1_node_pose, dg1_node_strain, dg1_node_velocity;
            }
            else
            {
                globalToLocal(node0, node0, g_hat0);
                globalToLocal(node1, node0, g_hat1);
            }

            const VectorNode g_hat_st = lambda * g_hat0 + psi * g_hat1;

            ValuesType query_values;
            localToGlobal(g_hat_st, node0, query_values);

            if (!jac && !chartToState)
                return query_values;

            const MatrixNode G = chartToStateJacobian(g_hat_st);
            if (chartToState)
                *chartToState = G;
            if (!jac)
                return query_values;

            // [Lambda | Psi] * [dg0dx0 dg0dx1; dg1dx0 dg1dx1], with dg0dx1 identically zero.
            // Coefficient-based (lazy) products: at node_dim scale Eigen's blocked GEMM costs
            // more in packing than it saves.
            AxisJacobian lambda_dgdx;
            lambda_dgdx << lambda.lazyProduct(dg0dx0) + psi.lazyProduct(dg1dx0), psi.lazyProduct(dg1dx1);

            // Chain rule through the chart: d(query state)/d(corners) = G * d(gamma_st)/d(corners).
            // T_ib (right-composition) re-derivation: T_query = T0 * Exp(xi_st), so
            // perturbing T0 directly (T0 -> T0*Exp(d)) while holding g_hat_st fixed
            // gives, exactly, d(local coords of query)/dd = Ad(Exp(xi_st))^-1 * d =
            // Ad(tran_st^-1) -- the inverse of what a left-composition (T_bi-style)
            // reconstruction would need. This is on top of (not instead of) the
            // regular chain-rule term through g_hat_st (dg0dx0/dg1dx0 -> lambda_dgdx),
            // which already carries T0's dependence via globalToLocal's own H_nk/H_node.
            AxisJacobian J = G.lazyProduct(lambda_dgdx);
            const auto tran_st = gtsam::traits<Pose>::Expmap(g_hat_st.template head<dim>());
            J.block(0, 0, dim, dim) += tran_st.inverse().AdjointMap();

            *jac = J;
            return query_values;
        }

        /**
         * @brief The chart-to-state Jacobian G = d(state perturbation)/d(gamma) at a
         *        chart point gamma = [xi; gamma_eps; gamma_varpi] (see localToGlobal),
         *        assets/covariance.tex eq. (7):
         *          G = [ Jr(xi)                   0       0      ]
         *              [ 1/2 ad(gamma_eps)        Jr(xi)  0      ]
         *              [ 1/2 ad(gamma_varpi)      0       Jr(xi) ]
         *        The 1/2 ad(.) blocks use the same first-order approximation of
         *        d(Jr(xi) v)/dxi as the rest of this class (see the class docs).
         */
        MatrixNode chartToStateJacobian(const VectorNode &gamma) const
        {
            Eigen::Matrix<double, dim, dim> Jr;
            gtsam::traits<Pose>::Expmap(gamma.template head<dim>(), Jr);

            MatrixNode G = MatrixNode::Zero();
            G.template block<dim, dim>(0, 0) = Jr;
            G.template block<dim, dim>(dim, dim) = Jr;
            G.template block<dim, dim>(2 * dim, 2 * dim) = Jr;
            G.template block<dim, dim>(dim, 0) = 0.5 * gtsam::traits<Pose>::adjointMap(gamma.template segment<dim>(dim));
            G.template block<dim, dim>(2 * dim, 0) = 0.5 * gtsam::traits<Pose>::adjointMap(gamma.template segment<dim>(2 * dim));
            return G;
        }

        /**
         * @brief Local coordinate of `node` relative to reference `node_nk`, in
         *        this codebase's own T_ib (right/body-frame) pose convention: T01 =
         *        node_nk^-1 * node = Exp(xi), so node.pose() == node_nk.pose() *
         *        Exp(xi) exactly (see localToGlobal, its exact inverse).
         *
         * Re-derived from scratch in T_ib convention (see PR discussion): the
         * eps/varpi rows' dependence on a pose perturbation of `node`/`node_nk`
         * uses the standard right-Jacobian identity
         *   d/dxi [Jr(xi)^-1 v] * dxi = -1/2 Jr(xi)^-1 ad(v) Jr(xi)^-1 dxi
         * (obtained via second-order BCH matching, valid to the same order as the
         * rest of this interpolation scheme), with v = node.epsilon() /
         * node.varpi() -- NOT node_nk's, since gamma's eps/varpi rows only ever
         * depend on `node`'s own epsilon/varpi. The previous version of this
         * function (ported from WNOA::Prior::globalToLocal) was derived in the
         * paper's T_bi convention and used node_nk's epsilon/varpi with a
         * different sign split between H_nk/H_node; that formula is not
         * consistent with this codebase's T_ib convention.
         *
         * Jacobians are returned one per KEY (pose/strain/velocity), each
         * node_dim x dim, rather than as a single node_dim x node_dim matrix
         * sliced by a stored column offset -- `node_nk` only ever contributes
         * through its pose (its own strain/velocity never appear in `gamma`
         * at all), so only `H_nk_pose` is offered for it.
         *
         * @param gamma Output chart coordinate [xi; Jr(xi)^-1 eps; Jr(xi)^-1 varpi].
         * @param H_nk_pose Optional d(gamma)/d(node_nk pose).
         * @param H_node_pose Optional d(gamma)/d(node pose).
         * @param H_node_strain Optional d(gamma)/d(node strain).
         * @param H_node_velocity Optional d(gamma)/d(node velocity).
         */
        static void globalToLocal(const ValuesType &node, const ValuesType &node_nk, VectorNode &gamma,
                                   ChartComponentJacobian *H_nk_pose = nullptr,
                                   ChartComponentJacobian *H_node_pose = nullptr,
                                   ChartComponentJacobian *H_node_strain = nullptr,
                                   ChartComponentJacobian *H_node_velocity = nullptr)
        {
            // NOTE: dT01_d_nk/dT01_d_node must be plain matrices passed by pointer.
            // A bare `Pose::ChartJacobian` (OptionalJacobian) value, passed without
            // `&`, is never bound to real storage, so `.between()`'s internal
            // `if (H1) ...` guard sees it as inactive and never writes to it -- and
            // subsequent uses like `dxi_dT01 * dT01_d_nk` then silently compile via
            // OptionalJacobian::operator bool() implicitly converting to 0, turning
            // the whole expression into a silent zero. This was verified directly
            // (see PR discussion) to be why the pose-perturbation Jacobian
            // contributions here were vanishing.
            Eigen::Matrix<double, dim, dim> dT01_d_nk, dT01_d_node;
            Pose T01 = node_nk.pose().between(node.pose(), &dT01_d_nk, &dT01_d_node);

            Eigen::Matrix<double, dim, dim> dxi_dT01;
            gamma.template block<dim, 1>(0, 0) = gtsam::traits<Pose>::Logmap(T01, &dxi_dT01);
            gamma.template block<dim, 1>(dim, 0) = dxi_dT01 * node.epsilon();
            gamma.template block<dim, 1>(2 * dim, 0) = dxi_dT01 * node.varpi();

            if (H_nk_pose || H_node_pose || H_node_strain || H_node_velocity)
            {
                const Eigen::Matrix<double, dim, dim> eps_term = -0.5 * dxi_dT01 * Pose::adjointMap(node.epsilon()) * dxi_dT01;
                const Eigen::Matrix<double, dim, dim> varpi_term = -0.5 * dxi_dT01 * Pose::adjointMap(node.varpi()) * dxi_dT01;

                if (H_nk_pose)
                {
                    *H_nk_pose << dxi_dT01 * dT01_d_nk,
                        eps_term * dT01_d_nk,
                        varpi_term * dT01_d_nk;
                }
                if (H_node_pose)
                {
                    *H_node_pose << dxi_dT01 * dT01_d_node,
                        eps_term * dT01_d_node,
                        varpi_term * dT01_d_node;
                }
                if (H_node_strain)
                {
                    *H_node_strain << MatrixN::Zero(), dxi_dT01, MatrixN::Zero();
                }
                if (H_node_velocity)
                {
                    *H_node_velocity << MatrixN::Zero(), MatrixN::Zero(), dxi_dT01;
                }
            }
        }

        /**
         * @brief Inverse of globalToLocal: reconstructs the node whose local
         *        coordinate (relative to node_nk, in T_ib/right convention) is
         *        gamma, i.e. node.pose() = node_nk.pose() * Exp(xi), eps = Jr(xi) gamma_eps,
         *        varpi = Jr(xi) gamma_varpi.
         *
         * The right composition must match globalToLocal's (right-convention) `between`
         * for the two to be exact inverses; the legacy left composition
         * Exp(xi) * node_nk.pose() was not.
         */
        static void localToGlobal(const VectorNode &gamma, const ValuesType &node_nk, ValuesType &node)
        {
            Eigen::Matrix<double, dim, dim> jac;
            const Eigen::Matrix<double, dim, 1> xi = gamma.template block<dim, 1>(0, 0);
            node.pose(node_nk.pose() * gtsam::traits<Pose>::Expmap(xi, jac));
            node.epsilon(jac * gamma.template block<dim, 1>(dim, 0));
            node.varpi(jac * gamma.template block<dim, 1>(2 * dim, 0));
        }

        MatrixN m_Q1, m_Q2, m_Q3, m_Q4;
    };
} // namespace Spacetime::Priors::WNOA

#endif // WNOA_INTERPOLATOR_HPP
