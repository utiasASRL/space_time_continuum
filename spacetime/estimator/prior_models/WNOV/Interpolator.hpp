#ifndef WNOV_INTERPOLATOR_HPP
#define WNOV_INTERPOLATOR_HPP

#include <Eigen/Dense>
#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/GridInterpolator.hpp>
#include <spacetime/estimator/prior_models/WNOV/WNOVState.hpp>
#include <spacetime/estimator/prior_models/WNOV/InterpolatedKeys.hpp>

namespace Spacetime::Priors::WNOV
{
    /**
     * @class Interpolator
     * @brief WNOV-specific interpolator: under white noise on velocity the pose is a
     *        Wiener process along each axis, so interpolation along one axis is geodesic
     *        (linear) interpolation and the full space-time interpolation, inherited from
     *        Priors::GridInterpolator, is bilinear on the group.
     *
     * Along one axis, with the query a fraction alpha = delta_i / delta of the way from
     * the left node T0 to the right node T1:
     *   mean   T = T0 * Exp(alpha * Log(T0^-1 T1))                    (gtsam::interpolate)
     *   Pcheck = Q * delta_i * (delta - delta_i) / delta              (Brownian bridge)
     * where Q is that axis's PSD (Q_space or Q_time). Pcheck lives in T0's local chart
     * gamma = alpha * Log(T0^-1 T1); it is reported in the interpolated pose's own
     * perturbation frame as Jr(gamma) Pcheck Jr(gamma)^T.
     *
     * The spatial prior has a nominal strain mean epsilon0 (its factors measure
     * Exp(ds * epsilon0)). A Brownian bridge's mean does not depend on the drift of the
     * process it is conditioned from, so interpolation ignores epsilon0 -- exactly in a
     * vector space, to first order on the group. Extrapolation, conditioned on one node
     * alone, does use it:
     *   space  T = T0 * Exp(delta_i * epsilon0),  Pcheck = Q_space * delta_i
     *   time   T = T0,                            Pcheck = Q_time * delta_i
     *
     * @tparam Pose Pose type for states (e.g., Pose2, Pose3).
     */
    /**
     * @brief The node-value-independent part of interpolating (or extrapolating) along one
     *        WNOV axis (see Priors::GridInterpolator's AxisPlanT): the geodesic fraction
     *        alpha, the extrapolation's chart offset gamma, and the chart-frame
     *        conditional covariance.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct AxisPlan
    {
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        using MatrixN = Eigen::Matrix<double, gtsam::traits<Pose>::dimension, gtsam::traits<Pose>::dimension>;
        double alpha = 0.0;                          ///< Interpolation fraction delta_i / delta (unused when extrapolating).
        TangentVector gamma = TangentVector::Zero(); ///< Extrapolation chart offset, T = T0 * Exp(gamma) (unused when interpolating).
        MatrixN chart_cov = MatrixN::Zero();         ///< Conditional covariance in the anchor's local chart.
    };

    template <typename Pose>
    class Interpolator : public GridInterpolator<Interpolator<Pose>, Node<Pose>, WNOV::InterpolatedKeys<Pose>, WNOV::AxisPlan<Pose>>
    {
        using Base = GridInterpolator<Interpolator<Pose>, Node<Pose>, WNOV::InterpolatedKeys<Pose>, WNOV::AxisPlan<Pose>>;

        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

    public:
        using NodeType = Node<Pose>;                      ///< WNOV grid node.
        using ValuesType = typename NodeType::ValuesType; ///< WNOV node values (the pose).
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        /// Interpolated-key layout, exposed for Priors::InterpolationFactor (see Types::InterpolatedKeyMap).
        using InterpolatedKeys = WNOV::InterpolatedKeys<Pose>;

        static constexpr int dim = gtsam::traits<Pose>::dimension; ///< Pose tangent dimension.
        static constexpr int node_dim = NodeType::dim;              ///< Node tangent dimension (== dim).

        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using MatrixNode = typename Base::MatrixNode;
        using AxisJacobian = typename Base::AxisJacobian;

        /**
         * @param Q_space Arclength-axis pose random-walk PSD (WNOV::Prior's Q1).
         * @param Q_time Time-axis pose random-walk PSD (WNOV::Prior's Q2).
         * @param epsilon0 Nominal strain, the drift of the arclength random walk.
         */
        Interpolator(const MatrixN &Q_space, const MatrixN &Q_time, const TangentVector &epsilon0)
            : m_Q_space(Q_space), m_Q_time(Q_time), m_epsilon0(epsilon0) {}

    private:
        friend Base;

        /// GridInterpolator hook: plan geodesic interpolation along one axis, with the
        /// Brownian-bridge conditional covariance.
        AxisPlan<Pose> planInterpolate(bool space, double delta_i, double delta) const
        {
            AxisPlan<Pose> plan;
            plan.alpha = delta_i / delta;
            plan.chart_cov = (space ? m_Q_space : m_Q_time) * (delta_i * (delta - delta_i) / delta);
            return plan;
        }

        /// GridInterpolator hook: plan extrapolation off the anchor alone, drifting by the
        /// nominal strain in arclength and holding the pose in time.
        AxisPlan<Pose> planExtrapolate(bool space, double delta_i) const
        {
            AxisPlan<Pose> plan;
            plan.gamma = space ? TangentVector(delta_i * m_epsilon0) : TangentVector(TangentVector::Zero());
            plan.chart_cov = (space ? m_Q_space : m_Q_time) * delta_i;
            return plan;
        }

        /// GridInterpolator hook: geodesic interpolation between n0 and n1 along a planned axis.
        ValuesType resolveInterpolate(
            const AxisPlan<Pose> &plan,
            const ValuesType &n0, const ValuesType &n1,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            const double alpha = plan.alpha;
            MatrixN H0, H1;
            const Pose pose = jac ? gtsam::interpolate<Pose>(n0.pose(), n1.pose(), alpha, H0, H1)
                                  : gtsam::interpolate<Pose>(n0.pose(), n1.pose(), alpha);
            if (jac)
            {
                *jac << H0, H1;
            }
            if (cov)
            {
                const TangentVector gamma = alpha * gtsam::traits<Pose>::Logmap(n0.pose().between(n1.pose()));
                *cov = chartToStateCovariance(gamma, plan.chart_cov);
            }
            return ValuesType(pose);
        }

        /// GridInterpolator hook: extrapolation off n0 alone along a planned axis.
        ValuesType resolveExtrapolate(
            const AxisPlan<Pose> &plan, const ValuesType &n0,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            const TangentVector &gamma = plan.gamma;
            MatrixN H0;
            const Pose pose = gtsam::traits<Pose>::Compose(n0.pose(), gtsam::traits<Pose>::Expmap(gamma), jac ? &H0 : nullptr);
            if (jac)
            {
                *jac << H0, MatrixN::Zero();
            }
            if (cov)
                *cov = chartToStateCovariance(gamma, plan.chart_cov);
            return ValuesType(pose);
        }

        /// Express a covariance of the chart coordinate gamma (T = T0 * Exp(gamma)) in the
        /// resolved pose's own right-perturbation frame: Jr(gamma) P Jr(gamma)^T.
        static MatrixNode chartToStateCovariance(const TangentVector &gamma, const MatrixN &chart_cov)
        {
            MatrixN Jr;
            gtsam::traits<Pose>::Expmap(gamma, Jr);
            const MatrixN state_cov = Jr * chart_cov * Jr.transpose();
            return 0.5 * (state_cov + state_cov.transpose());
        }

        MatrixN m_Q_space, m_Q_time;
        TangentVector m_epsilon0;
    };
} // namespace Spacetime::Priors::WNOV

#endif // WNOV_INTERPOLATOR_HPP
