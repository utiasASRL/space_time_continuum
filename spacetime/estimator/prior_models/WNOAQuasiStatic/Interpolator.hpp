#ifndef WNOAQUASISTATIC_INTERPOLATOR_HPP
#define WNOAQUASISTATIC_INTERPOLATOR_HPP

#include <Eigen/Dense>
#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/GridInterpolator.hpp>
#include <spacetime/estimator/prior_models/WNOA/Interpolator.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/WNOAQuasiStaticState.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/InterpolatedKeys.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    /**
     * @brief The node-value-independent part of resolving a query along one quasi-static
     *        axis (see Priors::GridInterpolator's AxisPlanT).
     *
     * Arclength keeps the offsets for the WNOA interpolation; time only records which of
     * the two bordering rows is nearer.
     */
    struct AxisPlan
    {
        bool space = false;      ///< Whether this is the arclength axis.
        double delta_i = 0.0;    ///< Arclength: the query's offset past the left node.
        double delta = 0.0;      ///< Arclength: the right node's offset (0 when extrapolating).
        bool take_right = false; ///< Time: the right row is the nearer one.
    };

    /**
     * @class Interpolator
     * @brief Quasi-static interpolator: WNOA in arclength, nearest-row in time.
     *
     * Arclength: the (pose, strain) WNOA interpolation of WNOA::Interpolator, which this
     * class delegates to with a zero velocity. Along arclength, WNOA's pose and strain
     * never depend on the velocity (Phi_s, Q_s and the chart are block-triangular with the
     * velocity last), so the top-left 2*dim x 2*dim blocks (12 x 12 for Pose3) of its
     * mean, Jacobian and conditional covariance are exactly the quasi-static ones.
     *
     * Time: the quasi-static prior has no motion model, so a query between two rows
     * resolves to the nearer row (the left one on a tie), and past the last row to that
     * row. Either adds no conditional covariance: the snapped node's posterior is the
     * row's own.
     *
     * @tparam Pose Pose type for states (e.g., Pose3).
     */
    template <typename Pose>
    class Interpolator : public GridInterpolator<Interpolator<Pose>, Node<Pose>, WNOAQuasiStatic::InterpolatedKeys<Pose>, AxisPlan>
    {
        using Base = GridInterpolator<Interpolator<Pose>, Node<Pose>, WNOAQuasiStatic::InterpolatedKeys<Pose>, AxisPlan>;
        using WNOAInterpolator = WNOA::Interpolator<Pose>;
        using WNOAValues = WNOA::Values<Pose>;

        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

    public:
        using NodeType = Node<Pose>;                      ///< Quasi-static grid node.
        using ValuesType = typename NodeType::ValuesType; ///< Quasi-static node values (pose, strain).
        /// Interpolated-key layout, exposed for Priors::InterpolationFactor (see Types::InterpolatedKeyMap).
        using InterpolatedKeys = WNOAQuasiStatic::InterpolatedKeys<Pose>;

        static constexpr int dim = gtsam::traits<Pose>::dimension; ///< Pose tangent dimension.
        static constexpr int node_dim = NodeType::dim;              ///< Node tangent dimension (2 * dim).

        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using MatrixNode = typename Base::MatrixNode;
        using AxisJacobian = typename Base::AxisJacobian;

        /// @param Q2 Arclength-axis (pose, strain) WNOA process PSD (the prior's Q2).
        explicit Interpolator(const MatrixN &Q2)
            : m_wnoa(MatrixN::Identity(), Q2, MatrixN::Identity(), MatrixN::Identity()) {}

    private:
        friend Base;
        using WNOAMatrixNode = typename WNOAInterpolator::MatrixNode;

        /// GridInterpolator hook: arclength keeps the offsets; time picks the nearer row.
        AxisPlan planInterpolate(bool space, double delta_i, double delta) const
        {
            AxisPlan plan;
            plan.space = space;
            if (space)
            {
                plan.delta_i = delta_i;
                plan.delta = delta;
            }
            else
                plan.take_right = delta_i > 0.5 * delta;
            return plan;
        }

        /// GridInterpolator hook: arclength extrapolates along the strain; time holds the anchor row.
        AxisPlan planExtrapolate(bool space, double delta_i) const
        {
            AxisPlan plan;
            plan.space = space;
            if (space)
                plan.delta_i = delta_i;
            return plan;
        }

        /// GridInterpolator hook: WNOA interpolation in arclength, nearest row in time.
        ValuesType resolveInterpolate(
            const AxisPlan &plan,
            const ValuesType &n0, const ValuesType &n1,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            if (!plan.space) // time: snap
            {
                if (jac)
                {
                    jac->setZero();
                    if (plan.take_right)
                        jac->template rightCols<node_dim>().setIdentity();
                    else
                        jac->template leftCols<node_dim>().setIdentity();
                }
                if (cov)
                    cov->setZero();
                return plan.take_right ? n1 : n0;
            }

            gtsam::Matrix wnoa_jac;
            WNOAMatrixNode wnoa_cov;
            const WNOAValues result = m_wnoa.interpolateSpace(
                0.0, plan.delta, plan.delta_i, toWNOA(n0), toWNOA(n1), jac ? &wnoa_jac : nullptr, cov ? &wnoa_cov : nullptr);
            if (jac)
                *jac << wnoa_jac.block(0, 0, node_dim, node_dim), wnoa_jac.block(0, WNOAInterpolator::node_dim, node_dim, node_dim);
            if (cov)
                *cov = wnoa_cov.template topLeftCorner<node_dim, node_dim>();
            return fromWNOA(result);
        }

        /// GridInterpolator hook: WNOA extrapolation in arclength, the anchor row held in time.
        ValuesType resolveExtrapolate(
            const AxisPlan &plan, const ValuesType &n0,
            AxisJacobian *jac,
            MatrixNode *cov) const
        {
            if (!plan.space) // time: hold
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

            gtsam::Matrix wnoa_jac;
            WNOAMatrixNode wnoa_cov;
            const WNOAValues result = m_wnoa.extrapolateSpace(
                0.0, plan.delta_i, toWNOA(n0), jac ? &wnoa_jac : nullptr, cov ? &wnoa_cov : nullptr);
            if (jac)
                *jac << wnoa_jac.block(0, 0, node_dim, node_dim), MatrixNode::Zero();
            if (cov)
                *cov = wnoa_cov.template topLeftCorner<node_dim, node_dim>();
            return fromWNOA(result);
        }

        /// The same pose and strain as a WNOA node, with a zero velocity.
        static WNOAValues toWNOA(const ValuesType &v)
        {
            WNOAValues out;
            out.pose(v.pose());
            out.epsilon(v.epsilon());
            return out;
        }

        /// The pose and strain of a WNOA node, dropping its velocity.
        static ValuesType fromWNOA(const WNOAValues &v) { return ValuesType(v.pose(), v.epsilon()); }

        WNOAInterpolator m_wnoa; // its Q1, Q3 and Q4 never enter the arclength pose/strain blocks
    };
} // namespace Spacetime::Priors::WNOAQuasiStatic

#endif // WNOAQUASISTATIC_INTERPOLATOR_HPP
