#ifndef ENDEFFECTORINTERPOLATOR_HPP
#define ENDEFFECTORINTERPOLATOR_HPP

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Lie.h>
#include <gtsam/nonlinear/WnoaInterpolator.h>

#include <spacetime/estimator/joint_models/EndEffectorState.hpp>
#include <spacetime/estimator/prior_models/GridInterpolator.hpp>

namespace Spacetime::Joints
{
    /// The node-value-independent part of interpolating along the end effector's time axis: its offsets.
    struct EndEffectorAxisPlan
    {
        double delta_i = 0.0; ///< The query's offset past the left node.
        double delta = 0.0;   ///< The right node's offset (0 when extrapolating).
    };

    /**
     * @class EndEffectorInterpolator
     * @brief Interpolates an end effector's (pose, velocity) node in time under the WNOA
     *        prior, through gtsam::Interpolator (the same model as gtsam::WnoaMotionFactor):
     *        its mean, its eight Jacobian blocks and its conditional covariance.
     *        Extrapolation is constant-velocity, T = T0 Exp(dt varpi0), varpi = varpi0, with
     *        the WNOA process covariance. An end effector has no length, so its arclength
     *        axis never interpolates.
     *
     * @tparam Pose Pose type (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    class EndEffectorInterpolator
        : public Priors::GridInterpolator<EndEffectorInterpolator<Pose>, EndEffectorNode<Pose>, EndEffectorKeys<Pose>, EndEffectorAxisPlan>
    {
        using Base = Priors::GridInterpolator<EndEffectorInterpolator<Pose>, EndEffectorNode<Pose>, EndEffectorKeys<Pose>, EndEffectorAxisPlan>;

    public:
        using NodeType = EndEffectorNode<Pose>;
        using ValuesType = typename NodeType::ValuesType;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        using InterpolatedKeys = EndEffectorKeys<Pose>;

        static constexpr int dim = gtsam::traits<Pose>::dimension;
        static constexpr int node_dim = NodeType::dim;

        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using MatrixNode = typename Base::MatrixNode;
        using AxisJacobian = typename Base::AxisJacobian;

        /// @param q_psd_diag Diagonal of the WNOA power spectral density (as gtsam::WnoaMotionFactor's).
        explicit EndEffectorInterpolator(const TangentVector &q_psd_diag) : m_q_psd_diag(q_psd_diag), m_gtsam(q_psd_diag) {}

        const TangentVector &qPsdDiagonal() const { return m_q_psd_diag; }

    private:
        friend Base;
        using TimestampedPoseVel = gtsam::TimestampedPoseVelocity<Pose>;

        static void rejectSpace(bool space)
        {
            if (space)
                throw std::logic_error("EndEffectorInterpolator: an end effector has no length to interpolate along.");
        }

        EndEffectorAxisPlan planInterpolate(bool space, double delta_i, double delta) const
        {
            rejectSpace(space);
            return EndEffectorAxisPlan{delta_i, delta};
        }

        EndEffectorAxisPlan planExtrapolate(bool space, double delta_i) const
        {
            rejectSpace(space);
            return EndEffectorAxisPlan{delta_i, 0.0};
        }

        ValuesType resolveInterpolate(const EndEffectorAxisPlan &plan, const ValuesType &n0, const ValuesType &n1, AxisJacobian *jac, MatrixNode *cov) const
        {
            const TimestampedPoseVel k(n0.pose(), n0.varpi(), 0.0), kp1(n1.pose(), n1.varpi(), plan.delta);
            std::vector<gtsam::Matrix> H(8);
            const auto interpolated = m_gtsam.interpolatePoseAndVelocity(k, kp1, plan.delta_i, jac ? &H : nullptr);
            if (jac)
                // Blocks [dT/dT0, dT/dv0, dT/dT1, dT/dv1, dv/dT0, dv/dv0, dv/dT1, dv/dv1] (see gtsam::Interpolator).
                *jac << H[0], H[1], H[2], H[3],
                    H[4], H[5], H[6], H[7];
            if (cov)
                *cov = m_gtsam.computeConditionalCov(k, kp1, TimestampedPoseVel(interpolated, plan.delta_i));
            return ValuesType(interpolated.pose, interpolated.vel);
        }

        ValuesType resolveExtrapolate(const EndEffectorAxisPlan &plan, const ValuesType &n0, AxisJacobian *jac, MatrixNode *cov) const
        {
            const double dt = plan.delta_i;
            const TangentVector xi = dt * n0.varpi();
            MatrixN H_compose_T0, H_compose_exp, H_exp;
            const Pose step = gtsam::traits<Pose>::Expmap(xi, jac ? &H_exp : nullptr);
            const Pose pose = gtsam::traits<Pose>::Compose(n0.pose(), step, jac ? &H_compose_T0 : nullptr, jac ? &H_compose_exp : nullptr);
            if (jac)
            {
                jac->setZero();
                jac->template block<dim, dim>(0, 0) = H_compose_T0;
                jac->template block<dim, dim>(0, dim) = dt * H_compose_exp * H_exp;
                jac->template block<dim, dim>(dim, dim) = MatrixN::Identity();
            }
            if (cov)
            {
                // gtsam extrapolates when the right border is missing; with a zero marginal, its
                // covariance is the WNOA process covariance alone.
                gtsam::Matrix covariance;
                m_gtsam.interpolatePoseAndVelocity(TimestampedPoseVel(n0.pose(), n0.varpi(), 0.0), std::nullopt, dt, nullptr,
                                                   std::make_shared<gtsam::Matrix>(gtsam::Matrix::Zero(node_dim, node_dim)), &covariance);
                *cov = covariance;
            }
            return ValuesType(pose, n0.varpi());
        }

        TangentVector m_q_psd_diag;
        gtsam::Interpolator<Pose> m_gtsam;
    };
} // namespace Spacetime::Joints

#endif // ENDEFFECTORINTERPOLATOR_HPP
