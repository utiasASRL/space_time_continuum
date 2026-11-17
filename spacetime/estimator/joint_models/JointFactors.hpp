#ifndef JOINTFACTORS_HPP
#define JOINTFACTORS_HPP

#include <Eigen/Core>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

/**
 * @file JointFactors.hpp
 * @brief Inner factors of Joints::Joint, evaluated on the two joined points' interpolated
 *        nodes (see Priors::TwoSidedInterpolationFactor). Poses are T_ib and twists body
 *        frame (T_dot = T varpi^), both in GTSAM [rotational, translational] order. The
 *        joint offset T_AB maps point A to point B, expressed in A's frame (T_B = T_A T_AB);
 *        a spherical joint only uses its translation r. A rigid joint's pose factor is a
 *        plain gtsam::BetweenFactor<Pose3>(A, B, T_AB).
 */
namespace Spacetime::Joints
{
    using Vector3 = Eigen::Vector3d;              ///< 3-vector.
    using Vector6 = Eigen::Matrix<double, 6, 1>;  ///< 6-vector, e.g. a body twist [omega; nu].
    using Matrix3 = Eigen::Matrix3d;              ///< 3x3 matrix.
    using Matrix6 = Eigen::Matrix<double, 6, 6>;  ///< 6x6 matrix.

    /**
     * @brief Rigid-joint velocity residual e = varpi_B - Ad(T_AB^-1) varpi_A: point B's body
     *        twist is that of A carried through the rigid offset.
     */
    class RigidVelocityFactor : public gtsam::NoiseModelFactorN<Vector6, Vector6>
    {
    public:
        using Base = gtsam::NoiseModelFactorN<Vector6, Vector6>;
        using Base::evaluateError;

        /**
         * @param velocity_a Key of point A's body twist varpi_A.
         * @param velocity_b Key of point B's body twist varpi_B.
         * @param offset The joint offset T_AB.
         * @param noise 6-dimensional noise model of the residual.
         */
        RigidVelocityFactor(gtsam::Key velocity_a, gtsam::Key velocity_b, const gtsam::Pose3 &offset, const gtsam::SharedNoiseModel &noise)
            : Base(noise, velocity_a, velocity_b), m_adjoint(offset.inverse().AdjointMap()) {}

        /// The residual varpi_B - Ad(T_AB^-1) varpi_A, with its Jacobians -Ad(T_AB^-1) and I.
        gtsam::Vector evaluateError(const Vector6 &varpi_a, const Vector6 &varpi_b, gtsam::OptionalMatrixType H_a, gtsam::OptionalMatrixType H_b) const override
        {
            if (H_a)
                *H_a = -m_adjoint;
            if (H_b)
                *H_b = Matrix6::Identity();
            return varpi_b - m_adjoint * varpi_a;
        }

        /**
         * @brief First-order covariance the offset's uncertainty `sigma` (in its own tangent
         *        space) induces on the residual: perturbing T_AB by Exp(xi) changes
         *        Ad(T_AB^-1) varpi_A by -ad(xi) w = ad(w) xi, w = Ad(T_AB^-1) varpi_A, so it is
         *        ad(w) sigma ad(w)^T. Zero at zero velocity.
         */
        static Matrix6 offsetCovariance(const Vector6 &varpi_a, const gtsam::Pose3 &offset, const Matrix6 &sigma)
        {
            const Matrix6 ad = gtsam::Pose3::adjointMap(offset.inverse().AdjointMap() * varpi_a);
            return ad * sigma * ad.transpose();
        }

    private:
        Matrix6 m_adjoint; // Ad(T_AB^-1)
    };

    /**
     * @brief Spherical-joint position residual e = R_A^T (t_B - t_A) - r: point B lies at
     *        offset r from point A, in A's frame. Rotations are free.
     */
    class SphericalPositionFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3>
    {
    public:
        using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3>;
        using Base::evaluateError;

        /**
         * @param pose_a Key of point A's pose T_A.
         * @param pose_b Key of point B's pose T_B.
         * @param offset Position r of point B relative to point A, in A's frame (m).
         * @param noise 3-dimensional noise model of the residual.
         */
        SphericalPositionFactor(gtsam::Key pose_a, gtsam::Key pose_b, const Vector3 &offset, const gtsam::SharedNoiseModel &noise)
            : Base(noise, pose_a, pose_b), m_offset(offset) {}

        /// The residual R_A^T (t_B - t_A) - r, with Jacobians w.r.t. right perturbations of T_A and T_B.
        gtsam::Vector evaluateError(const gtsam::Pose3 &T_a, const gtsam::Pose3 &T_b, gtsam::OptionalMatrixType H_a, gtsam::OptionalMatrixType H_b) const override
        {
            const Matrix3 R_a = T_a.rotation().matrix();
            const Vector3 d = R_a.transpose() * (T_b.translation() - T_a.translation());
            if (H_a)
            {
                // T_A Exp([phi; rho]): d -> (I - phi^)(d - rho)
                H_a->resize(3, 6);
                *H_a << gtsam::skewSymmetric(d), -Matrix3::Identity();
            }
            if (H_b)
            {
                H_b->resize(3, 6);
                *H_b << Matrix3::Zero(), R_a.transpose() * T_b.rotation().matrix();
            }
            return d - m_offset;
        }

    private:
        Vector3 m_offset;
    };

    /**
     * @brief Spherical-joint velocity residual e = nu_A + omega_A x r - R_A^T R_B nu_B: the
     *        joined point's linear velocity seen from A (rigidly carried by A's twist) and
     *        from B (B's own linear velocity), both in A's frame. Angular velocities are free.
     */
    class SphericalVelocityFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, Vector6, gtsam::Pose3, Vector6>
    {
    public:
        using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, Vector6, gtsam::Pose3, Vector6>;
        using Base::evaluateError;

        /**
         * @param pose_a Key of point A's pose T_A.
         * @param velocity_a Key of point A's body twist varpi_A = [omega_A; nu_A].
         * @param pose_b Key of point B's pose T_B.
         * @param velocity_b Key of point B's body twist varpi_B = [omega_B; nu_B].
         * @param offset Position r of point B relative to point A, in A's frame (m).
         * @param noise 3-dimensional noise model of the residual.
         */
        SphericalVelocityFactor(gtsam::Key pose_a, gtsam::Key velocity_a, gtsam::Key pose_b, gtsam::Key velocity_b,
                                const Vector3 &offset, const gtsam::SharedNoiseModel &noise)
            : Base(noise, pose_a, velocity_a, pose_b, velocity_b), m_offset(offset) {}

        /// The residual nu_A + omega_A x r - R_A^T R_B nu_B, with Jacobians w.r.t. right perturbations of the poses.
        gtsam::Vector evaluateError(const gtsam::Pose3 &T_a, const Vector6 &varpi_a, const gtsam::Pose3 &T_b, const Vector6 &varpi_b,
                                    gtsam::OptionalMatrixType H_pa, gtsam::OptionalMatrixType H_va,
                                    gtsam::OptionalMatrixType H_pb, gtsam::OptionalMatrixType H_vb) const override
        {
            const Matrix3 R_ab = T_a.rotation().matrix().transpose() * T_b.rotation().matrix();
            const Vector3 nu_b = varpi_b.tail<3>();
            const Vector3 nu_b_in_a = R_ab * nu_b;
            if (H_pa)
            {
                H_pa->resize(3, 6);
                *H_pa << -gtsam::skewSymmetric(nu_b_in_a), Matrix3::Zero();
            }
            if (H_va)
            {
                H_va->resize(3, 6);
                *H_va << -gtsam::skewSymmetric(m_offset), Matrix3::Identity();
            }
            if (H_pb)
            {
                H_pb->resize(3, 6);
                *H_pb << R_ab * gtsam::skewSymmetric(nu_b), Matrix3::Zero();
            }
            if (H_vb)
            {
                H_vb->resize(3, 6);
                *H_vb << Matrix3::Zero(), -R_ab;
            }
            return varpi_a.tail<3>() + varpi_a.head<3>().cross(m_offset) - nu_b_in_a;
        }

        /// First-order covariance the offset's uncertainty `sigma` induces on the residual: [omega_A]x sigma [omega_A]x^T.
        static Matrix3 offsetCovariance(const Vector6 &varpi_a, const Matrix3 &sigma)
        {
            const Matrix3 skew = gtsam::skewSymmetric(Vector3(varpi_a.head<3>()));
            return skew * sigma * skew.transpose();
        }

    private:
        Vector3 m_offset;
    };
} // namespace Spacetime::Joints

#endif // JOINTFACTORS_HPP
