#ifndef WNOA_BINARYSPACEFACTOR_HPP
#define WNOA_BINARYSPACEFACTOR_HPP

#include <Eigen/Dense>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA
{
    /*
     * BinarySpaceFactor, BinaryTimeFactor and UnaryFactor are the paper's GP prior factors,
     * defined on the group (rather than the Lie algebra), used by WNOA::Prior with
     * FactorType::WNOA and FactorType::Legacy; rewritten in this codebase's T_ib,
     * [rotational, translational] conventions. The Lie-algebra alternative
     * (FactorType::Experimental) is in WNOA::Experimental.
     */

    /// The paper's space factor between two arclength-adjacent WNOA nodes (FactorType::WNOA / FactorType::Legacy).
    namespace BinarySpaceFactor
    {

        /**
         * @brief Space factor on (pose, eps, varpi) of nodes 0 and 1, ds apart in arclength.
         * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
         */
        template <typename Pose>
        class Factor : public gtsam::NoiseModelFactorN<Pose, typename gtsam::traits<Pose>::TangentVector, typename gtsam::traits<Pose>::TangentVector, Pose, typename gtsam::traits<Pose>::TangentVector, typename gtsam::traits<Pose>::TangentVector>
        {
            // Check that Pose type is a testable Lie group
            GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
            // We currently support vector spaces and Lie groups
            static_assert(std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                         gtsam::lie_group_tag> ||
                              std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                             gtsam::vector_space_tag>,
                          "Pose type must be either a Lie group or vector space");

            GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

        public:
            static constexpr int dim = gtsam::traits<Pose>::dimension;
            using TangentVector = typename gtsam::traits<Pose>::TangentVector;
            using Base = gtsam::NoiseModelFactorN<Pose, TangentVector, TangentVector, Pose, TangentVector, TangentVector>;
            using ErrorVector = Eigen::Vector<double, 3 * dim>;
            using ErrorJacobian = Eigen::Matrix<double, 3 * dim, dim>;

            /**
             * @param key_pose0,key_epsilon0,key_varpi0 Node 0's pose, strain and velocity keys.
             * @param key_pose1,key_epsilon1,key_varpi1 Node 1's pose, strain and velocity keys.
             * @param noise_model Typically noiseModel(ds, Q2, Q4).
             * @param ds Arclength from node 0 to node 1.
             * @param strain_velocity_coupling Include the ad(eps0)*varpi0 term in the velocity row
             *        (the legacy code, FactorType::Legacy); without it, the factor is Eq. (20) of
             *        the TRO paper (FactorType::WNOA).
             */
            Factor(gtsam::Key key_pose0, gtsam::Key key_epsilon0, gtsam::Key key_varpi0, gtsam::Key key_pose1, gtsam::Key key_epsilon1, gtsam::Key key_varpi1, const gtsam::SharedNoiseModel &noise_model, double ds, bool strain_velocity_coupling = true) : Base(noise_model, key_pose0, key_epsilon0, key_varpi0, key_pose1, key_epsilon1, key_varpi1), m_ds(ds), m_coupling(strain_velocity_coupling) {}
            ~Factor() = default;

            /// Whether the velocity row has the ad(eps0)*varpi0 term.
            bool couplesStrainAndVelocity() const { return m_coupling; }

            /**
             * @brief WNOA space prior (on the group), expressed in the T_ib convention:
             *        e = [Log(T0^-1 T1) - ds*eps0 ; eps1 - eps0 ; varpi1 - varpi0 (+ ds*ad(eps0)*varpi0)].
             *        The ad() term, present only with strain-velocity coupling, is the legacy
             *        code's -ds*eps0^curlywedge*varpi0 (T_bi convention), whose sign flips with
             *        eps and varpi in the T_ib convention; it follows from the compatibility
             *        condition d(varpi)/ds = d(eps)/dt - ad(eps)*varpi.
             */
            gtsam::Vector evaluateError(const Pose &pose0, const TangentVector &epsilon0, const TangentVector &varpi0, const Pose &pose1, const TangentVector &epsilon1, const TangentVector &varpi1, gtsam::OptionalMatrixType H_pose0 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_epsilon0 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_varpi0 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_pose1 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_epsilon1 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_varpi1 = gtsam::OptionalMatrixType()) const override
            {
                Eigen::Matrix<double, dim, dim> dT01_dp0, dT01_dp1, dxi01_dT01;
                Pose tran01 = pose0.between(pose1, &dT01_dp0, &dT01_dp1);
                TangentVector xi01 = gtsam::traits<Pose>::Logmap(tran01, &dxi01_dT01);

                TangentVector error_T = xi01 - m_ds * epsilon0;
                TangentVector error_epsilon = epsilon1 - epsilon0;
                TangentVector error_varpi = varpi1 - varpi0;
                if (m_coupling)
                    error_varpi += m_ds * Pose::adjointMap(epsilon0) * varpi0;

                ErrorVector error;
                error << error_T, error_epsilon, error_varpi;

                if (H_pose0)
                {
                    *H_pose0 = (ErrorJacobian() << dxi01_dT01 * dT01_dp0, ZERO(dim), ZERO(dim)).finished();
                }
                if (H_epsilon0)
                {
                    // d(ad(eps0)*varpi0)/d(eps0) = -ad(varpi0), since ad(a)*b = -ad(b)*a
                    Eigen::Matrix<double, dim, dim> dvarpi_deps0 = Eigen::Matrix<double, dim, dim>::Zero();
                    if (m_coupling)
                        dvarpi_deps0 = -m_ds * Pose::adjointMap(varpi0);
                    *H_epsilon0 = (ErrorJacobian() << -m_ds * EYE(dim), -EYE(dim), dvarpi_deps0).finished();
                }
                if (H_varpi0)
                {
                    Eigen::Matrix<double, dim, dim> dvarpi_dvarpi0 = -Eigen::Matrix<double, dim, dim>::Identity();
                    if (m_coupling)
                        dvarpi_dvarpi0 += m_ds * Pose::adjointMap(epsilon0);
                    *H_varpi0 = (ErrorJacobian() << ZERO(dim), ZERO(dim), dvarpi_dvarpi0).finished();
                }
                if (H_pose1)
                {
                    *H_pose1 = (ErrorJacobian() << dxi01_dT01 * dT01_dp1, ZERO(dim), ZERO(dim)).finished();
                }
                if (H_epsilon1)
                {
                    *H_epsilon1 = (ErrorJacobian() << ZERO(dim), EYE(dim), ZERO(dim)).finished();
                }
                if (H_varpi1)
                {
                    *H_varpi1 = (ErrorJacobian() << ZERO(dim), ZERO(dim), EYE(dim)).finished();
                }
                return error;
            }

            /// Node-1 values at which the error is zero given node 0 (used as an initial guess),
            /// for a factor with the given strain-velocity coupling.
            static NodeValues<Pose> getZeroErrorNode(const NodeValues<Pose> &node0, double ds, bool strain_velocity_coupling = true)
            {
                NodeValues<Pose> zero_error_values;
                zero_error_values.pose(node0.pose().retract(ds * node0.epsilon()));
                zero_error_values.epsilon(node0.epsilon());
                if (strain_velocity_coupling)
                    zero_error_values.varpi(node0.varpi() - ds * Pose::adjointMap(node0.epsilon()) * node0.varpi());
                else
                    zero_error_values.varpi(node0.varpi());

                return zero_error_values;
            };

        private:
            double m_ds;
            bool m_coupling;
        };

        /// Gaussian noise model of the space factor over an arclength interval ds (Qs of the paper).
        template <typename Pose>
        gtsam::SharedNoiseModel noiseModel(double ds, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q2, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q4)
        {
            return gtsam::noiseModel::Gaussian::Information(Functions<Pose>::getQsInv(ds, Q2, Q4));
        }

        /// A space factor between two spatially adjacent nodes, with its noise model built from their own spacing.
        template <typename Pose>
        std::shared_ptr<Factor<Pose>> createFactor(std::shared_ptr<Node<Pose>> node0, std::shared_ptr<Node<Pose>> node1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q2, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q4, bool strain_velocity_coupling = true)
        {
            const double ds = node1->arclength - node0->arclength;
            return std::make_shared<Factor<Pose>>(node0->poseKey(), node0->strainKey(), node0->velocityKey(), node1->poseKey(), node1->strainKey(), node1->velocityKey(), noiseModel<Pose>(ds, Q2, Q4), ds, strain_velocity_coupling);
        };

    }
} // namespace Spacetime::Priors::WNOA
#endif // WNOA_BINARYSPACEFACTOR_HPP
