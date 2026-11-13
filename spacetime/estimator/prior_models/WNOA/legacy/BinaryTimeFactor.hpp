#ifndef BINARYTIMEFACTOR_HPP
#define BINARYTIMEFACTOR_HPP

#include <Eigen/Dense>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA::Legacy
{
    /// The paper's time factor between two temporally adjacent WNOA nodes (FactorType::Legacy and FactorType::Paper); see Legacy.
    namespace BinaryTimeFactor
    {
        /// Time factor on (pose, eps, varpi) of nodes 0 and 1, dt apart in time.
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

            /// @param dt Time from node 0 to node 1.
            Factor(gtsam::Key key_pose0, gtsam::Key key_epsilon0, gtsam::Key key_varpi0, gtsam::Key key_pose1, gtsam::Key key_epsilon1, gtsam::Key key_varpi1, const gtsam::SharedNoiseModel &noise_model, double dt) : Base(noise_model, key_pose0, key_epsilon0, key_varpi0, key_pose1, key_epsilon1, key_varpi1), m_dt(dt) {}
            ~Factor() = default;

            /// Paper time prior (on the group), T_ib convention:
            /// e = [Log(T0^-1 T1) - dt*varpi0 ; eps1 - eps0 ; varpi1 - varpi0], with its exact Jacobian.
            gtsam::Vector evaluateError(const Pose &pose0, const TangentVector &epsilon0, const TangentVector &varpi0, const Pose &pose1, const TangentVector &epsilon1, const TangentVector &varpi1, gtsam::OptionalMatrixType H_pose0 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_epsilon0 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_varpi0 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_pose1 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_epsilon1 = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType H_varpi1 = gtsam::OptionalMatrixType()) const override
            {
                Eigen::Matrix<double, dim, dim> dT01_dp0, dT01_dp1, dxi01_dT01;
                Pose tran01 = pose0.between(pose1, &dT01_dp0, &dT01_dp1);
                TangentVector xi01 = gtsam::traits<Pose>::Logmap(tran01, &dxi01_dT01);

                TangentVector error_T = xi01 - m_dt * varpi0;
                TangentVector error_epsilon = epsilon1 - epsilon0;
                TangentVector error_varpi = varpi1 - varpi0;

                ErrorVector error;
                error << error_T, error_epsilon, error_varpi;

                if (H_pose0)
                {
                    *H_pose0 = (ErrorJacobian() << dxi01_dT01 * dT01_dp0, ZERO(dim), ZERO(dim)).finished();
                }
                if (H_epsilon0)
                {
                    *H_epsilon0 = (ErrorJacobian() << ZERO(dim), -EYE(dim), ZERO(dim)).finished();
                }
                if (H_varpi0)
                {
                    *H_varpi0 = (ErrorJacobian() << -m_dt * EYE(dim), ZERO(dim), -EYE(dim)).finished();
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

            /// Node-1 values at which the error is zero given node 0 (used as an initial guess).
            static Values<Pose> getZeroErrorNode(const Values<Pose> &node0, double dt)
            {
                Values<Pose> zero_error_values;
                zero_error_values.pose(node0.pose().retract(dt * node0.varpi()));
                zero_error_values.epsilon(node0.epsilon());
                zero_error_values.varpi(node0.varpi());
                return zero_error_values;
            };

        private:
            double m_dt;
        };

        /// Gaussian noise model of the time factor over a time interval dt (Qt of the paper).
        template <typename Pose>
        gtsam::SharedNoiseModel noiseModel(double dt, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q3)
        {
            return gtsam::noiseModel::Gaussian::Information(Functions<Pose>::getQtInv(dt, Q1, Q3));
        }

        /// A time factor between two temporally adjacent nodes, with its noise model built from their own spacing.
        template <typename Pose>
        std::shared_ptr<Factor<Pose>> createFactor(std::shared_ptr<Node<Pose>> node0, std::shared_ptr<Node<Pose>> node1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q3)
        {
            const double dt = node1->time - node0->time;
            return std::make_shared<Factor<Pose>>(node0->poseKey(), node0->strainKey(), node0->velocityKey(), node1->poseKey(), node1->strainKey(), node1->velocityKey(), noiseModel<Pose>(dt, Q1, Q3), dt);
        };

    }
} // namespace Spacetime::Priors::WNOA::Legacy
#endif // BINARYTIMEFACTOR_HPP
