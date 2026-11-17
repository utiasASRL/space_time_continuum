#ifndef WNOA_UNARYFACTOR_HPP
#define WNOA_UNARYFACTOR_HPP

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

namespace Spacetime::Priors::WNOA
{
    /// The paper's unary (boundary) factor on a WNOA node (FactorType::WNOA and FactorType::Legacy); see BinarySpaceFactor.
    namespace UnaryFactor
    {
        /**
         * @brief Unary factor tying a node's (pose, eps, varpi) to reference values.
         * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
         */
        template <typename Pose>
        // using Pose = gtsam::Pose3; // For now, we will only support Pose3, but this can be templatized in the future if needed
        class Factor : public gtsam::NoiseModelFactorN<Pose, typename gtsam::traits<Pose>::TangentVector, typename gtsam::traits<Pose>::TangentVector>
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
            using Base = gtsam::NoiseModelFactorN<Pose, TangentVector, TangentVector>;
            using ErrorVector = Eigen::Vector<double, 3 * dim>;
            using ErrorJacobian = Eigen::Matrix<double, 3 * dim, dim>;

            /**
             * @param key_pose,key_epsilon,key_varpi The node's pose, strain and velocity keys.
             * @param noise_model Covariance over [pose, eps, varpi].
             * @param node0 Reference values the node is tied to.
             */
            Factor(gtsam::Key key_pose, gtsam::Key key_epsilon, gtsam::Key key_varpi, const gtsam::SharedNoiseModel &noise_model, const Values<Pose> &node0) : Base(noise_model, key_pose, key_epsilon, key_varpi), m_reference_values(node0) {};
            ~Factor() = default;

            /// e = [Log(T^-1 T_ref) ; eps - eps_ref ; varpi - varpi_ref], with its exact Jacobian.
            gtsam::Vector evaluateError(const Pose &pose, const TangentVector &epsilon, const TangentVector &varpi, gtsam::OptionalMatrixType Hpose = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType Hepsilon = gtsam::OptionalMatrixType(), gtsam::OptionalMatrixType Hvarpi = gtsam::OptionalMatrixType()) const override
            {
                Eigen::Matrix<double, dim, dim> dT_dp, de_dT;
                Pose T = pose.between(m_reference_values.pose(), &dT_dp);
                ErrorVector error;
                error << gtsam::traits<Pose>::Logmap(T, &de_dT),
                    epsilon - m_reference_values.epsilon(),
                    varpi - m_reference_values.varpi();

                if (Hpose)
                {
                    *Hpose = (ErrorJacobian() << de_dT * dT_dp, ZERO(dim), ZERO(dim)).finished();
                }
                if (Hepsilon)
                {
                    *Hepsilon = (ErrorJacobian() << ZERO(dim), EYE(dim), ZERO(dim)).finished();
                }
                if (Hvarpi)
                {
                    *Hvarpi = (ErrorJacobian() << ZERO(dim), ZERO(dim), EYE(dim)).finished();
                }
                return error;
            }

            /// The reference values, at which the error is zero.
            Values<Pose> getZeroErrorNode() const
            {
                Values<Pose> zero_error_values;
                zero_error_values.pose(m_reference_values.pose());
                zero_error_values.epsilon(m_reference_values.epsilon());
                zero_error_values.varpi(m_reference_values.varpi());
                return zero_error_values;
            };

        private:
            const Values<Pose> m_reference_values;
        };

        /// A unary factor tying a node to `node_reference`, with the given 3*dim x 3*dim (18 x 18 for Pose3) covariance of [pose, eps, varpi].
        template <typename Pose>
        std::shared_ptr<Factor<Pose>> createFactor(std::shared_ptr<Node<Pose>> node, const Values<Pose> &node_reference, const Eigen::Matrix<double, 3 * Pose::dimension, 3 * Pose::dimension> &covariance)
        {
            return std::make_shared<Factor<Pose>>(node->poseKey(), node->strainKey(), node->velocityKey(), gtsam::noiseModel::Gaussian::Covariance(covariance), node_reference);
        };
    }
} // namespace Spacetime::Priors::WNOA

#endif // WNOA_UNARYFACTOR_HPP