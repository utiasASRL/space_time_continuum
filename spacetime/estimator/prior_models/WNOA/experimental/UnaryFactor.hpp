#ifndef WNOA_EXPERIMENTAL_UNARYFACTOR_HPP
#define WNOA_EXPERIMENTAL_UNARYFACTOR_HPP

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>

#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA::Experimental
{
    /// Unary (boundary) factors on a WNOA node (FactorType::Experimental).
    namespace UnaryFactor
    {
        /**
         * @brief Separate gtsam::PriorFactor%s on the node's pose, strain and velocity, from
         *        the three diagonal blocks of Q0 (its off-diagonal blocks are ignored).
         *        gtsam::PriorFactor<Pose> uses an identity Jacobian, which is inaccurate far
         *        from the reference; WNOA::UnaryFactor uses the exact one.
         * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
         */
        template <typename Pose>
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> createFactors(std::shared_ptr<Node<Pose>> node, const Values<Pose> &reference, const Eigen::Matrix<double, 3 * Pose::dimension, 3 * Pose::dimension> &Q0)
        {
            using TangentVector = typename gtsam::traits<Pose>::TangentVector;
            constexpr int dim = Pose::dimension;
            return {
                std::make_shared<gtsam::PriorFactor<Pose>>(
                    node->poseKey(), // pose
                    reference.pose(),
                    gtsam::noiseModel::Gaussian::Covariance(Q0.template block<dim, dim>(0, 0))),
                std::make_shared<gtsam::PriorFactor<TangentVector>>(
                    node->strainKey(), // epsilon
                    reference.epsilon(),
                    gtsam::noiseModel::Gaussian::Covariance(Q0.template block<dim, dim>(dim, dim))),
                std::make_shared<gtsam::PriorFactor<TangentVector>>(
                    node->velocityKey(), // varpi
                    reference.varpi(),
                    gtsam::noiseModel::Gaussian::Covariance(Q0.template block<dim, dim>(dim * 2, dim * 2)))};
        }
    }
} // namespace Spacetime::Priors::WNOA::Experimental
#endif // WNOA_EXPERIMENTAL_UNARYFACTOR_HPP
