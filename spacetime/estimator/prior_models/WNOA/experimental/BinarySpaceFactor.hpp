#ifndef WNOA_EXPERIMENTAL_BINARYSPACEFACTOR_HPP
#define WNOA_EXPERIMENTAL_BINARYSPACEFACTOR_HPP

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/WnoaFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA::Experimental
{
    /// Space factors between two arclength-adjacent WNOA nodes (FactorType::Experimental).
    namespace BinarySpaceFactor
    {
        /**
         * @brief A gtsam::WnoaMotionFactor on (pose, eps) with Q2's diagonal, and a
         *        random-walk gtsam::BetweenFactor on the velocity with covariance ds * Q4.
         * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
         */
        template <typename Pose>
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> createFactors(std::shared_ptr<Node<Pose>> node0, std::shared_ptr<Node<Pose>> node1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q2, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q4)
        {
            using TangentVector = typename gtsam::traits<Pose>::TangentVector;
            const double ds = node1->arclength - node0->arclength;
            return {
                std::make_shared<gtsam::WnoaMotionFactor<Pose>>(
                    node0->poseKey(),   // pose
                    node0->strainKey(), // epsilon
                    node1->poseKey(),   // pose
                    node1->strainKey(), // epsilon
                    ds,
                    Q2.diagonal()),
                std::make_shared<gtsam::BetweenFactor<TangentVector>>(
                    node0->velocityKey(), // varpi
                    node1->velocityKey(), // varpi
                    TangentVector::Zero(),
                    gtsam::noiseModel::Gaussian::Covariance(ds * Q4))};
        }
    }
} // namespace Spacetime::Priors::WNOA::Experimental
#endif // WNOA_EXPERIMENTAL_BINARYSPACEFACTOR_HPP
