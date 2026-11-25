#ifndef WNOA_EXPERIMENTAL_BINARYTIMEFACTOR_HPP
#define WNOA_EXPERIMENTAL_BINARYTIMEFACTOR_HPP

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h> // before WnoaFactor.h, which uses it without including it
#include <gtsam/nonlinear/WnoaFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA::Experimental
{
    /*
     * The experimental GP prior factors (FactorType::Experimental): the WNOA prior defined on
     * the Lie algebra, built from gtsam's own factors instead of the group factors of
     * WNOA::BinaryTimeFactor / WNOA::BinarySpaceFactor / WNOA::UnaryFactor.
     */

    /// Time factors between two temporally adjacent WNOA nodes (FactorType::Experimental).
    namespace BinaryTimeFactor
    {
        /**
         * @brief A gtsam::WnoaMotionFactor on (pose, varpi) with Q1's diagonal, and a
         *        random-walk gtsam::BetweenFactor on the strain with covariance dt * Q3.
         * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
         */
        template <typename Pose>
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> createFactors(std::shared_ptr<Node<Pose>> node0, std::shared_ptr<Node<Pose>> node1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q1, const Eigen::Matrix<double, Pose::dimension, Pose::dimension> &Q3)
        {
            using TangentVector = typename gtsam::traits<Pose>::TangentVector;
            const double dt = node1->time - node0->time;
            return {
                std::make_shared<gtsam::WnoaMotionFactor<Pose>>(
                    node0->poseKey(),     // pose
                    node0->velocityKey(), // varpi
                    node1->poseKey(),     // pose
                    node1->velocityKey(), // varpi
                    dt,
                    Q1.diagonal()),
                std::make_shared<gtsam::BetweenFactor<TangentVector>>(
                    node0->strainKey(), // epsilon
                    node1->strainKey(), // epsilon
                    TangentVector::Zero(),
                    gtsam::noiseModel::Gaussian::Covariance(dt * Q3))};
        }
    }
} // namespace Spacetime::Priors::WNOA::Experimental
#endif // WNOA_EXPERIMENTAL_BINARYTIMEFACTOR_HPP
