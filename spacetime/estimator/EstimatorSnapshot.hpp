/**
 * @file EstimatorSnapshot.hpp
 * @brief Immutable view of an estimate, shared across threads (see Estimator::getSnapshot).
 */

#ifndef ESTIMATORSNAPSHOT_HPP
#define ESTIMATORSNAPSHOT_HPP

#include <memory>
#include <gtsam/nonlinear/Values.h>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/RuntimeStatistics.hpp>

namespace Spacetime
{
    /**
     * @brief An immutable, point-in-time view of the estimator's current values,
     *        published by Estimator::estimate() / finalize() under a mutex (see
     *        Estimator::getSnapshot()) and read from any other thread via a
     *        cheap shared_ptr copy -- the payload itself is never mutated once
     *        published, so reading it needs no further locking.
     */
    struct EstimatorSnapshot
    {
        double time;          ///< Estimation time the snapshot was published for.
        gtsam::Values values; ///< Estimated values, including locked constants.
        /// Posterior covariance blocks at this estimate; nullptr unless the estimator was
        /// configured with EstimatorOptions::compute_covariance.
        std::shared_ptr<const PosteriorCovariance> covariance = nullptr;
        /// The estimator's phase timings up to and including this estimate.
        RuntimeStatistics runtime{};
        /// Whether the estimator extracts covariances (EstimatorOptions::compute_covariance).
        bool covariance_enabled = false;
        /// Whether the estimator runs in batch mode (see Estimator::isBatch): graph construction is
        /// then timed once per estimate() but the solve once by finalize().
        bool batch = false;
    };
} // namespace Spacetime
#endif // ESTIMATORSNAPSHOT_HPP
