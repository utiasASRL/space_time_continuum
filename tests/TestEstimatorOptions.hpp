#ifndef TEST_ESTIMATOR_OPTIONS_HPP
#define TEST_ESTIMATOR_OPTIONS_HPP

// Estimator options for tests, with every field set explicitly so that tests do not
// depend on EstimatorOptions' defaults or on Estimator's convenience constructors.
// Tests build their estimators with Estimator(Testing::estimatorOptions(...)) and
// override the fields they care about. When a field is added to EstimatorOptions, set
// it here as well.

#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

#include <spacetime/estimator/Estimator.hpp>

namespace Spacetime::Testing
{
    /// GTSAM's Levenberg-Marquardt parameters, ordered with METIS.
    inline gtsam::LevenbergMarquardtParams lmParams()
    {
        gtsam::LevenbergMarquardtParams params;
        params.orderingType = gtsam::Ordering::METIS;
        return params;
    }

    /**
     * @param smootherLag The fixed-lag window [s], or Estimator::BATCH_LAG for batch mode.
     * @param lm_params Levenberg-Marquardt parameters of the batch solve and batch smoother.
     * @return Options using the batch fixed-lag smoother (in sliding-window mode), GTSAM's
     *         iSAM2 parameters, no covariance extraction and no quasi-static estimation.
     */
    inline EstimatorOptions estimatorOptions(double smootherLag, const gtsam::LevenbergMarquardtParams &lm_params = lmParams())
    {
        EstimatorOptions options;
        options.smootherLag = smootherLag;
        options.smoother = EstimatorOptions::Smoother::Batch;
        options.lm_params = lm_params;
        options.isam2_params = gtsam::ISAM2Params();
        options.compute_covariance = false;
        options.quasi_static = false;
        return options;
    }
}

#endif
