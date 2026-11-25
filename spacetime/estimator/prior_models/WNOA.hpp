/**
 * @file WNOA.hpp
 * @brief Umbrella header for the WNOA (white-noise-on-acceleration) prior: state, prior, interpolator, its factors (WNOA/ and WNOA/experimental/) and InterpolationFactor.
 */

#ifndef WNOA_HPP
#define WNOA_HPP

#include <spacetime/estimator/prior_models/WNOA/Covariances.hpp>
#include <spacetime/estimator/prior_models/WNOA/BinarySpaceFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/BinaryTimeFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/UnaryFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/experimental/BinarySpaceFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/experimental/BinaryTimeFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/experimental/UnaryFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>
#include <spacetime/estimator/prior_models/WNOA/InterpolatedKeys.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAPrior.hpp>
#include <spacetime/estimator/prior_models/WNOA/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/Interpolator.hpp>
#include <spacetime/estimator/prior_models/InterpolationFactor.hpp>

#endif // WNOA_HPP