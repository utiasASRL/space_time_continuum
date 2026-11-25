#ifndef SENSORBASE_HPP
#define SENSORBASE_HPP

#include <vector>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/types/StatefulModel.hpp>
#include <spacetime/estimator/types/States.hpp>
#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/estimator/types/Model.hpp>

namespace Spacetime::Types
{
    /**
     * @brief A model whose measurements each constrain the state of a stateful model (e.g.
     *        a prior) at a single (time, arclength): getNewFactors wraps every due
     *        measurement's getFactor in that model's outer factor there (see
     *        StatefulModel::getOuterFactor). Its measurements are queued through its
     *        Types::MeasurementQueue.
     */
    struct SensorBase : public Model, public MeasurementQueue // Expressed with T_ib frames
    {
    public:
        using pMeasurementType = MeasurementQueue::pMeasurementType; ///< Shared pointer to a queued measurement.
        using pValuesType = std::shared_ptr<NodeValuesBase>;             ///< Shared pointer to type-erased node values.
        using pStatefulType = Types::pStatefulType;                  ///< Shared pointer to the measured stateful model.

        /**
         * @brief Construct a sensor measuring the state of `stateful`, which builds its outer
         *        factors (see StatefulModel::getOuterFactor). Any StatefulType's shared_ptr
         *        (e.g. std::shared_ptr<Priors::WNOA::Prior<Pose>>) upcasts implicitly.
         * @throws std::runtime_error for a null stateful model.
         */
        explicit SensorBase(pStatefulType stateful) : mp_stateful(std::move(stateful))
        {
            if (!mp_stateful)
                throw std::runtime_error("SensorBase requires a non-null stateful model.");
        }

        virtual ~SensorBase() = default;

        /**
         * @brief The inner factor of `measurement`: on (a subset of) the node type's
         *        well-known "interpolated" keys (see Types::InterpolatedKeyMap, e.g.
         *        WNOA::InterpolatedKeys), with its own noise model (typically from
         *        measurement->covariance). This is the `inner_factor` that
         *        StatefulModel::getOuterFactor expects. nullptr drops the measurement.
         */
        virtual gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const = 0;

        /// The stateful model whose state this sensor measures (see SensorBase's constructor).
        const pStatefulType &stateful() const { return mp_stateful; }

        /**
         * @brief Phase 2: drains every queued measurement up to `current_time` and wraps
         *        each one's getFactor in the stateful model's outer factor at the
         *        measurement's (time, arclength). A measurement is dropped, with a warning,
         *        when getFactor or StatefulModel::getOuterFactor returns nullptr.
         */
        Factors getNewFactors(double current_time, gtsam::Values &new_values) override
        {
            (void)new_values;
            Factors factors;
            for (const pMeasurementType &measurement : drainUpTo(current_time))
            {
                // A null factor from either step means this measurement cannot be attached to the
                // graph (see StatefulModel::getOuterFactor -- e.g. the query lies before the
                // estimated state's earliest node, so no bordering node exists to anchor it).
                // Pushing it anyway would put a null shared_ptr into the NonlinearFactorGraph,
                // which segfaults on linearization; drop it with a warning instead.
                auto inner_factor = getFactor(measurement);
                if (!inner_factor)
                {
                    warning() << "Dropping measurement at (t=" << measurement->time
                              << ", s=" << measurement->arclength << "): sensor produced no factor.";
                    continue;
                }

                auto factor = mp_stateful->getOuterFactor(measurement->time, measurement->arclength, inner_factor);
                if (!factor)
                {
                    warning() << "Dropping measurement at (t=" << measurement->time
                              << ", s=" << measurement->arclength
                              << "): outside the estimated state's extent.";
                    continue;
                }

                factors.push_back(factor);
            }
            return factors;
        }

    private:
        pStatefulType mp_stateful;
    };

    /// Types derived from SensorBase. @tparam T The candidate sensor type.
    template <typename T>
    concept SensorType = std::derived_from<T, SensorBase>;
}

#endif // SENSORBASE_HPP