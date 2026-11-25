#ifndef ACTUATORBASE_HPP
#define ACTUATORBASE_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/types/ActuatorState.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/estimator/types/Model.hpp>
#include <spacetime/estimator/types/StatefulModel.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Types
{
    /**
     * @brief A model of an input acting along the robot, e.g. tendon tensions: each measured
     *        input becomes an estimated variable u_k, shared by every factor it enters. The
     *        input's measurement uncertainty is then counted once, and correlates those
     *        factors, rather than being counted again at every arclength as independent
     *        sensor measurements of the same reading would be.
     *
     *        A derived actuator overrides one function, getFactors, which only describes the
     *        physics: the inner factors of one input u_k. Each may reference
     *          - the stateful model's interpolated keys at any of the actuator's points
     *            (time, arclengths()[i]), as point-indexed keys (e.g.
     *            WNOA::InterpolatedKeys::Pose(i), see Types::InterpolatedKeyAt),
     *          - u_k's key, and any other genuine variable, which is passed through.
     *        One factor per point models an input acting at each point independently; one
     *        factor on several points models an input acting on all of them at once (e.g.
     *        Actuators::KinematicTendon's length over its routing disks); a factor on no
     *        point constrains u_k alone. The base resolves every factor on the state (see
     *        StatefulModel::getMultiPointOuterFactor) and drops, with a warning, any that
     *        cannot be (e.g. outside the estimated state's extent).
     *
     *        In extendState (phase 1 of Types::Model), every queued measurement up to the
     *        current time adds u_k, timestamped at the measurement, initialized at its value
     *        and tied to it by its covariance. Its factors are added in getNewFactors (phase 2).
     *
     *        Measurement values hold the input, as a column or row vector of input_dim
     *        entries, with an input_dim x input_dim covariance; their arclength is unused.
     *        An input whose factors are all dropped is constrained by its measurement alone.
     *
     *        Since its inputs are variables of the estimation problem, an actuator is a
     *        stateful model: its own Types::ActuatorState holds one input node per input
     *        (see state()), distinct from the state it acts on (see stateful()). Its outer
     *        factors and queries resolve the input nearest in time, so a sensor can measure
     *        its inputs too.
     */
    class ActuatorBase : public StatefulModel<ActuatorState>, public MeasurementQueue
    {
    public:
        using pStatefulType = Types::pStatefulType; ///< Shared pointer to the stateful model acted on.

        /**
         * @param stateful The stateful model (e.g. a prior) whose state the input acts on.
         * @param arclengths Where the input acts [m].
         * @param input_dim The size of every input.
         * @param input_symbol The symbol character of the input keys.
         * @throws std::runtime_error for a null stateful model; std::invalid_argument for no
         *         arclengths or a zero input_dim.
         */
        ActuatorBase(pStatefulType stateful, std::vector<double> arclengths, unsigned int input_dim, unsigned char input_symbol = 'u')
            : StatefulModel<ActuatorState>(std::make_shared<ActuatorState>(input_dim, input_symbol)),
              mp_stateful(std::move(stateful)), m_arclengths(std::move(arclengths))
        {
            if (!mp_stateful)
                throw std::runtime_error("ActuatorBase requires a non-null stateful model.");
            if (m_arclengths.empty())
                throw std::invalid_argument("An actuator needs at least one arclength to act at.");
        }

        virtual ~ActuatorBase() = default;

        /// Inner factors of one input (see getFactors).
        using InnerFactors = std::vector<gtsam::NoiseModelFactor::shared_ptr>;

        /**
         * @brief The inner factors of `measurement`'s input, the variable `input_key`: on
         *        point-indexed interpolated keys of the stateful model's node type (point i
         *        is arclengths()[i]), `input_key` and any other genuine variable, each with
         *        its own (Gaussian) noise model. Null factors are ignored; return none to add
         *        nothing but the input's measurement.
         */
        virtual InnerFactors getFactors(const MeasurementBase &measurement, gtsam::Key input_key) const = 0;

        /**
         * @brief Adds an input node u_k to the actuator's state, with a prior factor from its
         *        measurement's covariance, for every queued measurement up to `time`.
         * @throws std::invalid_argument if a due measurement's value or covariance is not of input_dim.
         */
        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            (void)current_values;
            Factors factors;
            for (const pMeasurementType &measurement : drainUpTo(time))
            {
                validate(*measurement);
                const gtsam::Vector input = measurement->value.reshaped();
                const gtsam::Key key = state()->addInput(km, measurement->time)->inputKey();
                new_values.insert(key, input);
                factors.push_back(std::make_shared<gtsam::PriorFactor<gtsam::Vector>>(
                    key, input, gtsam::noiseModel::Gaussian::Covariance(measurement->covariance)));
                m_pending.emplace_back(measurement, key);
            }
            return factors;
        }

        /// The factors of the inputs added by the last extendState (see getFactors).
        Factors getNewFactors(double time, gtsam::Values &new_values) override
        {
            (void)time, (void)new_values;
            Factors factors;
            for (const auto &[measurement, key] : m_pending)
                for (const auto &inner_factor : getFactors(*measurement, key))
                    if (auto factor = outerFactor(*measurement, inner_factor))
                        factors.push_back(std::move(factor));
            m_pending.clear();
            return factors;
        }

        /// The stateful model whose state this actuator acts on (see ActuatorBase's constructor).
        const pStatefulType &stateful() const { return mp_stateful; }

        /// Arclengths the input acts at [m].
        const std::vector<double> &arclengths() const { return m_arclengths; }

        /// Size of every input.
        unsigned int inputDim() const { return state()->input_dim; }

        /// The (measurement time, key) of every input added so far, oldest first, including marginalized ones (see ActuatorState::inputs).
        std::vector<std::pair<double, gtsam::Key>> inputKeys() const
        {
            std::vector<std::pair<double, gtsam::Key>> keys;
            for (const auto &input : state()->inputs())
                keys.emplace_back(input->time, input->inputKey());
            return keys;
        }

    private:
        /**
         * @brief `inner_factor` resolved on the stateful model's state at `measurement`'s
         *        time (see StatefulModel::getMultiPointOuterFactor); nullptr, with a
         *        warning, if it cannot be. A factor on no point is returned unchanged.
         * @throws std::out_of_range for a point index past arclengths().
         */
        std::shared_ptr<gtsam::NonlinearFactor> outerFactor(const MeasurementBase &measurement,
                                                            const gtsam::NoiseModelFactor::shared_ptr &inner_factor) const
        {
            if (!inner_factor)
                return nullptr;
            const std::vector<std::size_t> points = pointsOf(*inner_factor, m_arclengths.size());
            if (points.empty())
                return inner_factor;
            auto factor = mp_stateful->getMultiPointOuterFactor(measurement.time, m_arclengths, inner_factor);
            if (!factor)
                warning() << "Dropping an input factor at t=" << measurement.time << " on " << points.size()
                          << " point(s) from s=" << m_arclengths[points.front()] << ": it cannot be resolved on the estimated state.";
            return factor;
        }

        /// @throws std::invalid_argument if `measurement` is not an input_dim vector with an input_dim x input_dim covariance.
        void validate(const MeasurementBase &measurement) const
        {
            const Eigen::Index n = state()->input_dim;
            if ((measurement.value.cols() != 1 && measurement.value.rows() != 1) || measurement.value.size() != n)
                throw std::invalid_argument("An actuator input needs " + std::to_string(n) + " entries, got " +
                                            std::to_string(measurement.value.rows()) + "x" + std::to_string(measurement.value.cols()) + ".");
            if (measurement.covariance.rows() != n || measurement.covariance.cols() != n)
                throw std::invalid_argument("An actuator input's covariance must be " + std::to_string(n) + "x" + std::to_string(n) + ".");
        }

        pStatefulType mp_stateful;
        std::vector<double> m_arclengths;
        std::vector<std::pair<pMeasurementType, gtsam::Key>> m_pending; // added by extendState, awaiting getNewFactors
    };

    /// Types derived from ActuatorBase. @tparam T The candidate actuator type.
    template <typename T>
    concept ActuatorType = std::derived_from<T, ActuatorBase>;
} // namespace Spacetime::Types

#endif // ACTUATORBASE_HPP
