#ifndef MEASUREMENTHISTORY_HPP
#define MEASUREMENTHISTORY_HPP

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include <spacetime/estimator/types/SensorBase.hpp>

namespace Spacetime::Visualization
{
    /**
     * @brief Bounded record of every measurement added to a sensor, for a sensor's
     *        visualization to draw its last-received value at any playback time.
     *        Fed by Types::SensorBase::addMeasurementCallback rather than the sensor's
     *        live measurement queue, which is drained/consumed by getNewFactors and
     *        holds no history -- so this works for both live playback (recent
     *        entries) and offline scrubbing/rewind across the whole recorded range,
     *        and for a sensor that's never attached to an estimator (e.g. a
     *        ground-truth sensor). Subscribing replays whatever the sensor still has
     *        queued, so construct this before the sensor's data is consumed.
     */
    class MeasurementHistory
    {
    public:
        using pMeasurementType = Types::SensorBase::pMeasurementType;

        /// Subscribes to `sensor`, replaying the measurements it still has queued. Keeps at most `capacity` entries, dropping the oldest.
        explicit MeasurementHistory(Types::SensorBase &sensor, std::size_t capacity = 10000)
            : m_state(std::make_shared<State>())
        {
            m_state->capacity = capacity;
            sensor.addMeasurementCallback([state = m_state](const pMeasurementType &measurement)
                                          { state->record(measurement); });
        }

        /// The most recent recorded measurement with `.time <= time`, or nullptr if none exists yet.
        pMeasurementType latestMeasurementAt(double time) const
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            for (auto it = m_state->measurements.rbegin(); it != m_state->measurements.rend(); ++it)
                if ((*it)->time <= time)
                    return *it;
            return nullptr;
        }

        /// Every recorded measurement with `begin <= .time <= end`, in the order received (empty if `begin > end`).
        std::vector<pMeasurementType> measurementsBetween(double begin, double end) const
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            std::vector<pMeasurementType> out;
            for (const auto &measurement : m_state->measurements)
                if (measurement->time >= begin && measurement->time <= end)
                    out.push_back(measurement);
            return out;
        }

        /// Number of measurements currently recorded (at most the capacity).
        std::size_t size() const
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            return m_state->measurements.size();
        }

    private:
        /// Shared with the sensor's callback, so the callback never dangles if this history is destroyed first.
        struct State
        {
            /// Appends `measurement`, dropping the oldest entry once over capacity.
            void record(const pMeasurementType &measurement)
            {
                std::lock_guard<std::mutex> lock(mutex);
                measurements.push_back(measurement);
                if (measurements.size() > capacity)
                    measurements.pop_front();
            }

            mutable std::mutex mutex;
            std::deque<pMeasurementType> measurements;
            std::size_t capacity = 0;
        };
        std::shared_ptr<State> m_state;
    };
} // namespace Spacetime::Visualization

#endif // MEASUREMENTHISTORY_HPP
