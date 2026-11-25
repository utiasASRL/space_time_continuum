#ifndef MEASUREMENTQUEUE_HPP
#define MEASUREMENTQUEUE_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

#include <spacetime/estimator/types/Measurement.hpp>

namespace Spacetime::Types
{
    /**
     * @brief The time-ordered queue of a model's incoming measurements (see Types::SensorBase
     *        and Types::ActuatorBase), filled by addMeasurement, possibly from another thread
     *        (see Interface::Dataloader), and consumed by the estimator thread through
     *        drainUpTo. Every member is thread-safe.
     */
    struct MeasurementQueue
    {
        using pMeasurementType = std::shared_ptr<MeasurementBase>; ///< Shared pointer to a queued measurement.
        /// Observer of every measurement added to this queue (see addMeasurementCallback).
        using MeasurementCallback = std::function<void(const pMeasurementType &)>;

        virtual ~MeasurementQueue() = default;

        /**
         * @brief Queues `measurement` (thread-safe) and notifies every registered callback.
         *        Measurements are expected in time order: drainUpTo stops at the first one
         *        newer than its time, so an out-of-order one delays those queued after it.
         * @tparam T The measurement's type, derived from MeasurementBase.
         */
        template <MeasurementType T>
        void addMeasurement(const std::shared_ptr<T> measurement)
        {
            push(std::static_pointer_cast<MeasurementBase>(measurement), 0);
        }

        /**
         * @brief A measurement taken out by withholdAll, with the number of callbacks that
         *        had already seen it (the callbacks registered by then, which every queued
         *        measurement has been passed to).
         */
        struct Withheld
        {
            pMeasurementType measurement; ///< The withheld measurement.
            std::size_t notified = 0;     ///< How many of the queue's callbacks (in registration order) have seen it.
        };

        /**
         * @brief Removes and returns every queued measurement, oldest first, to hand back later
         *        with redeliver (e.g. by Interface::Standalone::FileDataloader, which replays
         *        pre-loaded data as if it arrived live). Calls no callback.
         */
        std::vector<Withheld> withholdAll()
        {
            std::vector<Withheld> all;
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            all.reserve(m_measurement_queue.size());
            for (; !m_measurement_queue.empty(); m_measurement_queue.pop())
                all.push_back({m_measurement_queue.front(), m_measurement_callbacks.size()});
            return all;
        }

        /**
         * @brief Queues a measurement taken out by withholdAll again, notifying only the
         *        callbacks registered since: the others saw it while it was first queued, so
         *        every callback sees each measurement exactly once, whenever it registered.
         */
        void redeliver(const Withheld &withheld)
        {
            push(withheld.measurement, withheld.notified);
        }

        /**
         * @brief Registers `callback` to be invoked with every measurement added from
         *        now on, after it is queued. On registration it is first invoked with
         *        each measurement still queued (oldest first), so an
         *        observer attached after the data was loaded but before it was
         *        consumed still sees all of it. Invoked with this queue's lock held, so
         *        a callback must not call back into it. Used e.g. by a visualizer to keep
         *        a measurement history (see Visualization::MeasurementHistory).
         */
        void addMeasurementCallback(MeasurementCallback callback)
        {
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            for (std::queue<pMeasurementType> queued = m_measurement_queue; !queued.empty(); queued.pop())
                callback(queued.front());
            m_measurement_callbacks.push_back(std::move(callback));
        }

        /// Removes and returns, oldest first, the measurements at the front of the queue up to and including `time`.
        std::vector<pMeasurementType> drainUpTo(double time)
        {
            std::vector<pMeasurementType> ready;
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            while (!m_measurement_queue.empty() && m_measurement_queue.front()->time <= time)
            {
                ready.push_back(m_measurement_queue.front());
                m_measurement_queue.pop();
            }
            return ready;
        }

        /// Removes and returns every queued measurement, oldest first, without notifying the callbacks (to hand them back later, see withholdAll).
        std::vector<pMeasurementType> takeAll()
        {
            std::vector<pMeasurementType> all;
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            all.reserve(m_measurement_queue.size());
            for (; !m_measurement_queue.empty(); m_measurement_queue.pop())
                all.push_back(m_measurement_queue.front());
            return all;
        }

        /// Copies of the queued measurements, oldest first, leaving the queue untouched.
        std::vector<pMeasurementType> peekAll() const
        {
            std::vector<pMeasurementType> all;
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            all.reserve(m_measurement_queue.size());
            for (std::queue<pMeasurementType> queued = m_measurement_queue; !queued.empty(); queued.pop())
                all.push_back(queued.front());
            return all;
        }

        /// The number of queued measurements.
        std::size_t size() const
        {
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            return m_measurement_queue.size();
        }

        /// True if no measurement is queued.
        bool empty() const
        {
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            return m_measurement_queue.empty();
        }

        /// The time of the newest queued measurement, or nullopt if none is queued.
        std::optional<double> lastTime() const
        {
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            if (m_measurement_queue.empty())
                return std::nullopt;
            return m_measurement_queue.back()->time;
        }

        /// The time of the oldest queued measurement (the next drainUpTo returns), or nullopt if none is queued.
        std::optional<double> frontTime() const
        {
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            if (m_measurement_queue.empty())
                return std::nullopt;
            return m_measurement_queue.front()->time;
        }

    private:
        /// Queues `measurement` and passes it to every callback from index `first_callback` on.
        void push(pMeasurementType measurement, std::size_t first_callback)
        {
            std::lock_guard<std::mutex> lock(m_measurement_mutex);
            m_measurement_queue.push(measurement);
            for (std::size_t i = first_callback; i < m_measurement_callbacks.size(); i++)
                m_measurement_callbacks[i](measurement);
        }

        std::queue<pMeasurementType> m_measurement_queue; // oldest first (expected in time order)
        mutable std::mutex m_measurement_mutex;
        std::vector<MeasurementCallback> m_measurement_callbacks;
    };

    /// Shared pointer to a MeasurementQueue.
    using pMeasurementQueueType = std::shared_ptr<MeasurementQueue>;
} // namespace Spacetime::Types

#endif // MEASUREMENTQUEUE_HPP
