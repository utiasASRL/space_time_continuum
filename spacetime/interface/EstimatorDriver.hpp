/**
 * @file EstimatorDriver.hpp
 * @brief Steps the estimator, whatever calls it: the standalone interface's estimator
 *        thread (see Standalone::EstimatorRunner), or e.g. a ROS2 node's timer callback.
 *        Results are read from any other thread through Estimator::getSnapshot().
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_ESTIMATORDRIVER_HPP
#define SPACETIME_INTERFACE_ESTIMATORDRIVER_HPP

#include <functional>
#include <stdexcept>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Interface
{
    /// One estimate per step(), plus a per-estimate callback (see the file comment).
    class EstimatorDriver
    {
    public:
        /// Called with the time of each estimate, right after it, on the caller's thread. Keep it cheap.
        using TickCallback = std::function<void(double)>;

        /**
         * @brief Drives `estimator`, which must outlive the driver.
         * @param real_time Whether estimation follows a real-time clock.
         * @throws std::invalid_argument for a batch-mode estimator (see Estimator::isBatch)
         *         in real time: it only solves once, in finish().
         */
        EstimatorDriver(Estimator &estimator, bool real_time) : m_estimator(estimator), m_real_time(real_time)
        {
            if (real_time && estimator.isBatch())
                throw std::invalid_argument("EstimatorDriver: a batch-mode estimator (smootherLag = -1) cannot run in real time.");
        }

        /// Sets the per-estimate callback (see TickCallback).
        void setTickCallback(TickCallback callback) { m_tick_callback = std::move(callback); }

        /// Estimates at `time`, then calls the tick callback.
        void step(double time)
        {
            debug() << "Running estimator at time: " << time;
            m_estimator.estimate(time);
            if (m_tick_callback)
                m_tick_callback(time);
        }

        /// Ends the run: a batch-mode estimator solves the whole accumulated problem (see Estimator::finalize).
        void finish() { m_estimator.finalize(); }

        bool isRealTime() const { return m_real_time; }

    private:
        Estimator &m_estimator;
        bool m_real_time;
        TickCallback m_tick_callback;
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_ESTIMATORDRIVER_HPP
