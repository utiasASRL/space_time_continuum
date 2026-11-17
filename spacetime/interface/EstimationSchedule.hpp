/**
 * @file EstimationSchedule.hpp
 * @brief When to estimate: one estimate per `period`, from a start to an end time. Each
 *        estimate adds one node per prior at its time, so the schedule sets the node
 *        spacing. Pure arithmetic; the interface reads its clock
 *        and waits for the returned times however it handles threads.
 *        - Offline, estimates are at exactly start + k * period (computed, not accumulated,
 *          so node spacing is exact with no floating-point drift).
 *        - In real time, estimates are paced to the fixed schedule start + k * period. If
 *          an estimate overruns, the next one runs immediately and the schedule resumes at
 *          the next future boundary: there is no catch-up burst, and the late estimate is
 *          reported by isLate.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_ESTIMATIONSCHEDULE_HPP
#define SPACETIME_INTERFACE_ESTIMATIONSCHEDULE_HPP

#include <cmath>
#include <stdexcept>

namespace Spacetime::Interface
{
    /// The estimate times of one run (see the file comment).
    class EstimationSchedule
    {
    public:
        /**
         * @brief A schedule from `start_time` to `end_time` (inclusive), one estimate per `period` seconds.
         * @param real_time Pace to the clock and skip overrun boundaries (see next()), rather than
         *        return every boundary.
         * @throws std::invalid_argument if `period` is not positive.
         */
        EstimationSchedule(double start_time, double end_time, double period, bool real_time)
            : m_start_time(start_time), m_end_time(end_time), m_period(period), m_real_time(real_time)
        {
            if (!(period > 0.0))
                throw std::invalid_argument("EstimationSchedule requires a positive period.");
        }

        /// The time of the first estimate.
        double firstTime() const { return m_start_time; }

        /// True once `time` is past the end time (the end time itself is estimated).
        bool isDone(double time) const { return time > m_end_time; }

        /**
         * @brief Advances past the estimate just made, given the current clock time `now`.
         * @return Offline: the time of the next estimate. In real time: the time to wait
         *         until before reading the clock for the next estimate -- the next schedule
         *         boundary, or `now` itself if the estimate just made overran it.
         */
        double next(double now)
        {
            ++m_tick;
            if (!m_real_time)
                return boundary(m_tick);

            const double elapsed = now - m_start_time;
            if (elapsed < m_tick * m_period)
                return boundary(m_tick);
            // Overran: move the schedule to the latest boundary already passed, so the
            // following wait lands on a future one.
            m_tick = static_cast<long long>(std::floor(elapsed / m_period));
            return now;
        }

        /**
         * @brief Whether an estimate at `time`, after one at `previous_time`, came later than
         *        the period by more than kLateSlack: estimation is running too slowly. The
         *        slack also absorbs the floating-point drift of an accumulated time.
         */
        bool isLate(double previous_time, double time) const { return time - previous_time > kLateSlack * m_period; }

        /// Gap between estimates, as a multiple of the period, beyond which isLate holds.
        static constexpr double kLateSlack = 1.1;

        /// Index of the schedule boundary the current estimate belongs to.
        long long tick() const { return m_tick; }

        double startTime() const { return m_start_time; } ///< Time of the first estimate, in seconds.
        double endTime() const { return m_end_time; }     ///< Time of the last estimate (inclusive), in seconds.
        double period() const { return m_period; }        ///< Seconds between schedule boundaries.
        bool isRealTime() const { return m_real_time; }   ///< Whether next() paces to a real-time clock.

    private:
        double m_start_time, m_end_time, m_period;
        bool m_real_time;
        long long m_tick = 0;

        double boundary(long long tick) const { return m_start_time + tick * m_period; }
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_ESTIMATIONSCHEDULE_HPP
