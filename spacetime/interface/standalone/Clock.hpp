/**
 * @file Clock.hpp
 * @brief The standalone interface's clocks, which its threads also sleep on: SteppedClock
 *        for offline playback as fast as possible, WallClock for real-time playback.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_CLOCK_HPP
#define SPACETIME_STANDALONE_CLOCK_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>

#include <spacetime/interface/Clock.hpp>

namespace Spacetime::Standalone
{
    /// An Interface::Clock that the standalone threads can (re)start and sleep on.
    struct PacingClock : public Interface::Clock
    {
        /// (Re)starts the clock so that now() reads `time`. Called by the runner as it starts.
        virtual void start(double time) = 0;
        /// Blocks until now() >= `time`, or a stop is requested on `stop`.
        virtual void sleepUntil(double time, std::stop_token stop) = 0;
    };

    /**
     * @brief Offline time, advanced only by the estimator runner: sleepUntil jumps straight
     *        to the requested time, so playback runs as fast as the estimator can go.
     */
    class SteppedClock : public PacingClock
    {
    public:
        void start(double time) override { m_time.store(time); }
        double now() const override { return m_time.load(); }
        /// Advances to `time` (never backwards) without waiting.
        void sleepUntil(double time, std::stop_token) override { m_time.store(std::max(m_time.load(), time)); }
        bool isRealTime() const override { return false; }

    private:
        std::atomic<double> m_time{0.0};
    };

    /// Real time: the wall-clock (steady_clock) time elapsed since start(), offset by the start time.
    class WallClock : public PacingClock
    {
    public:
        WallClock() { start(0.0); }

        void start(double time) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_start_time = time;
            m_start_wall_time = SteadyClock::now();
        }

        double now() const override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_start_time + std::chrono::duration<double>(SteadyClock::now() - m_start_wall_time).count();
        }

        void sleepUntil(double time, std::stop_token stop) override
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            const auto wake_time = m_start_wall_time + std::chrono::duration_cast<SteadyClock::duration>(std::chrono::duration<double>(time - m_start_time));
            // Wakes early only on a stop request; nothing else notifies m_stop.
            m_stop.wait_until(lock, stop, wake_time, []()
                              { return false; });
        }

        bool isRealTime() const override { return true; }

    private:
        using SteadyClock = std::chrono::steady_clock;

        mutable std::mutex m_mutex; // guards the start pair, so now() never mixes two start() calls
        std::condition_variable_any m_stop;
        double m_start_time = 0.0;
        SteadyClock::time_point m_start_wall_time;
    };
} // namespace Spacetime::Standalone

#endif // SPACETIME_STANDALONE_CLOCK_HPP
