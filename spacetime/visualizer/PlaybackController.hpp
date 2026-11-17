/**
 * @file PlaybackController.hpp
 * @brief Backend-agnostic play/pause/seek state machine for a visualizer.
 * @author Spencer Teetaert
 */

#ifndef PLAYBACKCONTROLLER_HPP
#define PLAYBACKCONTROLLER_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace Spacetime::Visualization
{
    /// Whether PlaybackController::tick() advances the displayed time.
    enum class PlaybackState
    {
        Paused, ///< tick() leaves the displayed time unchanged.
        Playing ///< tick() advances (offline) or tracks the live time (online).
    };

    /**
     * @brief Pure play/pause/seek/speed state machine driving what time a
     *        Visualizer should render on each frame. Deliberately has no
     *        Qt/VTK dependency so it can be unit-tested standalone.
     *
     *        Offline mode: play()/tick() advances displayedTime() by
     *        (wall_dt * speed); pause() freezes it; seek()/setSpeed() are
     *        available.
     *
     *        Online mode: pause() freezes displayedTime() regardless of the
     *        live time source; the first tick() after play() snaps
     *        displayedTime() to the live time source's current value rather
     *        than resuming from the paused value -- "pause holds position,
     *        play jumps to now."
     *
     *        Not thread-safe: use from one thread (the backend's GUI thread). Times are in
     *        seconds of estimation time.
     */
    class PlaybackController
    {
    public:
        /// Starts paused at time 0. @param online Selects online (live-tracking) vs. offline (replay) behavior; fixed for the controller's lifetime.
        explicit PlaybackController(bool online) : m_online(online) {}

        /// While playing, advances the displayed time by wall_dt * speed (offline; clamped to the time bounds, auto-pausing at the end -- or, if looping(), showing the end and restarting from the beginning on the next tick) or sets it to the live time source (online; unchanged if none is set). Returns the new displayedTime().
        double tick(double wall_dt)
        {
            if (m_state != PlaybackState::Playing)
                return m_displayed_time;

            if (m_online)
            {
                // Every tick while playing tracks the live time source directly (no
                // running total kept in this branch), so the very first tick after
                // play() naturally "jumps" to the current live time rather than
                // resuming from whatever displayedTime() held while paused.
                m_displayed_time = m_live_time_source ? m_live_time_source() : m_displayed_time;
            }
            else if (m_looping && m_displayed_time >= m_max_time)
            {
                m_displayed_time = m_min_time; // the end was shown last tick: replay
            }
            else
            {
                m_displayed_time += wall_dt * m_speed;
                if (m_displayed_time >= m_max_time)
                {
                    m_displayed_time = m_max_time;
                    if (!m_looping)
                        m_state = PlaybackState::Paused;
                }
                else if (m_displayed_time < m_min_time)
                {
                    m_displayed_time = m_min_time;
                }
            }
            return m_displayed_time;
        }

        /// Offline only: if playback is sitting at (or past) the end of the range, jumping back into Playing restarts from the beginning instead of doing nothing.
        void play()
        {
            if (!m_online && m_displayed_time >= m_max_time)
                m_displayed_time = m_min_time;
            m_state = PlaybackState::Playing;
        }

        /// Freezes displayedTime() (in both modes).
        void pause() { m_state = PlaybackState::Paused; }

        /// Offline only: sets displayedTime() to `time`, clamped to the time bounds. No-op in online mode, where displayed time always tracks the live time source while playing.
        void seek(double time)
        {
            if (!m_online)
                m_displayed_time = std::clamp(time, m_min_time, m_max_time);
        }

        /**
         * @brief Offline only: moves to the adjacent estimation tick (min_time + k * period,
         *        see setTimeBounds) in `direction`'s sign: one tick over from a tick, or the
         *        nearest one that way from between ticks. Within a rounding error of a tick
         *        counts as on it. Clamped to the time bounds like seek(). No-op online, or
         *        for a non-positive `period` or a zero `direction`.
         */
        void stepTick(double period, int direction)
        {
            if (m_online || !(period > 0.0) || direction == 0)
                return;
            const double ticks = (m_displayed_time - m_min_time) / period;
            const double nearest = std::round(ticks);
            const bool on_tick = std::abs(ticks - nearest) <= 1e-9 * std::max(1.0, std::abs(ticks));
            const double target = direction > 0 ? (on_tick ? nearest + 1.0 : std::ceil(ticks))
                                                 : (on_tick ? nearest - 1.0 : std::floor(ticks));
            seek(m_min_time + target * period);
        }

        /// Clamps subsequent seek()/tick() to [min_time, max_time]. Stored in both modes but only used offline. Does not re-clamp the current time.
        void setTimeBounds(double min_time, double max_time)
        {
            m_min_time = min_time;
            m_max_time = max_time;
        }

        /// Offline only: whether reaching the end of the time bounds replays from the beginning instead of pausing (off by default). Ignored in online mode.
        void setLooping(bool looping) { m_looping = looping && !m_online; }

        /// See setLooping.
        bool looping() const { return m_looping; }

        /// Offline only: playback-rate multiplier on wall-clock time. Ignored in online mode.
        void setSpeed(double speed)
        {
            if (!m_online)
                m_speed = speed;
        }

        /// Online only: the function tick() reads the displayed time from while playing.
        void setLiveTimeSource(std::function<double()> now_fn) { m_live_time_source = std::move(now_fn); }

        /// Upper bound for setDelay() (e.g. the estimator's fixed-lag window). Negative values are treated as 0; re-clamps the current delay.
        void setMaxDelay(double max_delay)
        {
            m_max_delay = std::max(0.0, max_delay);
            setDelay(m_delay);
        }

        /// Phase delay subtracted from displayedTime() to get queryTime(). Clamped to [0, max delay]; available in both modes, playing or paused.
        void setDelay(double delay) { m_delay = std::clamp(delay, 0.0, m_max_delay); }

        /// Current phase delay [s], see setDelay.
        double delay() const { return m_delay; }
        /// Upper bound on the phase delay [s]; infinite until setMaxDelay is called.
        double maxDelay() const { return m_max_delay; }

        /// The time objects should actually be drawn at: displayedTime() - delay(). displayedTime() itself (and so seeking, the timeline, and end-of-range auto-pause) is unaffected by the delay.
        double queryTime() const { return m_displayed_time - m_delay; }

        /// Playing or paused.
        PlaybackState state() const { return m_state; }
        /// The playback ("now") time, before the phase delay; see queryTime.
        double displayedTime() const { return m_displayed_time; }
        /// Offline playback-rate multiplier (1 = real time).
        double speed() const { return m_speed; }
        /// The mode chosen at construction.
        bool isOnline() const { return m_online; }

    private:
        bool m_online;
        PlaybackState m_state = PlaybackState::Paused;
        double m_displayed_time = 0.0;
        double m_speed = 1.0;
        bool m_looping = false;
        double m_min_time = 0.0;
        double m_max_time = std::numeric_limits<double>::infinity();
        double m_delay = 0.0;
        double m_max_delay = std::numeric_limits<double>::infinity();
        std::function<double()> m_live_time_source;
    };
} // namespace Spacetime::Visualization

#endif // PLAYBACKCONTROLLER_HPP
