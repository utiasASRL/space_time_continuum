/**
 * @file RosClock.hpp
 * @brief Estimation time from a ROS2 clock: live (system/ROS time) or replayed (use_sim_time,
 *        e.g. a rosbag played with --clock).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_ROSCLOCK_HPP
#define SPACETIME_ROS2_ROSCLOCK_HPP

#include <atomic>
#include <cstdint>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/clock.hpp>
#include <rclcpp/time.hpp>

#include <spacetime/interface/Clock.hpp>

namespace Spacetime::Interface::Ros2
{
    /**
     * @brief An Interface::Clock reading a ROS2 clock (typically a node's, so it follows
     *        use_sim_time), relative to an origin: estimation time is
     *        `start_time + (ros_time - origin)` seconds, which keeps the estimator's times
     *        small (ROS time counts from the epoch). The origin is the clock's first
     *        non-zero reading at tryStart(): a sim-time clock reads zero until the first
     *        /clock message. Thread-safe.
     */
    class RosClock : public Interface::Clock
    {
    public:
        /// Unstarted, reading `start_time` (seconds) until tryStart() succeeds. Throws std::invalid_argument if `clock` is null.
        explicit RosClock(rclcpp::Clock::SharedPtr clock, double start_time = 0.0);

        /// Sets the origin to the clock's current reading, unless it reads zero (sim time
        /// before the first /clock message) or the origin is set already. Returns started().
        bool tryStart();

        /// Whether the origin is set.
        bool started() const { return m_started.load(std::memory_order_acquire); }

        /// The current estimation time; the start time until started.
        double now() const override;

        /// Always: estimation follows ROS time, live or replayed.
        bool isRealTime() const override { return true; }

        /// The estimation time of a ROS stamp. Throws std::logic_error before started.
        double toEstimatorTime(const rclcpp::Time &stamp) const;

        /// The ROS stamp of an estimation time. Throws std::logic_error before started.
        builtin_interfaces::msg::Time toRosTime(double time) const;

        /// The ROS time the start time corresponds to. Throws std::logic_error before started.
        rclcpp::Time origin() const;

        /// The estimation time of the origin, in seconds.
        double startTime() const { return m_start_time; }

        /// The underlying ROS clock.
        const rclcpp::Clock::SharedPtr &rosClock() const { return mp_clock; }

    private:
        rclcpp::Clock::SharedPtr mp_clock;
        double m_start_time;
        /// The origin in nanoseconds of the clock's time source (compared as plain integers,
        /// since rclcpp refuses arithmetic between times of different clock types).
        std::atomic<int64_t> m_origin_ns{0};
        std::atomic<bool> m_started{false};

        /// Throws std::logic_error naming `caller` if not started.
        void requireStarted(const char *caller) const;
    };
} // namespace Spacetime::Interface::Ros2

#endif // SPACETIME_ROS2_ROSCLOCK_HPP
