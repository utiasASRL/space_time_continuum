#include <cmath>
#include <stdexcept>
#include <string>

#include "RosClock.hpp"

namespace Spacetime::Ros2
{
    namespace
    {
        constexpr double kNanoseconds = 1e9;
    }

    RosClock::RosClock(rclcpp::Clock::SharedPtr clock, double start_time) : mp_clock(std::move(clock)), m_start_time(start_time)
    {
        if (!mp_clock)
            throw std::invalid_argument("Ros2::RosClock requires a non-null clock.");
    }

    bool RosClock::tryStart()
    {
        if (started())
            return true;
        const int64_t now_ns = mp_clock->now().nanoseconds();
        if (now_ns == 0)
            return false;
        int64_t expected = 0;
        m_origin_ns.compare_exchange_strong(expected, now_ns); // the first caller wins
        m_started.store(true, std::memory_order_release);
        return true;
    }

    double RosClock::now() const
    {
        if (!started())
            return m_start_time;
        return m_start_time + static_cast<double>(mp_clock->now().nanoseconds() - m_origin_ns.load()) / kNanoseconds;
    }

    double RosClock::toEstimatorTime(const rclcpp::Time &stamp) const
    {
        requireStarted("toEstimatorTime");
        return m_start_time + static_cast<double>(stamp.nanoseconds() - m_origin_ns.load()) / kNanoseconds;
    }

    builtin_interfaces::msg::Time RosClock::toRosTime(double time) const
    {
        requireStarted("toRosTime");
        const int64_t ns = m_origin_ns.load() + std::llround((time - m_start_time) * kNanoseconds);
        return rclcpp::Time(ns, mp_clock->get_clock_type());
    }

    rclcpp::Time RosClock::origin() const
    {
        requireStarted("origin");
        return rclcpp::Time(m_origin_ns.load(), mp_clock->get_clock_type());
    }

    void RosClock::requireStarted(const char *caller) const
    {
        if (!started())
            throw std::logic_error(std::string("Ros2::RosClock::") + caller + ": the clock has not started (see tryStart).");
    }
} // namespace Spacetime::Ros2
