#ifndef TESTS_ROS2_ROS2TESTUTILITIES_HPP
#define TESTS_ROS2_ROS2TESTUTILITIES_HPP

#include <atomic>
#include <chrono>
#include <functional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "../ProcessId.hpp"

/// Helpers shared by the ROS2 interface tests (see main.cpp for the rclcpp context).
namespace Spacetime::Ros2Tests
{
    using namespace std::chrono_literals;

    /// Spins `executor` until `done` holds or `timeout` passes; returns whether `done` held.
    inline bool spinUntil(rclcpp::Executor &executor, const std::function<bool()> &done,
                          std::chrono::milliseconds timeout = 2000ms)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!done())
        {
            if (std::chrono::steady_clock::now() > deadline)
                return false;
            executor.spin_some(10ms);
        }
        return true;
    }

    /// Spins `executor` for `duration`, e.g. to show that nothing arrives.
    inline void spinFor(rclcpp::Executor &executor, std::chrono::milliseconds duration)
    {
        const auto deadline = std::chrono::steady_clock::now() + duration;
        while (std::chrono::steady_clock::now() < deadline)
            executor.spin_some(10ms);
    }

    /// A name no other test (or node on the network, e.g. a live sensor) uses: frames, topics and node names stay apart.
    inline std::string uniqueName(const std::string &prefix)
    {
        static std::atomic<int> counter{0};
        return prefix + "_" + std::to_string(Spacetime::Testing::processId()) + "_" + std::to_string(counter++);
    }
} // namespace Spacetime::Ros2Tests

#endif // TESTS_ROS2_ROS2TESTUTILITIES_HPP
