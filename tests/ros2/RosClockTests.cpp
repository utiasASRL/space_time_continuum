#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>

#include <spacetime/interface/ros2/RosClock.hpp>
#include "Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;

    namespace
    {
        builtin_interfaces::msg::Time stamp(int32_t sec, uint32_t nanosec)
        {
            builtin_interfaces::msg::Time time;
            time.sec = sec;
            time.nanosec = nanosec;
            return time;
        }
    } // namespace

    TEST(RosClock, RejectsNullClock)
    {
        EXPECT_THROW(Ros2::RosClock(nullptr), std::invalid_argument);
    }

    TEST(RosClock, IsRealTime)
    {
        Ros2::RosClock clock(std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME));
        EXPECT_TRUE(clock.isRealTime());
    }

    TEST(RosClock, ReadsTheStartTimeUntilStarted)
    {
        Ros2::RosClock clock(std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME), 5.0);
        EXPECT_FALSE(clock.started());
        EXPECT_DOUBLE_EQ(clock.now(), 5.0);
        EXPECT_THROW(clock.toEstimatorTime(stamp(1, 0)), std::logic_error);
        EXPECT_THROW(clock.toRosTime(5.0), std::logic_error);
    }

    TEST(RosClock, CountsFromTheStartTimeOnceStarted)
    {
        Ros2::RosClock clock(std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME), 5.0);
        ASSERT_TRUE(clock.tryStart());
        EXPECT_TRUE(clock.started());
        const double first = clock.now();
        EXPECT_GE(first, 5.0);
        EXPECT_LT(first, 5.5);
        rclcpp::sleep_for(20ms);
        EXPECT_GT(clock.now(), first);
    }

    TEST(RosClock, TryStartKeepsTheFirstOrigin)
    {
        Ros2::RosClock clock(std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME));
        ASSERT_TRUE(clock.tryStart());
        const rclcpp::Time origin = clock.origin();
        rclcpp::sleep_for(10ms);
        EXPECT_TRUE(clock.tryStart());
        EXPECT_EQ(clock.origin(), origin);
    }

    TEST(RosClock, StampsRoundTripThroughEstimatorTime)
    {
        Ros2::RosClock clock(std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME), 2.0);
        ASSERT_TRUE(clock.tryStart());
        const rclcpp::Time origin = clock.origin();

        const rclcpp::Time later = origin + rclcpp::Duration(1, 250000000);
        const double time = clock.toEstimatorTime(later);
        EXPECT_NEAR(time, 3.25, 1e-9);
        EXPECT_EQ(rclcpp::Time(clock.toRosTime(time)).nanoseconds(), later.nanoseconds());

        // Stamps before the origin are before the start time.
        const rclcpp::Time earlier = origin - rclcpp::Duration(0, 500000000);
        EXPECT_NEAR(clock.toEstimatorTime(earlier), 1.5, 1e-9);
    }

    // Under use_sim_time (e.g. rosbag replay with --clock) the node's clock reads zero until
    // the first /clock message: the origin must wait for it.
    TEST(RosClock, SimTimeStartsOnTheFirstClockMessage)
    {
        auto node = std::make_shared<rclcpp::Node>(uniqueName("rosclock_sim"),
                                                   rclcpp::NodeOptions().parameter_overrides({{"use_sim_time", true}}));
        auto clock_node = std::make_shared<rclcpp::Node>(uniqueName("rosclock_publisher"));
        auto clock_publisher = clock_node->create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        executor.add_node(clock_node);

        Ros2::RosClock clock(node->get_clock());
        EXPECT_FALSE(clock.tryStart());

        rosgraph_msgs::msg::Clock message;
        message.clock = stamp(100, 0);
        ASSERT_TRUE(spinUntil(executor, [&]()
                              { clock_publisher->publish(message); return node->get_clock()->now().seconds() >= 100.0; }));
        ASSERT_TRUE(clock.tryStart());
        EXPECT_NEAR(clock.now(), 0.0, 1e-9);

        message.clock = stamp(101, 500000000);
        ASSERT_TRUE(spinUntil(executor, [&]()
                              { clock_publisher->publish(message); return node->get_clock()->now().seconds() >= 101.5; }));
        EXPECT_NEAR(clock.now(), 1.5, 1e-9);
        EXPECT_NEAR(clock.toEstimatorTime(stamp(101, 0)), 1.0, 1e-9);
    }
} // namespace Spacetime
