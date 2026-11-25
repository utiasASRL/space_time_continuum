#include <chrono>
#include <memory>

#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <spacetime/interface/ros2/Conversions.hpp>
#include <spacetime/interface/ros2/WaitForTransform.hpp>
#include "Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;

    TEST(WaitForTransform, ReturnsTheTransformOnceAvailable)
    {
        const std::string parent = uniqueName("field"), child = uniqueName("sensor");
        const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3), gtsam::Point3(0.4, 0.5, 0.6));
        auto node = std::make_shared<rclcpp::Node>(uniqueName("static_broadcaster"));
        tf2_ros::StaticTransformBroadcaster broadcaster(node);
        geometry_msgs::msg::TransformStamped t;
        t.header.frame_id = parent;
        t.header.stamp = node->now();
        t.child_frame_id = child;
        t.transform = Interface::Ros2::toTransformMsg(pose);
        broadcaster.sendTransform(t);

        const auto found = Interface::Ros2::waitForTransform(parent, child, 3s);

        ASSERT_TRUE(found);
        EXPECT_TRUE(found->equals(pose, 1e-9));
    }

    TEST(WaitForTransform, TimesOutWithoutTheTransform)
    {
        const auto start = std::chrono::steady_clock::now();

        const auto found = Interface::Ros2::waitForTransform(uniqueName("field"), uniqueName("sensor"), 200ms);

        EXPECT_FALSE(found);
        EXPECT_GE(std::chrono::steady_clock::now() - start, 200ms);
    }
} // namespace Spacetime
