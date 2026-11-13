#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <spacetime/interface/ros2/Conversions.hpp>

#include "WaitForTransform.hpp"

namespace Spacetime::Ros2
{
    std::optional<gtsam::Pose3> waitForTransform(const std::string &parent, const std::string &child, std::chrono::nanoseconds timeout,
                                                 const rclcpp::NodeOptions &node_options)
    {
        auto node = std::make_shared<rclcpp::Node>("spacetime_wait_for_transform", node_options);
        tf2_ros::Buffer buffer(node->get_clock());
        tf2_ros::TransformListener listener(buffer, node, false);
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (rclcpp::ok())
        {
            std::string not_yet; // passed so tf2 does not warn while the frames are unknown
            if (buffer.canTransform(parent, child, tf2::TimePointZero, &not_yet))
                return toPose3(buffer.lookupTransform(parent, child, tf2::TimePointZero).transform);
            if (std::chrono::steady_clock::now() >= deadline)
                break;
            executor.spin_some(std::chrono::milliseconds(10));
        }
        return std::nullopt;
    }
} // namespace Spacetime::Ros2
