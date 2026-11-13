/**
 * @file Conversions.hpp
 * @brief Conversions between ROS2 geometry messages and gtsam poses.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_CONVERSIONS_HPP
#define SPACETIME_ROS2_CONVERSIONS_HPP

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <gtsam/geometry/Pose3.h>

namespace Spacetime::Ros2
{
    /// The pose a transform message holds, its quaternion normalized. Throws std::invalid_argument for a zero quaternion.
    gtsam::Pose3 toPose3(const geometry_msgs::msg::Transform &transform);

    /// The pose a pose message holds, its quaternion normalized. Throws std::invalid_argument for a zero quaternion.
    gtsam::Pose3 toPose3(const geometry_msgs::msg::Pose &pose);

    geometry_msgs::msg::Transform toTransformMsg(const gtsam::Pose3 &pose);

    geometry_msgs::msg::Pose toPoseMsg(const gtsam::Pose3 &pose);
} // namespace Spacetime::Ros2

#endif // SPACETIME_ROS2_CONVERSIONS_HPP
