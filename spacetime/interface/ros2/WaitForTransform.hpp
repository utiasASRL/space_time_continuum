/**
 * @file WaitForTransform.hpp
 * @brief Blocking lookup of a TF transform, e.g. to initialize a prior from a first sensor reading.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_WAITFORTRANSFORM_HPP
#define SPACETIME_ROS2_WAITFORTRANSFORM_HPP

#include <chrono>
#include <optional>
#include <string>

#include <gtsam/geometry/Pose3.h>
#include <rclcpp/node_options.hpp>

namespace Spacetime::Ros2
{
    /**
     * @brief The latest pose of `child` in `parent` from TF (/tf and /tf_static), waiting up
     *        to `timeout` for it on a temporary node of its own (so call it before the
     *        session starts). nullopt on timeout or rclcpp shutdown.
     */
    std::optional<gtsam::Pose3> waitForTransform(const std::string &parent, const std::string &child, std::chrono::nanoseconds timeout,
                                                 const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions());
} // namespace Spacetime::Ros2

#endif // SPACETIME_ROS2_WAITFORTRANSFORM_HPP
