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

namespace Spacetime::Interface::Ros2
{
    /**
     * @brief The latest pose of `child` in `parent` from TF (/tf and /tf_static), waiting up
     *        to `timeout` for it on a temporary node of its own (so call it before the
     *        session starts). Unlike PoseSensorNode, chains of transforms are resolved.
     *        Blocks the calling thread; `timeout` is measured on the wall (steady) clock, even
     *        with use_sim_time. nullopt on timeout or rclcpp shutdown.
     * @throws std::invalid_argument if the found transform's quaternion is zero.
     */
    std::optional<gtsam::Pose3> waitForTransform(const std::string &parent, const std::string &child, std::chrono::nanoseconds timeout,
                                                 const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions());
} // namespace Spacetime::Interface::Ros2

#endif // SPACETIME_ROS2_WAITFORTRANSFORM_HPP
