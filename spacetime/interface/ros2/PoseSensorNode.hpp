/**
 * @file PoseSensorNode.hpp
 * @brief A ROS2 node turning one TF edge (e.g. an electromagnetic tracker's
 *        field -> sensor) into pose measurements of a robot.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_POSESENSORNODE_HPP
#define SPACETIME_ROS2_POSESENSORNODE_HPP

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/qos.hpp>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>

namespace Spacetime::Ros2
{
    /// What a PoseSensorNode listens to, and the measurements it makes of it.
    struct PoseSensorOptions
    {
        /// The TF edge measured: parent_frame -> child_frame (the sensor), e.g. "aurora_field" -> "aurora_sensor_0".
        std::string parent_frame;
        std::string child_frame;
        /// T_bs: the sensor's pose in the robot's frame at `arclength` (it is rigidly attached
        /// there). The measured robot pose is T_ib = T_is * T_bs^-1.
        gtsam::Pose3 offset;
        /// Where along the robot the sensor is attached [m].
        double arclength = 0.0;
        /// Measurement covariance, in the robot frame's tangent space ([rot, trans], see Sensors::PoseSensor).
        Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
        /// Which components of the error are measured (1) or ignored (0), [rot, trans]: e.g.
        /// [1, 1, 0, 1, 1, 1] ignores the roll about the robot's tangent (z) of a 5DoF sensor.
        Eigen::Matrix<int, 6, 1> mask = Eigen::Matrix<int, 6, 1>::Ones();
        /// If set, T_ib is also broadcast on `topic` as parent_frame -> body_frame, e.g. to see it in rviz2.
        std::string body_frame;
        /// The TFMessage topic listened (and broadcast) on.
        std::string topic = "/tf";
    };

    /**
     * @brief Subscribes to TF and, for each transform of its edge, queues a pose measurement
     *        of the robot (see PoseSensorOptions) into `queue`, typically a
     *        Sensors::PoseSensor's, stamped with the transform's stamp in estimation time
     *        (see RosClock). Stamps before the clock starts, before its start time, or not
     *        newer than the last one queued are dropped: MeasurementQueue drains in insertion
     *        order. Only the direct edge is matched (no TF chain lookup); frame ids are
     *        compared without a leading '/'.
     */
    class PoseSensorNode : public rclcpp::Node
    {
    public:
        /**
         * @param name The node's name; derived from the child frame if empty.
         * @throws std::invalid_argument for a null queue or clock, an empty frame, a body
         *         frame equal to the edge's frames, or a covariance that is not symmetric
         *         with a positive diagonal.
         */
        PoseSensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, PoseSensorOptions options,
                       const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions(), const std::string &name = "");

        /// T_ib = T_is * T_bs^-1: the robot's pose from the sensor's (T_is) and its offset (T_bs).
        static gtsam::Pose3 bodyPose(const gtsam::Pose3 &T_is, const gtsam::Pose3 &T_bs) { return T_is * T_bs.inverse(); }

        const PoseSensorOptions &options() const { return m_options; }
        const Types::pMeasurementQueueType &queue() const { return mp_queue; }

        /// TF messages received, matching or not.
        std::size_t messagesReceived() const { return m_received.load(); }
        /// Measurements queued.
        std::size_t measurementsAdded() const { return m_added.load(); }
        /// Transforms of the edge dropped (see the class comment).
        std::size_t measurementsDropped() const { return m_dropped.load(); }

    private:
        Types::pMeasurementQueueType mp_queue;
        std::shared_ptr<const RosClock> mp_clock;
        PoseSensorOptions m_options;
        std::string m_parent, m_child; // without a leading '/'
        rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr mp_subscription;
        rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr mp_body_publisher;
        std::optional<double> m_last_time;
        std::atomic<std::size_t> m_received{0}, m_added{0}, m_dropped{0};

        void onTransforms(const tf2_msgs::msg::TFMessage &message);
        void onTransform(const geometry_msgs::msg::TransformStamped &transform);
    };
} // namespace Spacetime::Ros2

#endif // SPACETIME_ROS2_POSESENSORNODE_HPP
