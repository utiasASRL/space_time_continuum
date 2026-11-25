/**
 * @file KinematicTendonNode.hpp
 * @brief A ROS2 node turning one joint of sensor_msgs/JointState messages into tendon-length
 *        measurements (see Actuators::KinematicTendon).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_KINEMATICTENDONNODE_HPP
#define SPACETIME_ROS2_KINEMATICTENDONNODE_HPP

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include <spacetime/interface/ros2/SensorNode.hpp>

namespace Spacetime::Interface::Ros2
{
    /// What a KinematicTendonNode listens to, and the measurements it makes of it.
    struct KinematicTendonOptions
    {
        /// The sensor_msgs/JointState topic listened on.
        std::string topic = "/joint_states";
        /// The name of the joint (in JointState::name) whose position measures the tendon.
        std::string joint_name;
        /// Variance of the measured tendon length [m^2] (the 1x1 measurement covariance).
        double variance = 1e-6;
        /// The tendon length is scale * position + length_offset [m], e.g. a spool's angle times its radius.
        double scale = 1.0;
        double length_offset = 0.0; ///< See scale [m].
    };

    /**
     * @brief Subscribes to a sensor_msgs/JointState topic and, for each message naming its
     *        joint, queues the tendon length (a 1x1 measurement, see KinematicTendonOptions)
     *        into `queue`, typically an Actuators::KinematicTendon's, stamped with the
     *        message's stamp in estimation time (see RosClock); stamps are dropped as
     *        SensorNode describes. A message naming the joint without a position for it is
     *        dropped; one not naming it is ignored.
     */
    class KinematicTendonNode : public SensorNode
    {
    public:
        /**
         * @brief Validates `options`, then subscribes to `options.topic`.
         * @param queue The queue measurements are added to; shared with the node.
         * @param clock Converts stamps to estimation time; typically Session::clock().
         * @param name The node's name; derived from the joint name if empty.
         * @throws std::invalid_argument for a null queue or clock, an empty topic or joint
         *         name, a non-positive variance, or a non-finite scale or offset.
         */
        KinematicTendonNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, KinematicTendonOptions options,
                            const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions(), const std::string &name = "");

        /// The options the node was built with.
        const KinematicTendonOptions &options() const { return m_options; }

    private:
        KinematicTendonOptions m_options;
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr mp_subscription;

        /// Queues (or drops) the tendon length of one message.
        void onJointState(const sensor_msgs::msg::JointState &message);
    };
} // namespace Spacetime::Interface::Ros2

#endif // SPACETIME_ROS2_KINEMATICTENDONNODE_HPP
