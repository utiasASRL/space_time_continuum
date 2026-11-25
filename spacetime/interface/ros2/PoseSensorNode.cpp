#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/interface/ros2/Conversions.hpp>

#include "PoseSensorNode.hpp"

namespace Spacetime::Interface::Ros2
{
    namespace
    {
        const PoseSensorOptions &validated(const PoseSensorOptions &options)
        {
            const std::string parent = SensorNode::stripSlash(options.parent_frame), child = SensorNode::stripSlash(options.child_frame),
                              body = SensorNode::stripSlash(options.body_frame);
            if (parent.empty() || child.empty())
                throw std::invalid_argument("Ros2::PoseSensorNode: the parent and child frames must be set.");
            if (!body.empty() && (body == parent || body == child))
                throw std::invalid_argument("Ros2::PoseSensorNode: the body frame must differ from the measured edge's frames.");
            SensorNode::validateCovariance(options.covariance, "Ros2::PoseSensorNode");
            return options;
        }
    } // namespace

    PoseSensorNode::PoseSensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, PoseSensorOptions options,
                                   const rclcpp::NodeOptions &node_options, const std::string &name)
        : SensorNode(std::move(queue), std::move(clock), name.empty() ? nodeName("pose_sensor_", validated(options).child_frame) : name,
                     node_options, "PoseSensorNode"),
          m_options(std::move(options)), m_parent(stripSlash(m_options.parent_frame)), m_child(stripSlash(m_options.child_frame))
    {
        validated(m_options);

        if (!m_options.body_frame.empty())
            mp_body_publisher = create_publisher<tf2_msgs::msg::TFMessage>(m_options.topic, tf2_ros::DynamicBroadcasterQoS());
        mp_subscription = create_subscription<tf2_msgs::msg::TFMessage>(
            m_options.topic, tf2_ros::DynamicListenerQoS(), [this](const tf2_msgs::msg::TFMessage &message)
            { onTransforms(message); });
    }

    void PoseSensorNode::onTransforms(const tf2_msgs::msg::TFMessage &message)
    {
        for (const auto &transform : message.transforms)
            if (stripSlash(transform.header.frame_id) == m_parent && stripSlash(transform.child_frame_id) == m_child)
                onTransform(transform);
        countReceived();
    }

    void PoseSensorNode::onTransform(const geometry_msgs::msg::TransformStamped &transform)
    {
        const auto time = acceptStamp(transform.header.stamp, m_parent + " -> " + m_child);
        if (!time)
            return;

        const gtsam::Pose3 T_ib = bodyPose(toPose3(transform.transform), m_options.offset);
        queueMeasurement(std::make_shared<Types::MeasurementBase>(
            m_options.arclength, *time, T_ib.matrix(), Eigen::MatrixXd(m_options.covariance), Eigen::MatrixXi(m_options.mask)));

        if (mp_body_publisher)
        {
            tf2_msgs::msg::TFMessage body;
            body.transforms.emplace_back();
            body.transforms.back().header = transform.header;
            body.transforms.back().child_frame_id = m_options.body_frame;
            body.transforms.back().transform = toTransformMsg(T_ib);
            mp_body_publisher->publish(body);
        }
    }
} // namespace Spacetime::Interface::Ros2
