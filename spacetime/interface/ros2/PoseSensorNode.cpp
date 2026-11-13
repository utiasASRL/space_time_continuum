#include <cctype>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/interface/ros2/Conversions.hpp>

#include "PoseSensorNode.hpp"

namespace Spacetime::Ros2
{
    namespace
    {
        /// TF frame ids are compared without a leading '/' (as tf2 does).
        std::string stripSlash(const std::string &frame)
        {
            return (!frame.empty() && frame.front() == '/') ? frame.substr(1) : frame;
        }

        /// A valid node name for the sensor of `frame`.
        std::string nodeName(const std::string &frame)
        {
            std::string name = "pose_sensor_";
            for (const char c : stripSlash(frame))
                name += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
            return name;
        }

        const PoseSensorOptions &validated(const PoseSensorOptions &options)
        {
            const std::string parent = stripSlash(options.parent_frame), child = stripSlash(options.child_frame),
                              body = stripSlash(options.body_frame);
            if (parent.empty() || child.empty())
                throw std::invalid_argument("Ros2::PoseSensorNode: the parent and child frames must be set.");
            if (!body.empty() && (body == parent || body == child))
                throw std::invalid_argument("Ros2::PoseSensorNode: the body frame must differ from the measured edge's frames.");
            if (!options.covariance.isApprox(options.covariance.transpose()) || !(options.covariance.diagonal().array() > 0.0).all())
                throw std::invalid_argument("Ros2::PoseSensorNode: the covariance must be symmetric with a positive diagonal.");
            return options;
        }
    } // namespace

    PoseSensorNode::PoseSensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, PoseSensorOptions options,
                                   const rclcpp::NodeOptions &node_options, const std::string &name)
        : rclcpp::Node(name.empty() ? nodeName(validated(options).child_frame) : name, node_options),
          mp_queue(std::move(queue)), mp_clock(std::move(clock)), m_options(std::move(options)),
          m_parent(stripSlash(m_options.parent_frame)), m_child(stripSlash(m_options.child_frame))
    {
        validated(m_options);
        if (!mp_queue || !mp_clock)
            throw std::invalid_argument("Ros2::PoseSensorNode requires a non-null queue and clock.");

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
        m_received++;
    }

    void PoseSensorNode::onTransform(const geometry_msgs::msg::TransformStamped &transform)
    {
        if (!mp_clock->started())
        {
            m_dropped++;
            return;
        }
        const double time = mp_clock->toEstimatorTime(transform.header.stamp);
        if (time < mp_clock->startTime() || (m_last_time && time <= *m_last_time))
        {
            m_dropped++;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Dropping %s -> %s at t = %.4f s: before the start time or not newer than the last measurement.",
                                 m_parent.c_str(), m_child.c_str(), time);
            return;
        }
        m_last_time = time;

        const gtsam::Pose3 T_ib = bodyPose(toPose3(transform.transform), m_options.offset);
        mp_queue->addMeasurement(std::make_shared<Types::MeasurementBase>(
            m_options.arclength, time, T_ib.matrix(), Eigen::MatrixXd(m_options.covariance), Eigen::MatrixXi(m_options.mask)));
        m_added++;

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
} // namespace Spacetime::Ros2
