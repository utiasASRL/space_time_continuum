#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#include <spacetime/estimator/types/Measurement.hpp>

#include "KinematicTendonNode.hpp"

namespace Spacetime::Interface::Ros2
{
    namespace
    {
        /// Messages kept while the executor is busy: best effort (SensorDataQoS), deep enough for a high-rate sensor.
        constexpr std::size_t kQueueDepth = 100;

        const KinematicTendonOptions &validated(const KinematicTendonOptions &options)
        {
            if (options.topic.empty() || options.joint_name.empty())
                throw std::invalid_argument("Ros2::KinematicTendonNode: the topic and joint name must be set.");
            if (!(options.variance > 0.0) || !std::isfinite(options.variance))
                throw std::invalid_argument("Ros2::KinematicTendonNode: the variance must be positive.");
            if (!std::isfinite(options.scale) || !std::isfinite(options.length_offset))
                throw std::invalid_argument("Ros2::KinematicTendonNode: the scale and length offset must be finite.");
            return options;
        }
    } // namespace

    KinematicTendonNode::KinematicTendonNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, KinematicTendonOptions options,
                                             const rclcpp::NodeOptions &node_options, const std::string &name)
        : SensorNode(std::move(queue), std::move(clock), name.empty() ? nodeName("tendon_", validated(options).joint_name) : name,
                     node_options, "KinematicTendonNode"),
          m_options(std::move(options))
    {
        validated(m_options);
        mp_subscription = create_subscription<sensor_msgs::msg::JointState>(
            m_options.topic, rclcpp::SensorDataQoS().keep_last(kQueueDepth), [this](const sensor_msgs::msg::JointState &message)
            { onJointState(message); });
    }

    void KinematicTendonNode::onJointState(const sensor_msgs::msg::JointState &message)
    {
        countReceived();
        const auto it = std::find(message.name.begin(), message.name.end(), m_options.joint_name);
        if (it == message.name.end())
            return;
        const auto index = static_cast<std::size_t>(std::distance(message.name.begin(), it));
        if (index >= message.position.size() || !std::isfinite(message.position[index]))
        {
            countDropped();
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Dropping a joint state of '%s': it has no (finite) position for it.",
                                 m_options.joint_name.c_str());
            return;
        }
        const auto time = acceptStamp(message.header.stamp, "the tendon length of '" + m_options.joint_name + "'");
        if (!time)
            return;

        const double length = m_options.scale * message.position[index] + m_options.length_offset;
        queueMeasurement(std::make_shared<Types::MeasurementBase>(
            0.0, *time, Eigen::MatrixXd::Constant(1, 1, length), Eigen::MatrixXd::Constant(1, 1, m_options.variance)));
    }
} // namespace Spacetime::Interface::Ros2
