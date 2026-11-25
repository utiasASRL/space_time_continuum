#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include <spacetime/estimator/types/Measurement.hpp>

#include "ImuSensorNode.hpp"

namespace Spacetime::Interface::Ros2
{
    namespace
    {
        /// Messages kept while the executor is busy: best effort (SensorDataQoS), deep enough for a high-rate sensor.
        constexpr std::size_t kQueueDepth = 100;

        const ImuSensorOptions &validated(const ImuSensorOptions &options)
        {
            if (options.topic.empty())
                throw std::invalid_argument("Ros2::ImuSensorNode: the topic must be set.");
            SensorNode::validateCovariance(options.gyro_covariance, "Ros2::ImuSensorNode (gyro)");
            SensorNode::validateCovariance(options.accelerometer_covariance, "Ros2::ImuSensorNode (accelerometer)");
            return options;
        }

        std::string prefix(ImuMeasurement measurement)
        {
            switch (measurement)
            {
            case ImuMeasurement::Gyro:
                return "gyro_";
            case ImuMeasurement::Accelerometer:
                return "accelerometer_";
            case ImuMeasurement::IMU:
                return "imu_";
            }
            return "imu_";
        }

        Eigen::Vector3d toVector(const geometry_msgs::msg::Vector3 &v) { return Eigen::Vector3d(v.x, v.y, v.z); }
    } // namespace

    ImuSensorNode::ImuSensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, ImuMeasurement measurement,
                                 ImuSensorOptions options, const rclcpp::NodeOptions &node_options, const std::string &name)
        : SensorNode(std::move(queue), std::move(clock), name.empty() ? nodeName(prefix(measurement), validated(options).topic) : name,
                     node_options, "ImuSensorNode"),
          m_measurement(measurement), m_options(std::move(options)), m_frame(stripSlash(m_options.frame_id))
    {
        validated(m_options);
        mp_subscription = create_subscription<sensor_msgs::msg::Imu>(
            m_options.topic, rclcpp::SensorDataQoS().keep_last(kQueueDepth), [this](const sensor_msgs::msg::Imu &message)
            { onImu(message); });
    }

    Eigen::Matrix3d ImuSensorNode::messageCovariance(const std::array<double, 9> &covariance, const Eigen::Matrix3d &fallback)
    {
        const Eigen::Matrix3d C = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(covariance.data());
        if (C(0, 0) == -1.0 || C.isZero(0.0) || !C.isApprox(C.transpose()) || !(C.diagonal().array() > 0.0).all())
            return fallback;
        return C;
    }

    void ImuSensorNode::onImu(const sensor_msgs::msg::Imu &message)
    {
        countReceived();
        if (!m_frame.empty() && stripSlash(message.header.frame_id) != m_frame)
            return;
        const auto time = acceptStamp(message.header.stamp, "the IMU message on " + m_options.topic);
        if (!time)
            return;

        const bool from_message = m_options.use_message_covariance;
        const Eigen::Matrix3d gyro_covariance =
            from_message ? messageCovariance(message.angular_velocity_covariance, m_options.gyro_covariance) : m_options.gyro_covariance;
        const Eigen::Matrix3d accel_covariance =
            from_message ? messageCovariance(message.linear_acceleration_covariance, m_options.accelerometer_covariance) : m_options.accelerometer_covariance;
        const Eigen::Vector3d omega = toVector(message.angular_velocity), accel = toVector(message.linear_acceleration);

        Eigen::MatrixXd value, covariance;
        Eigen::MatrixXi mask;
        switch (m_measurement)
        {
        case ImuMeasurement::Gyro:
            value = omega;
            covariance = gyro_covariance;
            mask = m_options.mask.head<3>();
            break;
        case ImuMeasurement::Accelerometer:
            value = accel;
            covariance = accel_covariance;
            mask = m_options.mask.tail<3>();
            break;
        case ImuMeasurement::IMU:
            value.resize(6, 1);
            value << omega, accel;
            covariance = Eigen::MatrixXd::Zero(6, 6);
            covariance.topLeftCorner<3, 3>() = gyro_covariance;
            covariance.bottomRightCorner<3, 3>() = accel_covariance;
            mask = m_options.mask;
            break;
        }
        queueMeasurement(std::make_shared<Types::MeasurementBase>(m_options.arclength, *time, value, covariance, mask));
    }
} // namespace Spacetime::Interface::Ros2
