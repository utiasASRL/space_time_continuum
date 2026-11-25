/**
 * @file ImuSensorNode.hpp
 * @brief A ROS2 node turning sensor_msgs/Imu messages into gyroscope, accelerometer or IMU
 *        measurements (see Sensors::GyroSensor, Sensors::AccelerometerSensor and Sensors::IMUSensor).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_IMUSENSORNODE_HPP
#define SPACETIME_ROS2_IMUSENSORNODE_HPP

#include <array>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include <spacetime/interface/ros2/SensorNode.hpp>

namespace Spacetime::Interface::Ros2
{
    /// Which measurement an ImuSensorNode makes of each sensor_msgs/Imu message.
    enum class ImuMeasurement
    {
        Gyro,          ///< 3x1 angular_velocity [rad/s], for a Sensors::GyroSensor.
        Accelerometer, ///< 3x1 linear_acceleration (specific force) [m/s^2], for a Sensors::AccelerometerSensor.
        IMU,           ///< 6x1 [angular_velocity; linear_acceleration], for a Sensors::IMUSensor.
    };

    /// What an ImuSensorNode listens to, and the measurements it makes of it.
    struct ImuSensorOptions
    {
        /// The sensor_msgs/Imu topic listened on.
        std::string topic = "/imu/data";
        /// If set, only messages whose header.frame_id is this (compared without a leading '/') are used.
        std::string frame_id;
        /// Where along the robot the sensor is attached [m] (the estimator sensor's arclength).
        double arclength = 0.0;
        /// Covariance of the angular velocity, sensor frame [(rad/s)^2].
        Eigen::Matrix3d gyro_covariance = Eigen::Matrix3d::Identity() * 1e-4;
        /// Covariance of the specific force, sensor frame [(m/s^2)^2].
        Eigen::Matrix3d accelerometer_covariance = Eigen::Matrix3d::Identity() * 1e-2;
        /// Use each message's covariance instead, unless it is unknown (its first entry -1, or
        /// all zeros) or not symmetric with a positive diagonal: then the options' is used.
        bool use_message_covariance = false;
        /// Which components are measured (1) or ignored (0), [angular velocity; specific force];
        /// a Gyro or Accelerometer node uses its half. The accelerometer drops masked samples.
        Eigen::Matrix<int, 6, 1> mask = Eigen::Matrix<int, 6, 1>::Ones();
    };

    /**
     * @brief Subscribes to a sensor_msgs/Imu topic and, for each message (of its frame, if
     *        set), queues a measurement (see ImuMeasurement) into `queue`, stamped with the
     *        message's stamp in estimation time (see RosClock); stamps are dropped as
     *        SensorNode describes. Readings stay in the message's (sensor) frame: the
     *        sensor's mounting offset is the estimator sensor's (e.g. Sensors::IMUOptions::offset).
     */
    class ImuSensorNode : public SensorNode
    {
    public:
        /**
         * @brief Validates `options`, then subscribes to `options.topic`.
         * @param queue The queue measurements are added to; shared with the node.
         * @param clock Converts stamps to estimation time; typically Session::clock().
         * @param measurement Which measurement to make of each message.
         * @param name The node's name; derived from the measurement and topic if empty.
         * @throws std::invalid_argument for a null queue or clock, an empty topic, or a
         *         covariance that is not symmetric with a positive diagonal.
         */
        ImuSensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, ImuMeasurement measurement,
                      ImuSensorOptions options, const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions(), const std::string &name = "");

        /// The measurement the node makes.
        ImuMeasurement measurement() const { return m_measurement; }
        /// The options the node was built with.
        const ImuSensorOptions &options() const { return m_options; }

        /// The 3x3 covariance of a message's row-major 3x3 `covariance` field, or `fallback` if it is unknown or invalid (see ImuSensorOptions::use_message_covariance).
        static Eigen::Matrix3d messageCovariance(const std::array<double, 9> &covariance, const Eigen::Matrix3d &fallback);

    private:
        ImuMeasurement m_measurement;
        ImuSensorOptions m_options;
        std::string m_frame; // without a leading '/'
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mp_subscription;

        /// Queues (or drops) the measurement of one message.
        void onImu(const sensor_msgs::msg::Imu &message);
    };
} // namespace Spacetime::Interface::Ros2

#endif // SPACETIME_ROS2_IMUSENSORNODE_HPP
