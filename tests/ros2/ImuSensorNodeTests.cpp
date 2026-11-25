#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/ImuSensorNode.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include "Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;
    using Interface::Ros2::ImuMeasurement;

    namespace
    {
        const Eigen::Vector3d kOmega(0.1, -0.2, 0.3), kAccel(0.5, -0.4, 9.7);

        /// An IMU node on its own topic, a publisher feeding it, and the queue it fills.
        struct ImuRig
        {
            std::shared_ptr<Types::MeasurementQueue> queue = std::make_shared<Types::MeasurementQueue>();
            std::shared_ptr<Interface::Ros2::RosClock> clock = std::make_shared<Interface::Ros2::RosClock>(std::make_shared<rclcpp::Clock>(RCL_ROS_TIME), 0.0);
            Interface::Ros2::ImuSensorOptions options;
            std::shared_ptr<Interface::Ros2::ImuSensorNode> node;
            std::shared_ptr<rclcpp::Node> publisher_node;
            rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher;
            rclcpp::executors::SingleThreadedExecutor executor;

            ImuRig()
            {
                options.topic = "/" + uniqueName("imu");
                options.arclength = 0.4;
                options.gyro_covariance = Eigen::Vector3d(1e-4, 2e-4, 3e-4).asDiagonal();
                options.accelerometer_covariance = Eigen::Vector3d(1e-2, 2e-2, 3e-2).asDiagonal();
                clock->tryStart();
            }

            void create(ImuMeasurement measurement)
            {
                node = std::make_shared<Interface::Ros2::ImuSensorNode>(queue, clock, measurement, options);
                publisher_node = std::make_shared<rclcpp::Node>(uniqueName("imu_publisher"));
                publisher = publisher_node->create_publisher<sensor_msgs::msg::Imu>(options.topic, rclcpp::SensorDataQoS());
                executor.add_node(node);
                executor.add_node(publisher_node);
                ASSERT_TRUE(spinUntil(executor, [&]()
                                      { return publisher->get_subscription_count() > 0; }));
            }

            sensor_msgs::msg::Imu message(double seconds, const std::string &frame = "imu_link") const
            {
                sensor_msgs::msg::Imu m;
                m.header.stamp = clock->toRosTime(seconds);
                m.header.frame_id = frame;
                m.angular_velocity.x = kOmega.x();
                m.angular_velocity.y = kOmega.y();
                m.angular_velocity.z = kOmega.z();
                m.linear_acceleration.x = kAccel.x();
                m.linear_acceleration.y = kAccel.y();
                m.linear_acceleration.z = kAccel.z();
                return m;
            }

            /// Publishes, then spins until the node has seen it.
            bool publishAndWait(const sensor_msgs::msg::Imu &message)
            {
                const std::size_t seen = node->messagesReceived();
                publisher->publish(message);
                return spinUntil(executor, [&]()
                                 { return node->messagesReceived() > seen; });
            }
        };
    } // namespace

    TEST(ImuSensorNode, RejectsInvalidOptions)
    {
        ImuRig rig;
        auto make = [&](const Interface::Ros2::ImuSensorOptions &options)
        { return std::make_shared<Interface::Ros2::ImuSensorNode>(rig.queue, rig.clock, ImuMeasurement::IMU, options); };

        EXPECT_THROW(std::make_shared<Interface::Ros2::ImuSensorNode>(nullptr, rig.clock, ImuMeasurement::IMU, rig.options), std::invalid_argument);
        EXPECT_THROW(std::make_shared<Interface::Ros2::ImuSensorNode>(rig.queue, nullptr, ImuMeasurement::IMU, rig.options), std::invalid_argument);

        auto options = rig.options;
        options.topic.clear();
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.gyro_covariance(1, 1) = 0.0;
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.accelerometer_covariance(0, 2) = 1e-3; // not symmetric
        EXPECT_THROW(make(options), std::invalid_argument);
    }

    TEST(ImuSensorNode, EachModeQueuesItsMeasurement)
    {
        for (const ImuMeasurement measurement : {ImuMeasurement::Gyro, ImuMeasurement::Accelerometer, ImuMeasurement::IMU})
        {
            SCOPED_TRACE(static_cast<int>(measurement));
            ImuRig rig;
            rig.options.mask << 1, 0, 1, 1, 1, 0;
            rig.create(measurement);
            ASSERT_TRUE(rig.publishAndWait(rig.message(1.25)));

            const auto measurements = rig.queue->peekAll();
            ASSERT_EQ(measurements.size(), 1u);
            const auto &m = measurements.front();
            EXPECT_NEAR(m->time, 1.25, 1e-9);
            EXPECT_DOUBLE_EQ(m->arclength, 0.4);
            switch (measurement)
            {
            case ImuMeasurement::Gyro:
                EXPECT_TRUE(m->value.isApprox(Eigen::MatrixXd(kOmega)));
                EXPECT_TRUE(m->covariance.isApprox(Eigen::MatrixXd(rig.options.gyro_covariance)));
                EXPECT_EQ(m->mask, Eigen::MatrixXi(rig.options.mask.head<3>()));
                break;
            case ImuMeasurement::Accelerometer:
                EXPECT_TRUE(m->value.isApprox(Eigen::MatrixXd(kAccel)));
                EXPECT_TRUE(m->covariance.isApprox(Eigen::MatrixXd(rig.options.accelerometer_covariance)));
                EXPECT_EQ(m->mask, Eigen::MatrixXi(rig.options.mask.tail<3>()));
                break;
            case ImuMeasurement::IMU:
            {
                ASSERT_EQ(m->value.size(), 6);
                EXPECT_TRUE(m->value.topRows<3>().isApprox(Eigen::MatrixXd(kOmega)));
                EXPECT_TRUE(m->value.bottomRows<3>().isApprox(Eigen::MatrixXd(kAccel)));
                EXPECT_TRUE((m->covariance.topLeftCorner<3, 3>().isApprox(rig.options.gyro_covariance)));
                EXPECT_TRUE((m->covariance.bottomRightCorner<3, 3>().isApprox(rig.options.accelerometer_covariance)));
                EXPECT_TRUE((m->covariance.topRightCorner<3, 3>().isZero()));
                EXPECT_EQ(m->mask, Eigen::MatrixXi(rig.options.mask));
                break;
            }
            }
            EXPECT_EQ(rig.node->measurementsAdded(), 1u);
        }
    }

    TEST(ImuSensorNode, OnlyItsFrameIsUsedWhenSet)
    {
        ImuRig rig;
        rig.options.frame_id = "imu_link";
        rig.create(ImuMeasurement::Gyro);
        ASSERT_TRUE(rig.publishAndWait(rig.message(1.0, "other_link")));
        EXPECT_TRUE(rig.queue->empty());
        ASSERT_TRUE(rig.publishAndWait(rig.message(1.1, "/imu_link")));
        EXPECT_EQ(rig.queue->size(), 1u);
    }

    TEST(ImuSensorNode, MessageCovarianceIsUsedOnlyWhenKnownAndValid)
    {
        const Eigen::Matrix3d fallback = Eigen::Matrix3d::Identity() * 7.0;
        std::array<double, 9> covariance{};
        EXPECT_TRUE(Interface::Ros2::ImuSensorNode::messageCovariance(covariance, fallback).isApprox(fallback)); // all zeros: unknown
        covariance[0] = -1.0;
        EXPECT_TRUE(Interface::Ros2::ImuSensorNode::messageCovariance(covariance, fallback).isApprox(fallback)); // -1: not provided
        covariance = {1e-3, 1e-5, 0, 2e-5, 2e-3, 0, 0, 0, 3e-3};
        EXPECT_TRUE(Interface::Ros2::ImuSensorNode::messageCovariance(covariance, fallback).isApprox(fallback)); // not symmetric
        covariance = {1e-3, 1e-5, 0, 1e-5, 2e-3, 0, 0, 0, 3e-3};
        const Eigen::Matrix3d C = Interface::Ros2::ImuSensorNode::messageCovariance(covariance, fallback);
        EXPECT_DOUBLE_EQ(C(0, 1), 1e-5);
        EXPECT_DOUBLE_EQ(C(2, 2), 3e-3);

        ImuRig rig;
        rig.options.use_message_covariance = true;
        rig.create(ImuMeasurement::IMU);
        auto message = rig.message(1.0);
        message.angular_velocity_covariance = covariance;
        ASSERT_TRUE(rig.publishAndWait(message)); // the accelerometer's is unknown: the options'
        const auto m = rig.queue->peekAll().at(0);
        EXPECT_TRUE((m->covariance.topLeftCorner<3, 3>().isApprox(C)));
        EXPECT_TRUE((m->covariance.bottomRightCorner<3, 3>().isApprox(rig.options.accelerometer_covariance)));
    }

    TEST(ImuSensorNode, StampsBeforeTheStartOrNotNewerAreDropped)
    {
        ImuRig rig;
        rig.create(ImuMeasurement::IMU);
        ASSERT_TRUE(rig.publishAndWait(rig.message(-0.5)));
        ASSERT_TRUE(rig.publishAndWait(rig.message(2.0)));
        ASSERT_TRUE(rig.publishAndWait(rig.message(1.0)));
        ASSERT_TRUE(rig.publishAndWait(rig.message(2.0)));
        ASSERT_TRUE(rig.publishAndWait(rig.message(3.0)));
        const auto measurements = rig.queue->peekAll();
        ASSERT_EQ(measurements.size(), 2u);
        EXPECT_NEAR(measurements[0]->time, 2.0, 1e-9);
        EXPECT_NEAR(measurements[1]->time, 3.0, 1e-9);
        EXPECT_EQ(rig.node->measurementsDropped(), 3u);
    }
} // namespace Spacetime
