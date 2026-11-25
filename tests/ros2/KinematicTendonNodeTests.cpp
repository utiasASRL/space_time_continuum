#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/KinematicTendonNode.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include "Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;

    namespace
    {
        /// A tendon node on its own topic, a publisher feeding it, and the queue it fills.
        struct TendonRig
        {
            std::shared_ptr<Types::MeasurementQueue> queue = std::make_shared<Types::MeasurementQueue>();
            std::shared_ptr<Interface::Ros2::RosClock> clock = std::make_shared<Interface::Ros2::RosClock>(std::make_shared<rclcpp::Clock>(RCL_ROS_TIME), 0.0);
            Interface::Ros2::KinematicTendonOptions options;
            std::shared_ptr<Interface::Ros2::KinematicTendonNode> node;
            std::shared_ptr<rclcpp::Node> publisher_node;
            rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher;
            rclcpp::executors::SingleThreadedExecutor executor;

            TendonRig()
            {
                options.topic = "/" + uniqueName("joint_states");
                options.joint_name = "tendon_1";
                options.variance = 4e-6;
                options.scale = 0.01;        // e.g. a 1 cm spool radius
                options.length_offset = 0.2; // [m]
                clock->tryStart();
            }

            void create()
            {
                node = std::make_shared<Interface::Ros2::KinematicTendonNode>(queue, clock, options);
                publisher_node = std::make_shared<rclcpp::Node>(uniqueName("joint_state_publisher"));
                publisher = publisher_node->create_publisher<sensor_msgs::msg::JointState>(options.topic, rclcpp::SensorDataQoS());
                executor.add_node(node);
                executor.add_node(publisher_node);
                ASSERT_TRUE(spinUntil(executor, [&]()
                                      { return publisher->get_subscription_count() > 0; }));
            }

            sensor_msgs::msg::JointState message(double seconds, const std::vector<std::string> &names, const std::vector<double> &positions) const
            {
                sensor_msgs::msg::JointState m;
                m.header.stamp = clock->toRosTime(seconds);
                m.name = names;
                m.position = positions;
                return m;
            }

            bool publishAndWait(const sensor_msgs::msg::JointState &message)
            {
                const std::size_t seen = node->messagesReceived();
                publisher->publish(message);
                return spinUntil(executor, [&]()
                                 { return node->messagesReceived() > seen; });
            }
        };
    } // namespace

    TEST(KinematicTendonNode, RejectsInvalidOptions)
    {
        TendonRig rig;
        auto make = [&](const Interface::Ros2::KinematicTendonOptions &options)
        { return std::make_shared<Interface::Ros2::KinematicTendonNode>(rig.queue, rig.clock, options); };

        EXPECT_THROW(std::make_shared<Interface::Ros2::KinematicTendonNode>(nullptr, rig.clock, rig.options), std::invalid_argument);
        EXPECT_THROW(std::make_shared<Interface::Ros2::KinematicTendonNode>(rig.queue, nullptr, rig.options), std::invalid_argument);

        auto options = rig.options;
        options.joint_name.clear();
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.topic.clear();
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.variance = 0.0;
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.scale = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(make(options), std::invalid_argument);
    }

    TEST(KinematicTendonNode, QueuesTheScaledPositionOfItsJoint)
    {
        TendonRig rig;
        rig.create();
        ASSERT_TRUE(rig.publishAndWait(rig.message(1.5, {"tendon_0", "tendon_1", "tendon_2"}, {1.0, 3.0, 5.0})));

        const auto measurements = rig.queue->peekAll();
        ASSERT_EQ(measurements.size(), 1u);
        const auto &m = measurements.front();
        EXPECT_NEAR(m->time, 1.5, 1e-9);
        ASSERT_EQ(m->value.size(), 1);
        EXPECT_NEAR(m->value(0, 0), 0.01 * 3.0 + 0.2, 1e-12);
        ASSERT_EQ(m->covariance.size(), 1);
        EXPECT_DOUBLE_EQ(m->covariance(0, 0), 4e-6);
        EXPECT_EQ(rig.node->measurementsAdded(), 1u);
    }

    TEST(KinematicTendonNode, OtherJointsAreIgnoredAndMissingPositionsDropped)
    {
        TendonRig rig;
        rig.create();
        ASSERT_TRUE(rig.publishAndWait(rig.message(1.0, {"tendon_0"}, {1.0})));
        EXPECT_TRUE(rig.queue->empty());
        EXPECT_EQ(rig.node->measurementsDropped(), 0u);

        ASSERT_TRUE(rig.publishAndWait(rig.message(1.1, {"tendon_0", "tendon_1"}, {1.0}))); // no position for it
        EXPECT_TRUE(rig.queue->empty());
        EXPECT_EQ(rig.node->measurementsDropped(), 1u);

        ASSERT_TRUE(rig.publishAndWait(rig.message(1.2, {"tendon_1"}, {2.0})));
        EXPECT_EQ(rig.queue->size(), 1u);
    }

    TEST(KinematicTendonNode, StampsNotNewerThanTheLastAreDropped)
    {
        TendonRig rig;
        rig.create();
        ASSERT_TRUE(rig.publishAndWait(rig.message(2.0, {"tendon_1"}, {1.0})));
        ASSERT_TRUE(rig.publishAndWait(rig.message(1.0, {"tendon_1"}, {1.0})));
        ASSERT_TRUE(rig.publishAndWait(rig.message(3.0, {"tendon_1"}, {1.0})));
        EXPECT_EQ(rig.queue->size(), 2u);
        EXPECT_EQ(rig.node->measurementsDropped(), 1u);
    }
} // namespace Spacetime
