#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/Conversions.hpp>
#include <spacetime/interface/ros2/PoseSensorNode.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include "Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;

    namespace
    {
        /// A sensor node on its own /tf-like topic, a publisher feeding it, and the queue it fills.
        struct SensorRig
        {
            std::shared_ptr<Types::MeasurementQueue> queue = std::make_shared<Types::MeasurementQueue>();
            std::shared_ptr<Interface::Ros2::RosClock> clock = std::make_shared<Interface::Ros2::RosClock>(std::make_shared<rclcpp::Clock>(RCL_ROS_TIME), 0.0);
            Interface::Ros2::PoseSensorOptions options;
            std::shared_ptr<Interface::Ros2::PoseSensorNode> node;
            std::shared_ptr<rclcpp::Node> publisher_node;
            rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr publisher;
            rclcpp::executors::SingleThreadedExecutor executor;

            explicit SensorRig(bool start_clock = true)
            {
                options.parent_frame = uniqueName("field");
                options.child_frame = uniqueName("sensor");
                options.topic = "/" + uniqueName("tf");
                options.arclength = 0.25;
                options.offset = gtsam::Pose3(gtsam::Rot3::Rx(0.4), gtsam::Point3(0.01, 0.0, -0.02));
                options.covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
                options.covariance(0, 1) = options.covariance(1, 0) = 1e-5;
                options.mask << 1, 1, 0, 1, 1, 1;
                if (start_clock)
                    clock->tryStart();
            }

            void create()
            {
                node = std::make_shared<Interface::Ros2::PoseSensorNode>(queue, clock, options);
                publisher_node = std::make_shared<rclcpp::Node>(uniqueName("tf_publisher"));
                publisher = publisher_node->create_publisher<tf2_msgs::msg::TFMessage>(options.topic, tf2_ros::DynamicListenerQoS());
                executor.add_node(node);
                executor.add_node(publisher_node);
                // Wait for discovery, so nothing published below is lost.
                ASSERT_TRUE(spinUntil(executor, [&]()
                                      { return publisher->get_subscription_count() > 0; }));
            }

            /// The stamp `seconds` after the clock's origin.
            builtin_interfaces::msg::Time stampAt(double seconds) const { return clock->toRosTime(seconds); }

            geometry_msgs::msg::TransformStamped transform(const std::string &parent, const std::string &child, const gtsam::Pose3 &pose, double seconds) const
            {
                geometry_msgs::msg::TransformStamped t;
                t.header.frame_id = parent;
                t.header.stamp = stampAt(seconds);
                t.child_frame_id = child;
                t.transform = Interface::Ros2::toTransformMsg(pose);
                return t;
            }

            geometry_msgs::msg::TransformStamped matching(const gtsam::Pose3 &pose, double seconds) const
            {
                return transform(options.parent_frame, options.child_frame, pose, seconds);
            }

            void publish(const std::vector<geometry_msgs::msg::TransformStamped> &transforms)
            {
                tf2_msgs::msg::TFMessage message;
                message.transforms = transforms;
                publisher->publish(message);
            }

            /// Publishes, then spins until `expected_total` measurements are queued and the node has seen every message.
            bool publishAndWait(const std::vector<geometry_msgs::msg::TransformStamped> &transforms, std::size_t expected_total)
            {
                const std::size_t seen = node->messagesReceived();
                publish(transforms);
                return spinUntil(executor, [&]()
                                 { return node->messagesReceived() > seen && queue->size() >= expected_total; });
            }
        };

        const gtsam::Pose3 kSensorPose(gtsam::Rot3::RzRyRx(0.2, -0.5, 1.1), gtsam::Point3(0.3, -0.1, 0.05));
    } // namespace

    TEST(PoseSensorNode, ComputesTheBodyPoseFromTheSensorPoseAndOffset)
    {
        const gtsam::Pose3 T_bs(gtsam::Rot3::Ry(0.3), gtsam::Point3(0.0, 0.01, 0.02));
        const gtsam::Pose3 T_ib = Interface::Ros2::PoseSensorNode::bodyPose(kSensorPose, T_bs);
        EXPECT_TRUE((T_ib * T_bs).equals(kSensorPose, 1e-12)); // the sensor sits at T_bs in the body frame
    }

    TEST(PoseSensorNode, RejectsInvalidOptions)
    {
        SensorRig rig;
        auto make = [&](const Interface::Ros2::PoseSensorOptions &options)
        { return std::make_shared<Interface::Ros2::PoseSensorNode>(rig.queue, rig.clock, options); };

        EXPECT_THROW(std::make_shared<Interface::Ros2::PoseSensorNode>(nullptr, rig.clock, rig.options), std::invalid_argument);
        EXPECT_THROW(std::make_shared<Interface::Ros2::PoseSensorNode>(rig.queue, nullptr, rig.options), std::invalid_argument);

        Interface::Ros2::PoseSensorOptions options = rig.options;
        options.parent_frame.clear();
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.child_frame.clear();
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.body_frame = options.child_frame; // would feed itself
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.body_frame = options.parent_frame;
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.covariance(2, 2) = -1.0;
        EXPECT_THROW(make(options), std::invalid_argument);

        options = rig.options;
        options.covariance(0, 3) = 1e-3; // not symmetric
        EXPECT_THROW(make(options), std::invalid_argument);
    }

    TEST(PoseSensorNode, MatchingTransformQueuesAMeasurement)
    {
        SensorRig rig;
        rig.create();

        ASSERT_TRUE(rig.publishAndWait({rig.matching(kSensorPose, 1.5)}, 1));

        const auto measurements = rig.queue->peekAll();
        ASSERT_EQ(measurements.size(), 1u);
        const auto &m = measurements.front();
        EXPECT_NEAR(m->time, 1.5, 1e-9);
        EXPECT_DOUBLE_EQ(m->arclength, 0.25);
        EXPECT_TRUE(gtsam::Pose3(Eigen::Matrix4d(m->value)).equals(Interface::Ros2::PoseSensorNode::bodyPose(kSensorPose, rig.options.offset), 1e-9));
        EXPECT_TRUE(m->covariance.isApprox(Eigen::MatrixXd(rig.options.covariance)));
        EXPECT_EQ(m->mask, Eigen::MatrixXi(rig.options.mask));
        EXPECT_EQ(rig.node->measurementsAdded(), 1u);
    }

    TEST(PoseSensorNode, OtherFramesAreIgnored)
    {
        SensorRig rig;
        rig.create();

        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return true; }));
        const std::size_t seen = rig.node->messagesReceived();
        rig.publish({rig.transform(rig.options.parent_frame, "other_sensor", kSensorPose, 1.0)});
        rig.publish({rig.transform("other_field", rig.options.child_frame, kSensorPose, 1.1)});
        rig.publish({rig.transform(rig.options.child_frame, rig.options.parent_frame, kSensorPose, 1.2)}); // inverted edge
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return rig.node->messagesReceived() >= seen + 3; }));
        EXPECT_TRUE(rig.queue->empty());

        // Frame ids are compared without a leading slash (tf2 strips it too).
        ASSERT_TRUE(rig.publishAndWait({rig.transform("/" + rig.options.parent_frame, rig.options.child_frame, kSensorPose, 1.3)}, 1));
        EXPECT_EQ(rig.queue->size(), 1u);
    }

    TEST(PoseSensorNode, PicksTheMatchingTransformOutOfAMessage)
    {
        SensorRig rig;
        rig.create();

        ASSERT_TRUE(rig.publishAndWait({rig.transform(rig.options.parent_frame, "other_sensor", gtsam::Pose3(), 1.0),
                                        rig.matching(kSensorPose, 1.0),
                                        rig.transform("other_field", "other_sensor", gtsam::Pose3(), 1.0)},
                                       1));
        EXPECT_EQ(rig.queue->size(), 1u);
    }

    // MeasurementQueue drains in insertion order, so a stamp that is not newer than the last
    // one queued would be estimated out of order.
    TEST(PoseSensorNode, StampsNotNewerThanTheLastAreDropped)
    {
        SensorRig rig;
        rig.create();

        ASSERT_TRUE(rig.publishAndWait({rig.matching(kSensorPose, 2.0)}, 1));
        const std::size_t seen = rig.node->messagesReceived();
        rig.publish({rig.matching(kSensorPose, 1.0)}); // older
        rig.publish({rig.matching(kSensorPose, 2.0)}); // repeated
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return rig.node->messagesReceived() >= seen + 2; }));
        ASSERT_TRUE(rig.publishAndWait({rig.matching(kSensorPose, 3.0)}, 2));

        const auto measurements = rig.queue->peekAll();
        ASSERT_EQ(measurements.size(), 2u);
        EXPECT_NEAR(measurements[0]->time, 2.0, 1e-9);
        EXPECT_NEAR(measurements[1]->time, 3.0, 1e-9);
        EXPECT_EQ(rig.node->measurementsDropped(), 2u);
    }

    TEST(PoseSensorNode, StampsBeforeTheStartTimeAreDropped)
    {
        SensorRig rig;
        rig.create();

        const std::size_t seen = rig.node->messagesReceived();
        rig.publish({rig.matching(kSensorPose, -0.5)});
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return rig.node->messagesReceived() > seen; }));
        EXPECT_TRUE(rig.queue->empty());
        EXPECT_EQ(rig.node->measurementsDropped(), 1u);
    }

    TEST(PoseSensorNode, NothingIsQueuedBeforeTheClockStarts)
    {
        SensorRig rig(false);
        rig.create();

        geometry_msgs::msg::TransformStamped t;
        t.header.frame_id = rig.options.parent_frame;
        t.child_frame_id = rig.options.child_frame;
        t.header.stamp = rclcpp::Clock(RCL_SYSTEM_TIME).now();
        t.transform = Interface::Ros2::toTransformMsg(kSensorPose);
        rig.publish({t});
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return rig.node->messagesReceived() > 0; }));
        EXPECT_TRUE(rig.queue->empty());
        EXPECT_EQ(rig.node->measurementsDropped(), 1u);
    }

    // The Aurora's 5DoF sensors point their z axis down the robot (towards the base) with
    // x -> -y and y -> -x: their offset rotation turns the sensor's -z into the body tangent.
    TEST(PoseSensorNode, FlippedSensorOffsetAlignsTheBodyTangent)
    {
        Eigen::Matrix3d R_bs;
        R_bs << 0, -1, 0,
            -1, 0, 0,
            0, 0, -1; // columns: the sensor's x, y, z axes in the body frame
        const gtsam::Pose3 T_bs(gtsam::Rot3(R_bs), gtsam::Point3::Zero());

        const gtsam::Pose3 T_ib = Interface::Ros2::PoseSensorNode::bodyPose(kSensorPose, T_bs);

        const Eigen::Matrix3d R_is = kSensorPose.rotation().matrix(), R_ib = T_ib.rotation().matrix();
        EXPECT_TRUE(R_ib.col(2).isApprox(-R_is.col(2), 1e-12)); // body tangent = -sensor z
        EXPECT_TRUE(R_ib.col(0).isApprox(-R_is.col(1), 1e-12));
        EXPECT_TRUE(R_ib.col(1).isApprox(-R_is.col(0), 1e-12));
        EXPECT_TRUE(T_ib.translation().isApprox(kSensorPose.translation(), 1e-12));
    }

    TEST(PoseSensorNode, BroadcastsTheBodyFrame)
    {
        SensorRig rig;
        rig.options.body_frame = rig.options.child_frame + "/body";
        rig.create();

        std::vector<geometry_msgs::msg::TransformStamped> broadcast;
        auto subscription = rig.publisher_node->create_subscription<tf2_msgs::msg::TFMessage>(
            rig.options.topic, tf2_ros::DynamicListenerQoS(), [&](const tf2_msgs::msg::TFMessage &message)
            {
                for (const auto &t : message.transforms)
                    if (t.child_frame_id == rig.options.body_frame)
                        broadcast.push_back(t); });

        ASSERT_TRUE(rig.publishAndWait({rig.matching(kSensorPose, 1.0)}, 1));
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return !broadcast.empty(); }));

        EXPECT_EQ(broadcast.front().header.frame_id, rig.options.parent_frame);
        EXPECT_EQ(rclcpp::Time(broadcast.front().header.stamp).nanoseconds(), rclcpp::Time(rig.stampAt(1.0)).nanoseconds());
        EXPECT_TRUE(Interface::Ros2::toPose3(broadcast.front().transform).equals(Interface::Ros2::PoseSensorNode::bodyPose(kSensorPose, rig.options.offset), 1e-9));
        EXPECT_EQ(rig.queue->size(), 1u); // its own broadcast is not a measurement
    }
} // namespace Spacetime
