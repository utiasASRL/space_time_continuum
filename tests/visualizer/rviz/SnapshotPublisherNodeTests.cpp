#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/qos.hpp>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/utilities/Ellipsoid.hpp>
#include <spacetime/estimator/joint_models/Joint.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/RobotShape.hpp>
#include <spacetime/interface/ros2/Conversions.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include <spacetime/visualizer/rviz/SnapshotPublisherNode.hpp>
#include "../../ros2/Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;
    namespace Rviz = Visualization::Rviz;

    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using PriorT = Priors::WNOA::Prior<Pose>;
        using Marker = visualization_msgs::msg::Marker;

        constexpr unsigned int kSpatialDim = 4;

        /// Straight robots along z (unit strain), the second one shifted along x.
        struct PublisherRig
        {
            std::shared_ptr<StateType> state_a = std::make_shared<StateType>(1.0, kSpatialDim);
            std::shared_ptr<StateType> state_b = std::make_shared<StateType>(0.5, kSpatialDim);
            std::shared_ptr<PriorT> prior_a, prior_b;
            KeyManager km;
            gtsam::Values values;
            std::shared_ptr<const EstimatorSnapshot> snapshot;

            std::shared_ptr<Ros2::RosClock> clock = std::make_shared<Ros2::RosClock>(std::make_shared<rclcpp::Clock>(RCL_ROS_TIME), 0.0);
            Rviz::SnapshotPublisherOptions options;
            std::shared_ptr<Rviz::SnapshotPublisherNode> node;
            std::shared_ptr<rclcpp::Node> listener;
            rclcpp::executors::SingleThreadedExecutor executor;

            std::vector<geometry_msgs::msg::PoseArray> poses_a;
            std::vector<visualization_msgs::msg::MarkerArray> markers;
            std::vector<tf2_msgs::msg::TFMessage> frames;
            std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;

            PublisherRig()
            {
                ValueType reference;
                reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior_a = std::make_shared<PriorT>(state_a, reference);
                ValueType reference_b = reference;
                reference_b.pose(Pose(gtsam::Rot3(), gtsam::Point3(0.2, 0.0, 0.0)));
                prior_b = std::make_shared<PriorT>(state_b, reference_b);

                clock->tryStart();
                options.rate = 0.0; // published by hand (publishLatest)
                options.topic_prefix = uniqueName("spacetime");
                options.tf_topic = "/" + uniqueName("tf");
            }

            void addRow(double time)
            {
                gtsam::Values new_values;
                prior_a->extendState(km, time, values, new_values);
                prior_b->extendState(km, time, values, new_values);
                values.insert(new_values);
            }

            void publishSnapshot(double time, const gtsam::Values &snapshot_values)
            {
                snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{time, snapshot_values});
            }

            void create()
            {
                node = std::make_shared<Rviz::SnapshotPublisherNode>([this]()
                                                                     { return snapshot; }, clock, options);
            }

            /// Subscribes to robot "a"'s poses, the markers and the node frames; call after create() and addRobot().
            void listen()
            {
                listener = std::make_shared<rclcpp::Node>(uniqueName("snapshot_listener"));
                subscriptions.push_back(listener->create_subscription<geometry_msgs::msg::PoseArray>(
                    options.topic_prefix + "/a/poses", 10, [this](const geometry_msgs::msg::PoseArray &m)
                    { poses_a.push_back(m); }));
                subscriptions.push_back(listener->create_subscription<visualization_msgs::msg::MarkerArray>(
                    options.topic_prefix + "/markers", 10, [this](const visualization_msgs::msg::MarkerArray &m)
                    { markers.push_back(m); }));
                subscriptions.push_back(listener->create_subscription<tf2_msgs::msg::TFMessage>(
                    options.tf_topic, tf2_ros::DynamicListenerQoS(), [this](const tf2_msgs::msg::TFMessage &m)
                    { frames.push_back(m); }));
                executor.add_node(node);
                executor.add_node(listener);
                ASSERT_TRUE(spinUntil(executor, [&]()
                                      { return node->count_subscribers(options.topic_prefix + "/markers") > 0 &&
                                               node->count_subscribers(options.topic_prefix + "/a/poses") > 0; }));
            }

            /// publishLatest(), then waits for the markers it sent.
            bool publishAndReceive()
            {
                const std::size_t received = markers.size();
                if (!node->publishLatest())
                    return false;
                return spinUntil(executor, [&]()
                                 { return markers.size() > received; });
            }

            std::vector<Marker> markersIn(const std::string &ns) const
            {
                std::vector<Marker> found;
                for (const auto &marker : markers.back().markers)
                    if (marker.ns == ns && marker.action == Marker::ADD)
                        found.push_back(marker);
                return found;
            }
        };
    } // namespace

    TEST(SnapshotPublisherNode, RejectsInvalidArguments)
    {
        PublisherRig rig;
        EXPECT_THROW(Rviz::SnapshotPublisherNode(nullptr, rig.clock, rig.options), std::invalid_argument);
        EXPECT_THROW(Rviz::SnapshotPublisherNode([]()
                                                 { return nullptr; }, nullptr, rig.options),
                     std::invalid_argument);

        rig.create();
        EXPECT_THROW(rig.node->addRobot("a", std::shared_ptr<PriorT>()), std::invalid_argument);
        rig.node->addRobot("a", rig.prior_a);
        EXPECT_THROW(rig.node->addRobot("a", rig.prior_b), std::invalid_argument);    // name taken
        EXPECT_THROW(rig.node->addRobot("other", rig.prior_a), std::invalid_argument); // robot added already
        EXPECT_THROW(rig.node->addRobot("bad name!", rig.prior_b), std::invalid_argument);

        // Joints are drawn between robots added before them, in one frame.
        Joints::Joint<PriorT, PriorT> joint(rig.prior_a, 1.0, rig.prior_b, 0.0, Eigen::MatrixXd::Identity(6, 6) * 1e-6);
        EXPECT_THROW(rig.node->addJoint("joint", joint), std::invalid_argument);
        Rviz::RobotDisplayOptions other_frame;
        other_frame.frame_id = "elsewhere";
        rig.node->addRobot("b", rig.prior_b, other_frame);
        EXPECT_THROW(rig.node->addJoint("joint", joint), std::invalid_argument);
    }

    TEST(SnapshotPublisherNode, NothingIsPublishedWithoutANewSnapshot)
    {
        PublisherRig rig;
        rig.create();
        rig.node->addRobot("a", rig.prior_a);

        EXPECT_FALSE(rig.node->publishLatest()); // no snapshot yet

        rig.addRow(0.0);
        rig.publishSnapshot(0.0, rig.values);
        EXPECT_TRUE(rig.node->publishLatest());
        EXPECT_FALSE(rig.node->publishLatest()); // the same one again

        rig.addRow(0.1);
        rig.publishSnapshot(0.1, rig.values);
        EXPECT_TRUE(rig.node->publishLatest());
    }

    TEST(SnapshotPublisherNode, PublishesTheBackboneAsAPoseArray)
    {
        PublisherRig rig;
        rig.create();
        Rviz::RobotDisplayOptions display;
        display.frame_id = "aurora_field";
        display.extra_points = 2;
        rig.node->addRobot("a", rig.prior_a, display);
        rig.listen();
        rig.addRow(0.0);
        rig.addRow(0.1);
        rig.publishSnapshot(0.1, rig.values);

        ASSERT_TRUE(rig.publishAndReceive());
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return !rig.poses_a.empty(); }));

        Utilities::RobotShape<NodeType> shape(rig.state_a);
        shape.bindPriorInterpolator(rig.prior_a);
        const auto backbone = shape.backbone(0.1, *rig.snapshot, rig.state_a->snapshotEstimationNodes(), 2, false);
        ASSERT_TRUE(backbone);

        const auto &poses = rig.poses_a.back();
        EXPECT_EQ(poses.header.frame_id, "aurora_field");
        EXPECT_EQ(rclcpp::Time(poses.header.stamp).nanoseconds(), rclcpp::Time(rig.clock->toRosTime(0.1)).nanoseconds());
        ASSERT_EQ(poses.poses.size(), backbone->points.size());
        for (std::size_t i = 0; i < poses.poses.size(); i++)
            EXPECT_TRUE(Ros2::toPose3(poses.poses[i]).equals(Pose(backbone->points[i].pose), 1e-9)) << "point " << i;
    }

    TEST(SnapshotPublisherNode, DrawsTheRobotAsATubeOfCylinders)
    {
        PublisherRig rig;
        rig.create();
        Rviz::RobotDisplayOptions display;
        display.radius = 0.005;
        display.extra_points = 3;
        rig.node->addRobot("a", rig.prior_a, display);
        rig.listen();
        rig.addRow(0.0);
        rig.publishSnapshot(0.0, rig.values);

        ASSERT_TRUE(rig.publishAndReceive());

        const auto &array = rig.markers.back().markers;
        ASSERT_FALSE(array.empty());
        EXPECT_EQ(array.front().action, Marker::DELETEALL); // clears the previous frame's markers

        const auto tube = rig.markersIn("a");
        const std::size_t points = kSpatialDim + (kSpatialDim - 1) * display.extra_points;
        ASSERT_EQ(tube.size(), points - 1);
        double total_length = 0.0;
        for (std::size_t i = 0; i < tube.size(); i++)
        {
            EXPECT_EQ(tube[i].type, Marker::CYLINDER);
            EXPECT_EQ(tube[i].id, static_cast<int>(i));
            EXPECT_DOUBLE_EQ(tube[i].scale.x, 0.01);
            EXPECT_DOUBLE_EQ(tube[i].scale.y, 0.01);
            EXPECT_EQ(tube[i].header.frame_id, display.frame_id);
            total_length += tube[i].scale.z;
        }
        EXPECT_NEAR(total_length, 1.0, 1e-9);
        // Alternating shades per node-to-node segment (as the VTK view draws it).
        const std::size_t per_segment = display.extra_points + 1;
        EXPECT_FLOAT_EQ(tube[0].color.r, tube[per_segment - 1].color.r);
        EXPECT_NE(tube[0].color.r, tube[per_segment].color.r);
        EXPECT_FLOAT_EQ(tube[0].color.r, tube[2 * per_segment].color.r);
    }

    TEST(SnapshotPublisherNode, HidesARobotTheSnapshotCannotPlace)
    {
        PublisherRig rig;
        rig.create();
        rig.node->addRobot("a", rig.prior_a);
        rig.listen();
        rig.addRow(0.0);
        rig.publishSnapshot(0.0, gtsam::Values()); // e.g. marginalized out

        ASSERT_TRUE(rig.publishAndReceive());
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return !rig.poses_a.empty(); }));

        EXPECT_TRUE(rig.poses_a.back().poses.empty());
        EXPECT_TRUE(rig.markersIn("a").empty());
    }

    TEST(SnapshotPublisherNode, BroadcastsNodeFramesWhenAsked)
    {
        PublisherRig rig;
        rig.create();
        Rviz::RobotDisplayOptions display;
        display.frame_id = "aurora_field";
        display.node_frames = true;
        rig.node->addRobot("a", rig.prior_a, display);
        rig.listen();
        rig.addRow(0.0);
        rig.publishSnapshot(0.0, rig.values);

        ASSERT_TRUE(rig.publishAndReceive());
        ASSERT_TRUE(spinUntil(rig.executor, [&]()
                              { return !rig.frames.empty(); }));

        const auto &transforms = rig.frames.back().transforms;
        ASSERT_EQ(transforms.size(), kSpatialDim);
        const auto nodes = rig.state_a->snapshotEstimationNodes();
        for (unsigned int i = 0; i < kSpatialDim; i++)
        {
            EXPECT_EQ(transforms[i].header.frame_id, "aurora_field");
            EXPECT_EQ(transforms[i].child_frame_id, "a/node_" + std::to_string(i));
            EXPECT_TRUE(Ros2::toPose3(transforms[i].transform).equals(rig.values.at<Pose>(nodes[i]->poseKey()), 1e-9));
        }
    }

    TEST(SnapshotPublisherNode, DrawsJoints)
    {
        PublisherRig rig;
        rig.create();
        rig.node->addRobot("a", rig.prior_a);
        rig.node->addRobot("b", rig.prior_b);

        // A ball joint (no offset) between a's tip and b's base, and a link (with an offset) between their bases.
        Joints::Joint<PriorT, PriorT> ball(rig.prior_a, 1.0, rig.prior_b, 0.0, Eigen::MatrixXd::Identity(6, 6) * 1e-6);
        Joints::JointOptions link_options;
        link_options.offset = Pose(gtsam::Rot3(), gtsam::Point3(0.2, 0.0, 0.0));
        Joints::Joint<PriorT, PriorT> link(rig.prior_a, 0.0, rig.prior_b, 0.0, Eigen::MatrixXd::Identity(6, 6) * 1e-6, link_options);
        rig.node->addJoint("ball", ball, 0.01);
        rig.node->addJoint("link", link, 0.01);
        EXPECT_THROW(rig.node->addJoint("ball", ball), std::invalid_argument); // name taken
        rig.listen();
        rig.addRow(0.0);
        rig.publishSnapshot(0.0, rig.values);

        ASSERT_TRUE(rig.publishAndReceive());

        const auto ball_markers = rig.markersIn("ball"), link_markers = rig.markersIn("link");
        ASSERT_EQ(ball_markers.size(), 1u);
        ASSERT_EQ(link_markers.size(), 1u);

        // a's tip is at (0, 0, 1), b's base at (0.2, 0, 0).
        EXPECT_EQ(ball_markers[0].type, Marker::SPHERE);
        EXPECT_NEAR(ball_markers[0].pose.position.x, 0.1, 1e-9);
        EXPECT_NEAR(ball_markers[0].pose.position.z, 0.5, 1e-9);
        EXPECT_DOUBLE_EQ(ball_markers[0].scale.x, 0.02);

        EXPECT_EQ(link_markers[0].type, Marker::CYLINDER);
        EXPECT_NEAR(link_markers[0].pose.position.x, 0.1, 1e-9);
        EXPECT_NEAR(link_markers[0].pose.position.z, 0.0, 1e-9);
        EXPECT_NEAR(link_markers[0].scale.z, 0.2, 1e-9);
    }

    TEST(SnapshotPublisherNode, CovarianceEllipsoidsNeedASnapshotCovariance)
    {
        PublisherRig rig;
        rig.create();
        Rviz::RobotDisplayOptions display;
        display.covariance = true;
        rig.node->addRobot("a", rig.prior_a, display);
        rig.listen();
        rig.addRow(0.0);
        rig.publishSnapshot(0.0, rig.values); // without a covariance (compute_covariance off)

        ASSERT_TRUE(rig.publishAndReceive());

        EXPECT_FALSE(rig.markersIn("a").empty());
        EXPECT_TRUE(rig.markersIn("a/covariance").empty());
    }

    namespace
    {
        /// Solves robot "a" over two rows with posterior covariances, as a live estimator does.
        std::shared_ptr<const EstimatorSnapshot> solveWithCovariance(PublisherRig &rig)
        {
            EstimatorOptions options;
            options.smootherLag = 1.0;
            options.compute_covariance = true;
            auto estimator = std::make_shared<Estimator>(options);
            estimator->addPrior(rig.prior_a);
            estimator->estimate(0.0);
            estimator->estimate(0.1);
            return estimator->getSnapshot();
        }

        Eigen::Matrix3d orientation(const Marker &marker)
        {
            return Eigen::Quaterniond(marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z)
                .toRotationMatrix();
        }
    } // namespace

    // One 99.7% position ellipsoid per backbone point (nodes and extra points), from the
    // posterior pose covariance rotated into the world: R Sigma_tt R^T.
    TEST(SnapshotPublisherNode, DrawsTheCovarianceEllipsoidOfEveryBackbonePoint)
    {
        PublisherRig rig;
        rig.create();
        Rviz::RobotDisplayOptions display;
        display.covariance = true;
        display.extra_points = 2;
        rig.node->addRobot("a", rig.prior_a, display);
        rig.listen();
        rig.snapshot = solveWithCovariance(rig);
        ASSERT_TRUE(rig.snapshot && rig.snapshot->covariance);

        ASSERT_TRUE(rig.publishAndReceive());

        Utilities::RobotShape<NodeType> shape(rig.state_a);
        shape.bindPriorInterpolator(rig.prior_a);
        const auto backbone = shape.backbone(rig.snapshot->time, *rig.snapshot, rig.state_a->snapshotEstimationNodes(), 2, false,
                                             rig.snapshot->covariance.get());
        ASSERT_TRUE(backbone);

        const auto ellipsoids = rig.markersIn("a/covariance");
        ASSERT_EQ(ellipsoids.size(), backbone->points.size());
        for (std::size_t k = 0; k < ellipsoids.size(); k++)
        {
            const auto &point = backbone->points[k];
            ASSERT_GE(point.covariance.rows(), 6) << "point " << k;
            const Eigen::Matrix3d R = point.pose.block<3, 3>(0, 0);
            const Eigen::Matrix3d expected = R * point.covariance.block<3, 3>(3, 3) * R.transpose();
            const double s = Utilities::kEllipsoidScale99_7;

            const Marker &marker = ellipsoids[k];
            EXPECT_EQ(marker.type, Marker::SPHERE);
            EXPECT_EQ(marker.id, static_cast<int>(k));
            EXPECT_EQ(marker.header.frame_id, display.frame_id);
            EXPECT_LT(marker.color.a, 1.0f); // translucent, so the robot shows through
            EXPECT_NEAR(marker.pose.position.x, point.pose(0, 3), 1e-9);
            EXPECT_NEAR(marker.pose.position.y, point.pose(1, 3), 1e-9);
            EXPECT_NEAR(marker.pose.position.z, point.pose(2, 3), 1e-9);
            // The marker's semi-axes (half its scale, along its orientation) span the ellipsoid.
            const Eigen::Matrix3d axes = orientation(marker) * Eigen::Vector3d(marker.scale.x, marker.scale.y, marker.scale.z).asDiagonal() * 0.5;
            EXPECT_TRUE((axes * axes.transpose()).isApprox(s * s * expected, 1e-6)) << "point " << k;
        }
    }

    TEST(SnapshotPublisherNode, CovarianceEllipsoidsAreOnlyDrawnWhenAsked)
    {
        PublisherRig rig;
        rig.create();
        rig.node->addRobot("a", rig.prior_a); // covariance off by default
        rig.listen();
        rig.snapshot = solveWithCovariance(rig);
        ASSERT_TRUE(rig.snapshot && rig.snapshot->covariance);

        ASSERT_TRUE(rig.publishAndReceive());

        EXPECT_FALSE(rig.markersIn("a").empty());
        EXPECT_TRUE(rig.markersIn("a/covariance").empty());
    }

    TEST(SnapshotPublisherNode, TimerPublishesAtTheRate)
    {
        PublisherRig rig;
        rig.options.rate = 50.0;
        rig.create();
        rig.node->addRobot("a", rig.prior_a);
        rig.listen();
        rig.addRow(0.0);
        rig.publishSnapshot(0.0, rig.values);

        EXPECT_TRUE(spinUntil(rig.executor, [&]()
                              { return !rig.markers.empty(); }));
    }
} // namespace Spacetime
