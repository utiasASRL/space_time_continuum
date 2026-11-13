#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/RobotShape.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/Interfaces.hpp>
#include <spacetime/interface/ros2/Conversions.hpp>
#include <spacetime/interface/ros2/Interface.hpp>
#include "Ros2TestUtilities.hpp"

namespace Spacetime
{
    using namespace Ros2Tests;

    static_assert(Interface::InterfaceType<Interfaces::ROS2>);

    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using PriorT = Priors::WNOA::Prior<Pose>;
        using SensorT = Sensors::PoseSensor<ValueType>;

        constexpr double kLength = 0.2;

        /// A sliding-window estimator of one straight robot along z.
        struct EstimatorRig
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(kLength, 3);
            std::shared_ptr<PriorT> prior;
            Estimator estimator;

            explicit EstimatorRig(double smoother_lag = 0.3) : estimator(options(smoother_lag))
            {
                ValueType reference;
                reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<PriorT>(state, reference);
                estimator.addPrior(prior);
            }

            static EstimatorOptions options(double smoother_lag)
            {
                EstimatorOptions options;
                options.smootherLag = smoother_lag;
                return options;
            }
        };

        Ros2::SessionRosOptions rosOptions()
        {
            Ros2::SessionRosOptions options;
            options.node_prefix = uniqueName("spacetime");
            return options;
        }

        Interface::SessionOptions sessionOptions(double end_time, double period = 0.02)
        {
            Interface::SessionOptions options;
            options.end_time = end_time;
            options.period = period;
            options.real_time = true;
            return options;
        }
    } // namespace

    TEST(Ros2Session, NameIsRos2)
    {
        EXPECT_EQ(Interfaces::ROS2::name, "ros2");
    }

    TEST(Ros2Session, IsRealTime)
    {
        EstimatorRig rig;
        Ros2::Session session(rig.estimator, sessionOptions(-1.0), rosOptions());
        EXPECT_TRUE(session.isRealTime());
    }

    TEST(Ros2Session, BatchEstimatorCannotStart)
    {
        EstimatorRig rig(Estimator::BATCH_LAG);
        Ros2::Session session(rig.estimator, sessionOptions(-1.0), rosOptions());
        EXPECT_THROW(session.start(), std::invalid_argument);
    }

    TEST(Ros2Session, NonPositivePeriodCannotStart)
    {
        EstimatorRig rig;
        Ros2::Session session(rig.estimator, sessionOptions(-1.0, 0.0), rosOptions());
        EXPECT_THROW(session.start(), std::invalid_argument);
    }

    TEST(Ros2Session, EndTime)
    {
        EstimatorRig rig;
        Ros2::Session live(rig.estimator, sessionOptions(-1.0), rosOptions());
        EXPECT_EQ(live.endTime(), std::numeric_limits<double>::infinity());

        Ros2::Session bounded(rig.estimator, sessionOptions(3.0), rosOptions());
        EXPECT_DOUBLE_EQ(bounded.endTime(), 3.0);
    }

    TEST(Ros2Session, ReportsQueuesNothingFeeds)
    {
        EstimatorRig rig;
        Ros2::Session session(rig.estimator, sessionOptions(-1.0), rosOptions());
        auto sensor = std::make_shared<SensorT>(Eigen::Matrix<double, 6, 6>::Identity(), rig.prior);
        rig.estimator.addSensor(sensor);

        session.addMeasurementQueues(rig.estimator.getMeasurementQueues());
        ASSERT_EQ(session.unfedQueues().size(), 1u);
        EXPECT_EQ(session.unfedQueues().front(), sensor);

        Ros2::PoseSensorOptions options;
        options.parent_frame = uniqueName("field");
        options.child_frame = uniqueName("sensor");
        options.topic = "/" + uniqueName("tf");
        session.addPoseSensor(sensor, options);
        EXPECT_TRUE(session.unfedQueues().empty());
    }

    TEST(Ros2Session, EstimatesEveryPeriodUntilTheEndTime)
    {
        EstimatorRig rig;
        Ros2::Session session(rig.estimator, sessionOptions(0.2), rosOptions());
        std::mutex mutex;
        std::vector<double> times;
        session.setTickCallback([&](double time)
                                { std::lock_guard<std::mutex> lock(mutex); times.push_back(time); });

        session.start();
        session.join();

        EXPECT_TRUE(session.finished());
        std::lock_guard<std::mutex> lock(mutex);
        ASSERT_GE(times.size(), 3u);
        EXPECT_GE(times.front(), 0.0);
        // A period apart (never a near-duplicate row, e.g. one clamped to the end time), up to the end time.
        for (std::size_t i = 1; i < times.size(); i++)
            EXPECT_GT(times[i] - times[i - 1], 0.5 * 0.02) << "estimate " << i;
        EXPECT_LE(times.back(), 0.2);
        EXPECT_GT(times.back(), 0.2 - 2 * 0.02);
        ASSERT_TRUE(rig.estimator.getSnapshot());
        EXPECT_DOUBLE_EQ(rig.estimator.getSnapshot()->time, times.back());
    }

    TEST(Ros2Session, RequestStopEndsALiveRun)
    {
        EstimatorRig rig;
        Ros2::Session session(rig.estimator, sessionOptions(-1.0), rosOptions());
        std::atomic<int> ticks{0};
        session.setTickCallback([&](double)
                                { ticks++; });

        session.start();
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (ticks.load() < 2 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(5ms);
        EXPECT_GE(ticks.load(), 2);
        EXPECT_FALSE(session.finished());
        EXPECT_GT(session.now(), 0.0);

        session.requestStop();
        session.join();
        EXPECT_TRUE(session.finished());
    }

    TEST(Ros2Session, JoinRethrowsAnErrorRaisedWhileEstimating)
    {
        EstimatorRig rig;
        Ros2::Session session(rig.estimator, sessionOptions(-1.0), rosOptions());
        session.setTickCallback([](double)
                                { throw std::runtime_error("tick failed"); });

        session.start();
        EXPECT_THROW(session.join(), std::runtime_error);
        EXPECT_TRUE(session.finished());
    }

    // End to end: a static robot published on TF (6DoF at the base, a flipped 5DoF sensor at the
    // tip, as on the Aurora) is estimated from the sensor nodes' measurements.
    TEST(Ros2Session, EstimatesARobotFromTransforms)
    {
        EstimatorRig rig;
        const Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        auto base_sensor = std::make_shared<SensorT>(covariance, rig.prior);
        auto tip_sensor = std::make_shared<SensorT>(covariance, rig.prior);
        rig.estimator.addSensor(base_sensor);
        rig.estimator.addSensor(tip_sensor);

        Ros2::Session session(rig.estimator, sessionOptions(1.0), rosOptions());
        session.addMeasurementQueues(rig.estimator.getMeasurementQueues());

        const std::string topic = "/" + uniqueName("tf"), field = uniqueName("field");
        Ros2::PoseSensorOptions base_options;
        base_options.parent_frame = field;
        base_options.child_frame = uniqueName("base_sensor");
        base_options.topic = topic;
        base_options.covariance = covariance;
        session.addPoseSensor(base_sensor, base_options);

        Eigen::Matrix3d R_bs;
        R_bs << 0, -1, 0,
            -1, 0, 0,
            0, 0, -1;
        Ros2::PoseSensorOptions tip_options = base_options;
        tip_options.child_frame = uniqueName("tip_sensor");
        tip_options.arclength = kLength;
        tip_options.offset = Pose(gtsam::Rot3(R_bs), gtsam::Point3::Zero());
        tip_options.mask << 1, 1, 0, 1, 1, 1;
        session.addPoseSensor(tip_sensor, tip_options);
        EXPECT_TRUE(session.unfedQueues().empty());

        // The true robot: straight, based at T_base.
        const Pose T_base(gtsam::Rot3::Rx(0.3), gtsam::Point3(0.05, -0.02, 0.01));
        const Pose T_tip = T_base * Pose(gtsam::Rot3(), gtsam::Point3(0, 0, kLength));
        auto broadcaster = std::make_shared<rclcpp::Node>(uniqueName("tf_broadcaster"));
        auto publisher = broadcaster->create_publisher<tf2_msgs::msg::TFMessage>(topic, tf2_ros::DynamicListenerQoS());
        auto timer = broadcaster->create_wall_timer(20ms, [&]()
                                                    {
            tf2_msgs::msg::TFMessage message;
            geometry_msgs::msg::TransformStamped t;
            t.header.frame_id = field;
            t.header.stamp = broadcaster->now();
            t.child_frame_id = base_options.child_frame;
            t.transform = Ros2::toTransformMsg(T_base * base_options.offset);
            message.transforms.push_back(t);
            t.child_frame_id = tip_options.child_frame;
            t.transform = Ros2::toTransformMsg(T_tip * tip_options.offset);
            message.transforms.push_back(t);
            publisher->publish(message); });
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(broadcaster);
        std::jthread broadcasting([&](std::stop_token stop)
                                  { while (!stop.stop_requested()) executor.spin_some(10ms); });

        session.start();
        session.join();
        broadcasting.request_stop();

        const auto snapshot = rig.estimator.getSnapshot();
        ASSERT_TRUE(snapshot);
        Utilities::RobotShape<NodeType> shape(rig.state);
        shape.bindPriorInterpolator(rig.prior);
        const auto nodes = rig.state->snapshotEstimationNodes();
        const auto base = shape.poseAt(snapshot->time, 0.0, *snapshot, nodes, false);
        const auto tip = shape.poseAt(snapshot->time, kLength, *snapshot, nodes, false);
        ASSERT_TRUE(base && tip);
        EXPECT_LT((Pose(*base).translation() - T_base.translation()).norm(), 1e-3);
        EXPECT_LT((Pose(*tip).translation() - T_tip.translation()).norm(), 1e-3);
        EXPECT_LT(gtsam::Rot3::Logmap(Pose(*base).rotation().between(T_base.rotation())).norm(), 1e-2);
    }
} // namespace Spacetime
