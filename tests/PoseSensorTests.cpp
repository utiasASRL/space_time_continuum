#include <vector>

#include <Eigen/Dense>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    using Pose = gtsam::Pose3;
    using NodeType = Priors::WNOA::Node<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using ValueType = Priors::WNOA::NodeValues<Pose>;

    namespace
    {
        std::shared_ptr<Types::MeasurementBase> makePoseMeasurement(double time)
        {
            Eigen::Matrix4d value = Eigen::Matrix4d::Identity();
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
            return std::make_shared<Types::MeasurementBase>(0.0, time, value, covariance, Eigen::MatrixXi::Ones(6, 1));
        }

        std::shared_ptr<Sensors::PoseSensor<ValueType>> makeUnboundPoseSensor()
        {
            unsigned int spatial_dim = 2;
            auto state = std::make_shared<StateType>(1.0, spatial_dim);
            ValueType node_reference;
            auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
            return std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
        }

        std::vector<double> times(const std::vector<std::shared_ptr<Types::MeasurementBase>> &measurements)
        {
            std::vector<double> result;
            for (const auto &measurement : measurements)
                result.push_back(measurement->time);
            return result;
        }
    }

    TEST(SensorBase, MeasurementCallbackSeesEveryLaterMeasurement)
    {
        auto sensor = makeUnboundPoseSensor();
        std::vector<std::shared_ptr<Types::MeasurementBase>> seen;
        sensor->addMeasurementCallback([&seen](const auto &measurement)
                                       { seen.push_back(measurement); });
        auto measurement = makePoseMeasurement(0.5);
        sensor->addMeasurement(measurement);
        sensor->addMeasurement(makePoseMeasurement(1.0));

        EXPECT_EQ(times(seen), (std::vector<double>{0.5, 1.0}));
        EXPECT_EQ(seen.front(), measurement); // the very same measurement, not a copy
        EXPECT_EQ(sensor->size(), 2u); // observing doesn't consume
    }

    TEST(SensorBase, MeasurementCallbackReplaysQueuedMeasurementsInOrder)
    {
        auto sensor = makeUnboundPoseSensor();
        sensor->addMeasurement(makePoseMeasurement(0.0));
        sensor->addMeasurement(makePoseMeasurement(0.1));

        std::vector<std::shared_ptr<Types::MeasurementBase>> seen;
        sensor->addMeasurementCallback([&seen](const auto &measurement)
                                       { seen.push_back(measurement); });
        EXPECT_EQ(times(seen), (std::vector<double>{0.0, 0.1}));
        EXPECT_EQ(sensor->size(), 2u); // replay leaves the queue intact

        sensor->addMeasurement(makePoseMeasurement(0.2));
        EXPECT_EQ(times(seen), (std::vector<double>{0.0, 0.1, 0.2}));
    }

    TEST(SensorBase, MeasurementCallbackDoesNotReplayConsumedMeasurements)
    {
        auto sensor = makeUnboundPoseSensor();
        sensor->addMeasurement(makePoseMeasurement(0.0));
        sensor->addMeasurement(makePoseMeasurement(5.0));
        gtsam::Values new_values;
        sensor->getNewFactors(1.0, new_values); // drains the t = 0 measurement only

        std::vector<std::shared_ptr<Types::MeasurementBase>> seen;
        sensor->addMeasurementCallback([&seen](const auto &measurement)
                                       { seen.push_back(measurement); });
        EXPECT_EQ(times(seen), (std::vector<double>{5.0}));
    }

    TEST(SensorBase, EveryMeasurementCallbackIsInvoked)
    {
        auto sensor = makeUnboundPoseSensor();
        int first = 0, second = 0;
        sensor->addMeasurementCallback([&first](const auto &)
                                       { ++first; });
        sensor->addMeasurementCallback([&second](const auto &)
                                       { ++second; });
        sensor->addMeasurement(makePoseMeasurement(0.0));
        EXPECT_EQ(first, 1);
        EXPECT_EQ(second, 1);
    }

    TEST(PoseSensor, MeasurementsPastTheGraphAreExtrapolatedRatherThanDropped)
    {
        // Nodes sit at the estimation times, so a measurement arriving after the newest row
        // has no node after it. It must still reach the graph, by extrapolating off the
        // last row -- this is the behaviour that replaced extendState's extra off-grid node.
        unsigned int spatial_dim = 4;
        auto state = std::make_shared<StateType>(1.0, spatial_dim);

        ValueType node_reference;
        node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);

        Eigen::Matrix<double, 6, 6> measurement_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
        auto pose_sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);

        gtsam::Values current_values, new_values;
        KeyManager km;
        prior->extendState(km, 0.0, current_values, new_values);
        current_values.insert(new_values);
        new_values.clear();

        // The only row is at t = 0; this measurement lands 0.05 past it, at an
        // arclength that coincides with a spatial node -- so the time axis extrapolates
        // and the arclength axis does not participate.
        // NOTE: MeasurementBase takes (arclength, time, ...), in that order.
        Eigen::Matrix4d measurement_value = Eigen::Matrix4d::Identity();
        pose_sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(
            /*arclength=*/0.0, /*time=*/0.05, measurement_value, measurement_covariance, Eigen::MatrixXi::Ones(6, 1)));

        auto factors = pose_sensor->getNewFactors(0.05, new_values);
        ASSERT_EQ(factors.size(), 1u);
        ASSERT_NE(factors.front(), nullptr);
        // Anchored on the single (s=0, t=0) node alone: 3 keys, not the 6 an
        // interpolating factor would span.
        EXPECT_EQ(factors.front()->keys().size(), 3u);
    }

    TEST(PoseSensor, MeasurementsBeforeTheGraphAreDroppedWithoutNullFactors)
    {
        // Extrapolation is forward-only, so a measurement predating the earliest node
        // still cannot be attached. It must be dropped outright: pushing a null factor
        // into the NonlinearFactorGraph segfaults on linearization.
        unsigned int spatial_dim = 4;
        auto state = std::make_shared<StateType>(1.0, spatial_dim);

        ValueType node_reference;
        node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);

        Eigen::Matrix<double, 6, 6> measurement_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
        auto pose_sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);

        gtsam::Values current_values, new_values;
        KeyManager km;
        prior->extendState(km, 1.0, current_values, new_values);
        current_values.insert(new_values);
        new_values.clear();

        // NOTE: MeasurementBase takes (arclength, time, ...), in that order.
        Eigen::Matrix4d measurement_value = Eigen::Matrix4d::Identity();
        pose_sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(
            /*arclength=*/0.0, /*time=*/0.5, measurement_value, measurement_covariance, Eigen::MatrixXi::Ones(6, 1)));

        auto factors = pose_sensor->getNewFactors(1.0, new_values);
        EXPECT_TRUE(factors.empty());
        for (const auto &factor : factors)
            EXPECT_NE(factor, nullptr);

        // The measurement must also be consumed, not left to be retried forever.
        EXPECT_TRUE(pose_sensor->empty());
    }


    TEST(PoseSensor, MeasurementBetweenNodesInfluencesTheSolve)
    {
        // End to end through the estimator: a measurement timestamped BETWEEN two estimate()
        // ticks must not merely produce a factor, it must pull the solution. It is drained on
        // the later tick, whose node (added at exactly that tick's time) brackets it, so it
        // is attached by interpolation.
        //
        // Runs the same scenario twice -- once with the measurement, once with none -- and
        // requires the tip pose to differ.
        auto solve = [](bool add_measurement)
        {
            unsigned int spatial_dim = 4;
            auto state = std::make_shared<StateType>(1.0, spatial_dim);

            ValueType node_reference;
            node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
            auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);

            Estimator estimator(Testing::estimatorOptions(1.0));
            estimator.addPrior(prior);

            Eigen::Matrix<double, 6, 6> measurement_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
            auto pose_sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
            estimator.addSensor(pose_sensor);

            estimator.estimate(0.0);
            estimator.estimate(0.1);

            if (add_measurement)
            {
                // t = 0.12 falls between the 0.1 and 0.2 ticks. It is not yet due at 0.1, and
                // on the 0.2 tick both bordering nodes exist.
                // NOTE: MeasurementBase takes (arclength, time, ...), in that order.
                Eigen::Matrix4d measurement_value = Eigen::Matrix4d::Identity();
                measurement_value(0, 3) = 0.25; // a clear lateral offset at the robot tip
                pose_sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(
                    /*arclength=*/1.0, /*time=*/0.12, measurement_value, measurement_covariance, Eigen::MatrixXi::Ones(6, 1)));
            }

            estimator.estimate(0.2);

            gtsam::Values values = estimator.getCurrentValues();
            auto tip = state->estimation_nodes.back();
            return tip->getValues(values).pose();
        };

        const Pose without = solve(false);
        const Pose with = solve(true);

        EXPECT_FALSE(with.equals(without, 1e-6))
            << "a measurement between nodes left the solution unchanged";
        // And it should move the tip toward the measurement, not somewhere arbitrary.
        EXPECT_GT(with.translation().x(), without.translation().x());
    }


}