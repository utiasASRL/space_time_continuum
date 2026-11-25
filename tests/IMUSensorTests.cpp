#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/joint_models/EndEffectorPrior.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAPrior.hpp>
#include <spacetime/estimator/sensor_models/IMUSensor.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector3 = gtsam::Vector3;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using WNOAPrior = Priors::WNOA::Prior<Pose>;
        using WNOAState = Types::ContinuumRobotState<Priors::WNOA::Node<Pose>>;
        using WNOAIMU = Sensors::IMUSensor<WNOAPrior>;
        using EEPrior = Joints::EndEffectorPrior<Pose>;
        using EEState = Types::ContinuumRobotState<EEPrior::NodeType>;
        using EEIMU = Sensors::IMUSensor<EEPrior>;

        std::shared_ptr<WNOAPrior> makeWNOAPrior(std::shared_ptr<WNOAState> &state)
        {
            state = std::make_shared<WNOAState>(1.0, 4);
            Priors::WNOA::NodeValues<Pose> reference;
            reference.epsilon((Vector6() << 0, 0, 0, 1.0, 0, 0).finished());
            return std::make_shared<WNOAPrior>(state, reference);
        }

        std::shared_ptr<Types::MeasurementBase> imuReading(double s, double t, const Vector3 &omega, const Vector3 &accel,
                                                           const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(6, 1))
        {
            Eigen::Matrix<double, 6, 1> value;
            value << omega, accel;
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
            covariance.topLeftCorner<3, 3>() = 1e-4 * Eigen::Matrix3d::Identity();
            covariance.bottomRightCorner<3, 3>() = 1e-2 * Eigen::Matrix3d::Identity();
            return std::make_shared<Types::MeasurementBase>(s, t, Eigen::MatrixXd(value), Eigen::MatrixXd(covariance), mask);
        }

        /// One estimation step's phases at `time`, by hand (see AccelerometerSensorTests' step).
        Types::Model::Factors step(Types::Model &prior, Types::Model &sensor, KeyManager &km, gtsam::Values &values, double time)
        {
            gtsam::Values new_values;
            prior.extendState(km, time, values, new_values);
            sensor.extendState(km, time, values, new_values);
            values.insert(new_values);
            new_values.clear();
            return sensor.getNewFactors(time, new_values);
        }
    } // namespace

    TEST(IMUSensor, DefaultsEstimateBothBiasesAndShareTheOffset)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        Sensors::IMUOptions options;
        options.offset = Pose(gtsam::Rot3::Rz(0.3), gtsam::Point3(0.01, 0.02, 0.0));
        options.gyro.offset = Pose(gtsam::Rot3::Rx(1.0), gtsam::Point3::Zero());
        WNOAIMU imu(prior, 0.5, options);

        EXPECT_TRUE(imu.options().gyro.estimate_bias);
        EXPECT_TRUE(imu.options().accelerometer.estimate_bias);
        ASSERT_NE(imu.gyro()->biasModel(), nullptr);
        ASSERT_NE(imu.accelerometer()->biasModel(), nullptr);
        EXPECT_TRUE(imu.gyro()->options().offset.equals(options.offset));
        EXPECT_TRUE(imu.accelerometer()->options().offset.equals(options.offset));
        EXPECT_NEAR(imu.arclength(), 0.5, 1e-15);
        EXPECT_EQ(imu.getFactor(imuReading(0.5, 0.0, Vector3::Zero(), Vector3::Zero())), nullptr);
        EXPECT_THROW(WNOAIMU(nullptr, 0.5), std::invalid_argument);
        EXPECT_THROW(WNOAIMU(prior, 2.0), std::invalid_argument);
    }

    TEST(IMUSensor, SplitsEachMeasurementBetweenTheGyroAndTheAccelerometer)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAIMU imu(prior, 1.0 / 3.0);
        for (int i = 0; i <= 20; i++)
            imu.addMeasurement(imuReading(1.0 / 3.0, 0.01 * i, Vector3(0.1, 0.2, 0.3), Vector3(0.0, 0.0, 9.81)));

        KeyManager km;
        gtsam::Values values;
        // At t = 0: one gyro factor (the first accelerometer interval ends at the next row).
        EXPECT_EQ(step(*prior, imu, km, values, 0.0).size(), 1u);
        // At t = 0.1: ten gyro factors (t = 0.01 .. 0.1) and one accelerometer interval.
        const auto factors = step(*prior, imu, km, values, 0.1);
        EXPECT_EQ(factors.size(), 11u);
        for (const auto &factor : factors)
            for (const gtsam::Key key : factor->keys())
                EXPECT_TRUE(values.exists(key));
        EXPECT_EQ(step(*prior, imu, km, values, 0.2).size(), 11u);
        EXPECT_TRUE(imu.empty());
        EXPECT_TRUE(imu.gyro()->empty());
        EXPECT_TRUE(imu.accelerometer()->empty());

        // Both bias models: three nodes each.
        EXPECT_EQ(imu.covarianceKeyGroups(values).size(), 6u);
    }

    TEST(IMUSensor, MalformedMeasurementsAreDropped)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAIMU imu(prior, 0.5);
        imu.addMeasurement(std::make_shared<Types::MeasurementBase>(0.5, 0.0, Eigen::MatrixXd::Zero(3, 1), Eigen::MatrixXd::Identity(3, 3)));
        KeyManager km;
        gtsam::Values values;
        EXPECT_TRUE(step(*prior, imu, km, values, 0.0).empty());
        EXPECT_TRUE(imu.gyro()->empty());
        EXPECT_TRUE(imu.accelerometer()->empty());
    }

    TEST(IMUSensor, OnlyItsOwnQueueIsRegistered)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        auto imu = std::make_shared<WNOAIMU>(prior, 0.5);
        Estimator estimator(Testing::estimatorOptions(0.5));
        estimator.addPrior(prior);
        estimator.addSensor(imu);
        ASSERT_EQ(estimator.getMeasurementQueues().size(), 1u);
        EXPECT_EQ(estimator.getMeasurementQueues().front(), std::static_pointer_cast<Types::MeasurementQueue>(imu));
    }

    TEST(IMUSensor, EstimatesAStationaryEndEffectorAndItsBiases)
    {
        auto state = std::make_shared<EEState>(std::vector<double>{0.0});
        auto prior = std::make_shared<EEPrior>(state, Pose(), Vector6::Ones());
        auto imu = std::make_shared<EEIMU>(prior, 0.0);
        // An IMU alone leaves the pose unobservable: anchor it at the origin.
        auto pose_sensor = std::make_shared<Sensors::PoseSensor<Joints::EndEffectorNodeValues<Pose>>>(prior);
        Estimator estimator(Testing::estimatorOptions(0.5));
        estimator.addPrior(prior);
        estimator.addSensor(imu);
        estimator.addSensor(pose_sensor);
        for (int k = 0; k <= 10; k++)
            pose_sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(
                0.0, 0.1 * k, Pose().matrix(), Eigen::MatrixXd(Eigen::Matrix<double, 6, 6>::Identity() * 1e-6)));

        // At rest the accelerometer reads -g (specific force), the gyro zero.
        for (int i = 0; i <= 100; i++)
            imu->addMeasurement(imuReading(0.0, 0.01 * i, Vector3::Zero(), Vector3(0.0, 0.0, 9.81)));
        for (int k = 0; k <= 10; k++)
            estimator.estimate(0.1 * k);

        auto queried = std::dynamic_pointer_cast<Joints::EndEffectorNodeValues<Pose>>(prior->query(1.0, 0.0, estimator.getCurrentValues()));
        ASSERT_NE(queried, nullptr);
        EXPECT_LT(queried->varpi().norm(), 1e-2) << queried->varpi().transpose();
        auto gyro_bias = std::dynamic_pointer_cast<Biases::BiasNodeValues<Vector3>>(imu->gyro()->biasModel()->query(1.0, 0.0, estimator.getCurrentValues()));
        ASSERT_NE(gyro_bias, nullptr);
        EXPECT_LT(gyro_bias->bias().norm(), 1e-2);
    }
} // namespace Spacetime
