#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using Sensor = Sensors::GyroSensor<ValueType>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior(std::shared_ptr<StateType> &state, unsigned int spatial_dim = 4)
        {
            state = std::make_shared<StateType>(1.0, spatial_dim);
            ValueType node_reference;
            node_reference.epsilon((Vector6() << 0, 0, 0, 0, 0, 1.0).finished());
            return std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);
        }

        std::shared_ptr<Types::MeasurementBase> makeGyroMeasurement(double arclength, double time, const Eigen::Vector3d &omega,
                                                                    const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(3, 1))
        {
            return std::make_shared<Types::MeasurementBase>(arclength, time, omega, Eigen::Matrix3d::Identity() * 1e-2, mask);
        }

        Sensor::Factor makeFactor(const Eigen::Vector3d &omega, const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(3, 1))
        {
            return Sensor::Factor(gtsam::noiseModel::Unit::Create(3), omega, mask);
        }

        std::string writeFile(const std::filesystem::path &path, const std::string &contents)
        {
            std::ofstream(path) << contents;
            return path.string();
        }
    } // namespace

    TEST(GyroSensor, ErrorIsMeasuredMinusAngularVelocity)
    {
        // GTSAM varpi layout is [rotational, translational]; only the rotational
        // block is observed, so the translational velocity must not enter the error.
        const Eigen::Vector3d omega(0.3, -0.2, 0.5);
        const Vector6 varpi = (Vector6() << 0.1, 0.4, -0.3, 7.0, -8.0, 9.0).finished();
        const gtsam::Vector e = makeFactor(omega).evaluateError(varpi);
        ASSERT_EQ(e.size(), 3);
        EXPECT_TRUE(e.isApprox(Eigen::Vector3d(0.2, -0.6, 0.8), 1e-12));
    }

    TEST(GyroSensor, JacobianMatchesNumerical)
    {
        const auto factor = makeFactor(Eigen::Vector3d(0.3, -0.2, 0.5));
        gtsam::Values values;
        values.insert(ValueType::InterpolatedKeys::Velocity(), Vector6((Vector6() << 0.1, 0.4, -0.3, 0.2, -0.5, 0.6).finished()));
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("GyroFactor", factor, values, 1e-6, 1e-8));

        gtsam::Matrix H;
        factor.evaluateError(values.at<Vector6>(ValueType::InterpolatedKeys::Velocity()), &H);
        Eigen::Matrix<double, 3, 6> expected = Eigen::Matrix<double, 3, 6>::Zero();
        expected.leftCols<3>() = -Eigen::Matrix3d::Identity();
        EXPECT_TRUE(H.isApprox(expected));
    }

    TEST(GyroSensor, MaskZeroesErrorAndJacobianRows)
    {
        Eigen::MatrixXi mask = Eigen::MatrixXi::Ones(3, 1);
        mask(1, 0) = 0;
        const auto factor = makeFactor(Eigen::Vector3d(0.3, -0.2, 0.5), mask);
        const Vector6 varpi = (Vector6() << 0.1, 0.4, -0.3, 0.2, -0.5, 0.6).finished();

        gtsam::Matrix H;
        const gtsam::Vector e = factor.evaluateError(varpi, &H);
        EXPECT_DOUBLE_EQ(e(1), 0.0);
        EXPECT_NE(e(0), 0.0);
        EXPECT_NE(e(2), 0.0);
        EXPECT_TRUE(H.row(1).isZero());
        EXPECT_FALSE(H.row(0).isZero());
        EXPECT_FALSE(H.row(2).isZero());
    }

    TEST(GyroSensor, InterpolatedFactorJacobiansMatchNumerical)
    {
        // Two time rows, measurement off both the spatial and temporal grid, so the
        // velocity row range of every corner's interpolation Jacobian is exercised.
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        auto sensor = std::make_shared<Sensor>(prior);

        KeyManager km;
        gtsam::Values values, new_values;
        for (double t : {0.0, 0.1})
        {
            prior->extendState(km, t, values, new_values);
            values.insert(new_values);
            new_values.clear();
        }
        // Perturb every node's velocity so the interpolated angular velocity is nontrivial.
        int i = 0;
        for (const gtsam::Key key : values.keys())
            if (gtsam::Symbol(key).chr() == 'v')
                values.update(key, Vector6((Vector6() << 0.1, -0.2, 0.3, 0.05, 0.02, -0.01).finished() * (1.0 + 0.1 * i++)));

        sensor->addMeasurement(makeGyroMeasurement(0.5, 0.05, Eigen::Vector3d(0.3, -0.2, 0.1)));
        auto factors = sensor->getNewFactors(0.1, new_values);
        ASSERT_EQ(factors.size(), 1u);
        auto factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factors.front());
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(factor->keys().size(), 12u); // four corners x (pose, strain, velocity)

        // Rebuild with a fixed noise model: see WNOAInterpolationTests'
        // InterpolationFactorTimeOnlyJacobians for why testFactorJacobians needs it.
        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        std::shared_ptr<NodeType> node00, node01, node10, node11;
        node00 = state->getClosestShiftedNode(0.5, 0.05, -1, -1);
        ASSERT_NE(node00, nullptr);
        node01 = state->getNextTemporalNode(node00);
        node10 = state->getNextSpatialNode(node00);
        node11 = node01 ? state->getNextSpatialNode(node01) : nullptr;
        ASSERT_TRUE(node00 && node01 && node10 && node11);
        auto interpolator = InterpolatorType(Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity(),
                                             Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity());
        Priors::InterpolationFactor<InterpolatorType> fixed_factor(
            sensor->getFactor(makeGyroMeasurement(0.5, 0.05, Eigen::Vector3d(0.3, -0.2, 0.1))),
            interpolator, 0.05, 0.5, node00, node01, node10, node11, true);
        // Tolerance: the gyro factor's own Jacobian is exact (see JacobianMatchesNumerical);
        // the residual (~2e-4, pose-key columns only) is the Interpolator's small-angle
        // cross-term approximation -- see WNOAInterpolationTests'
        // InterpolationFactorTimeOnlyJacobians, which uses 1e-2 for the same reason.
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("GyroInterpolationFactor", fixed_factor, values, 1e-6, 1e-3));
    }

    TEST(GyroSensor, EstimatorTracksConstantAngularVelocity)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        auto sensor = std::make_shared<Sensor>(prior);

        Estimator estimator;
        estimator.addPrior(prior);
        estimator.addSensor(sensor);

        const Eigen::Vector3d omega(0.3, -0.2, 0.1);
        const std::vector<double> arclengths = {0.0, 0.5, 1.0};
        for (int k = 0; k <= 20; k++)
        {
            const double time = 0.1 * k;
            for (double s : arclengths)
            {
                sensor->addMeasurement(makeGyroMeasurement(s, time, omega));
                if (k < 20)
                    sensor->addMeasurement(makeGyroMeasurement(s, time + 0.05, omega));
            }
        }
        for (int k = 0; k <= 20; k++)
            estimator.estimate(0.1 * k);

        for (double s : arclengths)
        {
            auto queried = std::dynamic_pointer_cast<ValueType>(prior->query(2.0, s, estimator.getCurrentValues()));
            ASSERT_NE(queried, nullptr);
            EXPECT_TRUE(queried->varpi().head<3>().isApprox(omega, 5e-2))
                << "s=" << s << " estimated omega " << queried->varpi().head<3>().transpose();
        }
    }

    TEST(GyroSensor, ConfigLoaderLoadsGyroSensorsAndDiagonalCovariances)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "gyro_sensor_demo_data";
        std::filesystem::create_directories(dir);
        writeFile(dir / "gyro_ee.csv",
                  "sensor type:,gyro\nground truth:,0\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0.1,0.2,0.3\n0.05,1.0,0.4,0.5,0.6\n2.0,1.0,0.7,0.8,0.9\n");
        writeFile(dir / "aurora1.csv",
                  "sensor type:,aurora\nground truth:,0\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0,0,1,0,0,0\n");

        auto loadWith = [&](bool include_gyro)
        {
            const std::string config = writeFile(dir / "config.json", std::string(R"({
                "demo_data": {
                    "weights": {
                        "R_pose": [1, 2, 3, 4, 5, 6],
                        "R_gyro": [1e-4, 2e-4, 3e-4],
                        "measurement_factor": 2
                    },
                    "folder_path": ")") + dir.string() + R"(",
                    "start_time": 0,
                    "end_time": 1,
                    "include_aurora": true,
                    "include_gyro": )" + (include_gyro ? "true" : "false") + R"(
                }
            })");
            Utilities::ConfigLoader loader;
            EXPECT_TRUE(loader.loadConfig(config));
            std::shared_ptr<StateType> state;
            auto prior = makePrior(state);
            std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
            loader.loadDemoData(sensors, gt_sensors, prior);
            return sensors;
        };

        const auto with_gyro = loadWith(true);
        ASSERT_EQ(with_gyro.size(), 2u);
        int gyro_count = 0;
        for (const auto &sensor : with_gyro)
        {
            ASSERT_FALSE(sensor->empty());
            const auto measurement = sensor->peekAll().front();
            if (std::dynamic_pointer_cast<Sensor>(sensor))
            {
                gyro_count++;
                EXPECT_EQ(sensor->size(), 2u); // t=2.0 is past end_time
                const Eigen::Matrix3d expected = 2.0 * Eigen::Vector3d(1e-4, 2e-4, 3e-4).asDiagonal().toDenseMatrix();
                EXPECT_TRUE(measurement->covariance.isApprox(expected));
                EXPECT_TRUE(measurement->value.isApprox(Eigen::Vector3d(0.1, 0.2, 0.3)));
                EXPECT_DOUBLE_EQ(measurement->arclength, 1.0);
            }
            else
            {
                ASSERT_NE(std::dynamic_pointer_cast<Sensors::PoseSensor<ValueType>>(sensor), nullptr);
                const Eigen::Matrix<double, 6, 6> expected = 2.0 * (Vector6() << 1, 2, 3, 4, 5, 6).finished().asDiagonal().toDenseMatrix();
                EXPECT_TRUE(measurement->covariance.isApprox(expected));
            }
        }
        EXPECT_EQ(gyro_count, 1);

        const auto without_gyro = loadWith(false);
        ASSERT_EQ(without_gyro.size(), 1u);
        EXPECT_EQ(std::dynamic_pointer_cast<Sensor>(without_gyro.front()), nullptr);

        std::filesystem::remove_all(dir);
    }
} // namespace Spacetime
