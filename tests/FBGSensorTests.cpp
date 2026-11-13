#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/sensor_models/FBGSensor.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using Sensor = Sensors::FBGSensor<ValueType>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior(std::shared_ptr<StateType> &state, unsigned int spatial_dim = 4)
        {
            state = std::make_shared<StateType>(1.0, spatial_dim);
            ValueType node_reference;
            node_reference.epsilon((Vector6() << 0, 0, 0, 1.0, 0, 0).finished());
            return std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);
        }

        std::shared_ptr<Types::MeasurementBase> makeStrainMeasurement(double arclength, double time, const Vector6 &epsilon,
                                                                      const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(6, 1))
        {
            return std::make_shared<Types::MeasurementBase>(arclength, time, epsilon, Eigen::Matrix<double, 6, 6>::Identity() * 1e-4, mask);
        }

        Sensor::Factor makeFactor(const Vector6 &epsilon, const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(6, 1))
        {
            return Sensor::Factor(gtsam::noiseModel::Unit::Create(6), epsilon, mask);
        }

        std::string writeFile(const std::filesystem::path &path, const std::string &contents)
        {
            std::ofstream(path) << contents;
            return path.string();
        }

        const Vector6 kMeasured = (Vector6() << 0.0, -0.4, 0.02, 1.0, 0.0, 0.0).finished(); // [kappa; nu]
    } // namespace

    TEST(FBGSensor, ErrorIsStrainMinusMeasured)
    {
        const Vector6 epsilon = (Vector6() << 0.1, -0.3, 0.0, 1.1, 0.05, -0.02).finished();
        const gtsam::Vector e = makeFactor(kMeasured).evaluateError(epsilon);
        ASSERT_EQ(e.size(), 6);
        EXPECT_TRUE(e.isApprox(epsilon - kMeasured, 1e-12));
        EXPECT_NEAR(makeFactor(kMeasured).evaluateError(kMeasured).norm(), 0.0, 1e-12);
    }

    TEST(FBGSensor, JacobianMatchesNumerical)
    {
        const auto factor = makeFactor(kMeasured);
        gtsam::Values values;
        values.insert(ValueType::InterpolatedKeys::Strain(), Vector6((Vector6() << 0.1, -0.3, 0.0, 1.1, 0.05, -0.02).finished()));
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("FBGFactor", factor, values, 1e-6, 1e-8));

        gtsam::Matrix H;
        factor.evaluateError(values.at<Vector6>(ValueType::InterpolatedKeys::Strain()), &H);
        EXPECT_TRUE(H.isApprox(Eigen::Matrix<double, 6, 6>::Identity()));
    }

    TEST(FBGSensor, MaskZeroesErrorAndJacobianRows)
    {
        Eigen::MatrixXi mask = Eigen::MatrixXi::Ones(6, 1);
        mask(0, 0) = 0; // e.g. torsion, which an FBG does not observe
        const auto factor = makeFactor(kMeasured, mask);
        gtsam::Matrix H;
        const gtsam::Vector e = factor.evaluateError((Vector6() << 0.3, 0.1, 0.2, 1.2, 0.1, 0.1).finished(), &H);
        EXPECT_DOUBLE_EQ(e(0), 0.0);
        EXPECT_TRUE(H.row(0).isZero());
        for (int i = 1; i < 6; i++)
        {
            EXPECT_NE(e(i), 0.0) << "row " << i;
            EXPECT_FALSE(H.row(i).isZero()) << "row " << i;
        }
    }

    TEST(FBGSensor, FactorReferencesOnlyTheInterpolatedStrainKey)
    {
        std::shared_ptr<StateType> state;
        auto sensor = std::make_shared<Sensor>(makePrior(state));
        auto inner = sensor->getFactor(makeStrainMeasurement(0.5, 0.0, kMeasured));
        ASSERT_NE(inner, nullptr);
        ASSERT_EQ(inner->keys().size(), 1u);
        EXPECT_EQ(inner->keys().front(), ValueType::InterpolatedKeys::Strain());
        EXPECT_EQ(inner->dim(), 6u);
    }

    TEST(FBGSensor, InterpolatedFactorJacobiansMatchNumerical)
    {
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
        // Perturb every node's strain about the nominal one, so the interpolated strain is nontrivial.
        int i = 0;
        for (const gtsam::Key key : values.keys())
            if (gtsam::Symbol(key).chr() == 's')
                values.update(key, Vector6(values.at<Vector6>(key) + (Vector6() << 0.05, -0.04, 0.03, 0.02, -0.01, 0.01).finished() * (1.0 + 0.1 * i++)));

        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        auto node00 = state->getClosestShiftedNode(0.4, 0.05, -1, -1);
        ASSERT_NE(node00, nullptr);
        auto node01 = state->getNextTemporalNode(node00);
        auto node10 = state->getNextSpatialNode(node00);
        auto node11 = node01 ? state->getNextSpatialNode(node01) : nullptr;
        ASSERT_TRUE(node00 && node01 && node10 && node11);
        auto interpolator = InterpolatorType(Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity(),
                                             Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity());
        Priors::InterpolationFactor<InterpolatorType> fixed_factor(sensor->getFactor(makeStrainMeasurement(0.4, 0.05, kMeasured)),
                                                                   interpolator, 0.05, 0.4, node00, node01, node10, node11, true);
        // The interpolator's small-angle approximation leaves an error in the pose-key
        // columns (see GyroSensorTests). Strain is a spatial derivative, so those columns
        // scale with 1 / ds^2 (here ~144), and the error is checked relative to each block.
        auto analytic = std::dynamic_pointer_cast<gtsam::JacobianFactor>(fixed_factor.linearize(values));
        ASSERT_NE(analytic, nullptr);
        const gtsam::JacobianFactor numerical = gtsam::linearizeNumerically(fixed_factor, values, 1e-6);
        ASSERT_EQ(analytic->size(), numerical.size());
        for (size_t k = 0; k < analytic->size(); k++)
        {
            ASSERT_EQ(analytic->keys()[k], numerical.keys()[k]);
            const gtsam::Matrix A = analytic->getA(analytic->begin() + k);
            const gtsam::Matrix N = numerical.getA(numerical.begin() + k);
            EXPECT_LT((A - N).cwiseAbs().maxCoeff(), 1e-3 * std::max(1.0, N.cwiseAbs().maxCoeff()))
                << "key " << gtsam::DefaultKeyFormatter(analytic->keys()[k]);
        }
        EXPECT_TRUE(analytic->getb().isApprox(numerical.getb(), 1e-9));
    }

    TEST(FBGSensor, EstimatorTracksAConstantCurvature)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        auto sensor = std::make_shared<Sensor>(prior);

        Estimator estimator;
        estimator.addPrior(prior);
        estimator.addSensor(sensor);

        // A constant bend about the body z-axis, measured every 10 cm along the robot.
        const Vector6 epsilon = (Vector6() << 0.0, 0.0, 0.8, 1.0, 0.0, 0.0).finished();
        for (int k = 0; k <= 10; k++)
            for (int j = 0; j <= 10; j++)
                sensor->addMeasurement(makeStrainMeasurement(0.1 * j, 0.1 * k, epsilon));
        for (int k = 0; k <= 10; k++)
            estimator.estimate(0.1 * k);

        for (double s : {0.0, 0.5, 1.0})
        {
            auto queried = std::dynamic_pointer_cast<ValueType>(prior->query(1.0, s, estimator.getCurrentValues()));
            ASSERT_NE(queried, nullptr);
            EXPECT_TRUE(queried->epsilon().isApprox(epsilon, 5e-2)) << "s=" << s << " estimated strain " << queried->epsilon().transpose();
        }
        // The bend shows in the shape: the tip has turned by kappa * L about z.
        auto tip = std::dynamic_pointer_cast<ValueType>(prior->query(1.0, 1.0, estimator.getCurrentValues()));
        auto base = std::dynamic_pointer_cast<ValueType>(prior->query(1.0, 0.0, estimator.getCurrentValues()));
        const gtsam::Vector3 turn = gtsam::Rot3::Logmap(base->pose().rotation().between(tip->pose().rotation()));
        EXPECT_NEAR(turn.z(), 0.8, 5e-2);
    }

    TEST(FBGSensor, ConfigLoaderReordersStrainsAndSkipsBadRows)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "fbg_sensor_demo_data";
        std::filesystem::create_directories(dir);
        // As engine4's robot_fbgs_data.csv: a 2-line header, [nu; kappa] strains.
        writeFile(dir / "robot_fbgs_data.csv",
                  "sensor type:,fbgs\ntimestamp (s),arclength (m),curvature (1/m),bending_angle (rad)\n"
                  "0.0,0.0,1.0,0.0,0.0,0.0,-0.4,0.02\n0.0,0.03,1.0,0.0,0.0,0.0\n0.1,0.03,1.0,0.0,0.0,0.1,-0.3,0.01\n2.0,0.0,1,0,0,0,0,0\n");
        auto load = [&](bool include_strain)
        {
            const std::string config = writeFile(dir / "config.json", std::string(R"({
                "demo_data": {
                    "weights": { "R_strain": [1, 2, 3, 4, 5, 6], "measurement_factor": 2 },
                    "folder_path": ")") + dir.string() + R"(",
                    "start_time": 0, "end_time": 1,
                    "include_strain": )" + (include_strain ? "true" : "false") + R"(
                }
            })");
            Utilities::ConfigLoader loader;
            EXPECT_TRUE(loader.loadConfig(config));
            std::shared_ptr<StateType> state;
            std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
            loader.loadDemoData(sensors, gt_sensors, makePrior(state));
            return sensors;
        };

        const auto sensors = load(true);
        ASSERT_EQ(sensors.size(), 1u);
        ASSERT_NE(std::dynamic_pointer_cast<Sensor>(sensors.front()), nullptr);
        const auto queue = sensors.front()->peekAll();
        ASSERT_EQ(queue.size(), 2u); // the short row is skipped, t = 2.0 is past end_time
        const auto first = queue.front();
        EXPECT_DOUBLE_EQ(first->arclength, 0.0);
        EXPECT_TRUE(first->value.isApprox(kMeasured)); // reordered to [kappa; nu]
        EXPECT_TRUE(first->covariance.isApprox(2.0 * (Vector6() << 1, 2, 3, 4, 5, 6).finished().asDiagonal().toDenseMatrix()));
        EXPECT_DOUBLE_EQ(queue[1]->time, 0.1);

        EXPECT_TRUE(load(false).empty());
        std::filesystem::remove_all(dir);
    }

    TEST(FBGSensor, ConfigLoaderSkipsFBGFilesWithoutAStrainState)
    {
        using WNOVPrior = Priors::WNOV::Prior<Pose>;
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "fbg_sensor_wnov";
        std::filesystem::create_directories(dir);
        writeFile(dir / "robot_fbgs_data.csv", "sensor type:,fbgs\ntimestamp (s),arclength (m),data\n0.0,0.0,1,0,0,0,0,0\n");
        const std::string config = writeFile(dir / "config.json", std::string(R"({
            "demo_data": { "folder_path": ")") + dir.string() + R"(", "start_time": 0, "end_time": 1, "include_strain": true }
        })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(config));

        auto state = std::make_shared<Types::ContinuumRobotState<WNOVPrior::NodeType>>(1.0, 3);
        auto prior = std::make_shared<WNOVPrior>(state,
                                                 Pose(), WNOVPrior::TangentVector::Zero());
        std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
        loader.loadDemoData(sensors, gt_sensors, prior);
        EXPECT_TRUE(sensors.empty());
        std::filesystem::remove_all(dir);
    }
} // namespace Spacetime
