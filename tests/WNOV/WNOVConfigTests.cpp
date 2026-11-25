#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime::Priors::WNOV
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using PriorType = Prior<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        std::string writeFile(const std::filesystem::path &path, const std::string &contents)
        {
            std::ofstream(path) << contents;
            return path.string();
        }

        Utilities::ConfigLoader loaderFor(const std::string &name, const std::string &contents)
        {
            const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
            Utilities::ConfigLoader loader;
            EXPECT_TRUE(loader.loadConfig(writeFile(path, contents)));
            std::filesystem::remove(path);
            return loader;
        }

        struct LoadedWNOV
        {
            Pose T0;
            Vector6 epsilon0 = Vector6::Zero();
            double time_period = -1.0;
            Matrix6 Q0 = Matrix6::Identity(), Q1 = Matrix6::Identity(), Q2 = Matrix6::Identity();
        };

        LoadedWNOV loadWNOV(const Utilities::ConfigLoader &loader)
        {
            LoadedWNOV out;
            loader.loadWNOVPrior<Pose>(out.T0, out.epsilon0, out.time_period, out.Q0, out.Q1, out.Q2);
            return out;
        }
    }

    TEST(WNOVConfig, PriorTypeIsReadFromThePriorSection)
    {
        EXPECT_EQ(loaderFor("type_wnov.json", R"({ "prior": { "type": "wnov" } })").loadPriorType(), "wnov");
        EXPECT_EQ(loaderFor("type_wnoa.json", R"({ "prior": { "type": "wnoa" } })").loadPriorType(), "wnoa");
    }

    TEST(WNOVConfig, PriorTypeDefaultsToWNOA)
    {
        EXPECT_EQ(loaderFor("type_missing.json", R"({ "estimator": {} })").loadPriorType(), "wnoa");
        EXPECT_EQ(loaderFor("type_missing_type.json", R"({ "prior": { "wnov": {} } })").loadPriorType(), "wnoa");
        // A legacy top-level "WNOA" section is warned about, not read.
        EXPECT_EQ(loaderFor("type_legacy.json", R"({ "WNOA": { "time_period": 0.1 } })").loadPriorType(), "wnoa");
    }

    TEST(WNOVConfig, LoadsTheWNOVSection)
    {
        const auto loader = loaderFor("wnov_full.json", R"({ "prior": {
            "type": "wnov",
            "time_period": 0.05,
            "wnov": {
                "fixed_noise_model": true,
                "remove_unary": true,
                "weights": {
                    "prior_factor": 2,
                    "Q0": [1, 2, 3, 4, 5, 6],
                    "Q1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "Q2": [6, 5, 4, 3, 2, 1]
                },
                "initial_condition": {
                    "T0": [1, 0, 0, 0.1,  0, 1, 0, 0.2,  0, 0, 1, 0.3,  0, 0, 0, 1],
                    "epsilon0": [0, 0, 0, 0, 0, 1]
                }
            }
        } })");
        const LoadedWNOV loaded = loadWNOV(loader);
        EXPECT_DOUBLE_EQ(loaded.time_period, 0.05);
        EXPECT_TRUE(loaded.Q0.isApprox(2.0 * Vector6(1, 2, 3, 4, 5, 6).asDiagonal().toDenseMatrix()));
        EXPECT_TRUE(loaded.Q1.isApprox(2.0 * Vector6(0.1, 0.2, 0.3, 0.4, 0.5, 0.6).asDiagonal().toDenseMatrix()));
        EXPECT_TRUE(loaded.Q2.isApprox(2.0 * Vector6(6, 5, 4, 3, 2, 1).asDiagonal().toDenseMatrix()));
        EXPECT_TRUE(loaded.T0.equals(Pose(gtsam::Rot3(), gtsam::Point3(0.1, 0.2, 0.3)))) << "T0 is row-major";
        EXPECT_TRUE(loaded.epsilon0.isApprox(Vector6(0, 0, 0, 0, 0, 1)));
        EXPECT_TRUE(loader.loadWNOVPriorOptions().fixed_noise_model);
        EXPECT_TRUE(loader.loadWNOVPriorOptions().remove_unary);
    }

    TEST(WNOVConfig, MissingEntriesKeepTheirValues)
    {
        const auto loader = loaderFor("wnov_partial.json", R"({ "prior": { "type": "wnov", "wnov": { "weights": { "Q1": [1, 1, 1, 1, 1, 1] } } } })");
        const LoadedWNOV loaded = loadWNOV(loader);
        EXPECT_DOUBLE_EQ(loaded.time_period, -1.0);
        EXPECT_TRUE(loaded.Q0.isIdentity());
        EXPECT_TRUE(loaded.Q2.isIdentity());
        EXPECT_TRUE(loaded.T0.equals(Pose()));
        EXPECT_TRUE(loaded.epsilon0.isZero());
        EXPECT_FALSE(loader.loadWNOVPriorOptions().fixed_noise_model);
        EXPECT_FALSE(loader.loadWNOVPriorOptions().remove_unary);
    }

    TEST(WNOVConfig, WrongSizedEntriesThrow)
    {
        for (const std::string &entry : {R"("weights": { "Q0": [1, 2, 3, 4, 5] })",
                                         R"("weights": { "Q2": [1, 2, 3, 4, 5, 6, 7] })",
                                         R"("weights": { "Q1": 3 })",
                                         R"("initial_condition": { "T0": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] })",
                                         R"("initial_condition": { "epsilon0": [0, 0, 1] })"})
        {
            const auto loader = loaderFor("wnov_bad.json", R"({ "prior": { "type": "wnov", "wnov": { )" + entry + " } } }");
            EXPECT_THROW(loadWNOV(loader), std::invalid_argument) << entry;
        }
    }

    TEST(WNOVConfig, WNOAIsReadFromItsOwnSubsection)
    {
        const auto loader = loaderFor("wnoa_nested.json", R"({ "prior": {
            "type": "wnov",
            "time_period": 0.02,
            "wnoa": {
                "factor_type": "legacy",
                "weights": { "Q1": [1, 2, 3, 4, 5, 6], "Q2": [1, 1, 1, 1, 1, 1], "Q3": [1, 1, 1, 1, 1, 1],
                             "Q0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] }
            },
            "wnov": { "weights": { "Q1": [9, 9, 9, 9, 9, 9] } }
        } })");
        WNOA::NodeValues<Pose> reference;
        double time_period = -1.0;
        WNOA::Prior<Pose>::Matrix3N Q0;
        WNOA::Prior<Pose>::MatrixN Q1, Q2, Q3, Q4;
        loader.loadWNOAPrior(reference, time_period, Q0, Q1, Q2, Q3, Q4);
        EXPECT_DOUBLE_EQ(time_period, 0.02);
        EXPECT_TRUE(Q1.isApprox(Vector6(1, 2, 3, 4, 5, 6).asDiagonal().toDenseMatrix()));
        EXPECT_EQ(loader.loadWNOAPriorOptions().factor_type, WNOA::FactorType::Legacy);

        // Each model reads only its own subsection.
        EXPECT_TRUE(loadWNOV(loader).Q1.isApprox(9.0 * Matrix6::Identity()));
    }

    TEST(WNOVConfig, MissingWNOAEntriesKeepThePassedInValues)
    {
        const auto loader = loaderFor("wnoa_partial.json", R"({ "prior": {
            "time_period": 0.02,
            "wnoa": {
                "weights": { "prior_factor": 2.0, "Q2": [1, 1, 1, 1, 1, 1] },
                "initial_condition": { "epsilon0": [0, 0, 0, 1, 0, 0] }
            }
        } })");
        WNOA::NodeValues<Pose> reference;
        reference.pose(Pose(gtsam::Rot3::Rz(0.3), gtsam::Point3(0.1, 0.2, 0.3)));
        reference.varpi(Vector6(1, 2, 3, 4, 5, 6));
        double time_period = -1.0;
        WNOA::Prior<Pose>::Matrix3N Q0 = 3.0 * WNOA::Prior<Pose>::Matrix3N::Identity();
        WNOA::Prior<Pose>::MatrixN Q1 = 5.0 * Matrix6::Identity(), Q2 = Q1, Q3 = 7.0 * Matrix6::Identity(), Q4 = Q1;
        loader.loadWNOAPrior(reference, time_period, Q0, Q1, Q2, Q3, Q4);

        EXPECT_TRUE(Q0.isApprox(3.0 * WNOA::Prior<Pose>::Matrix3N::Identity()));
        EXPECT_TRUE(Q1.isApprox(5.0 * Matrix6::Identity()));
        EXPECT_TRUE(Q2.isApprox(2.0 * Matrix6::Identity())); // read, times prior_factor
        EXPECT_TRUE(Q3.isApprox(7.0 * Matrix6::Identity()));
        EXPECT_TRUE(Q4.isApprox(Q3)); // a missing Q4 takes Q3's
        EXPECT_TRUE(reference.pose().equals(Pose(gtsam::Rot3::Rz(0.3), gtsam::Point3(0.1, 0.2, 0.3))));
        EXPECT_TRUE(reference.epsilon().isApprox(Vector6(0, 0, 0, 1, 0, 0)));
        EXPECT_TRUE(reference.varpi().isApprox(Vector6(1, 2, 3, 4, 5, 6)));
    }

    TEST(WNOVConfig, GyroDataIsSkippedForAPoseOnlyPrior)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "wnov_demo_data";
        std::filesystem::create_directories(dir);
        writeFile(dir / "gyro_ee.csv",
                  "sensor type:,gyro\nground truth:,0\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0.1,0.2,0.3\n");
        writeFile(dir / "aurora1.csv",
                  "sensor type:,aurora\nground truth:,0\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0,0,1,0,0,0\n0.5,1.0,0,0,1,0,0,0\n");

        auto loader = loaderFor("wnov_demo.json", std::string(R"({ "demo_data": {
            "folder_path": ")") + dir.string() + R"(",
            "start_time": 0, "end_time": 1, "include_aurora": true, "include_gyro": true } })");

        auto state = std::make_shared<StateType>(1.0, 3);
        auto prior = std::make_shared<PriorType>(state, Pose(), Vector6::Zero());
        std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
        loader.loadDemoData(sensors, gt_sensors, prior);

        ASSERT_EQ(sensors.size(), 1u); // the aurora only
        EXPECT_NE(std::dynamic_pointer_cast<Sensors::PoseSensor<NodeValues<Pose>>>(sensors.front()), nullptr);
        EXPECT_EQ(sensors.front()->size(), 2u);
        EXPECT_TRUE(gt_sensors.empty());
        std::filesystem::remove_all(dir);
    }
} // namespace Spacetime::Priors::WNOV
