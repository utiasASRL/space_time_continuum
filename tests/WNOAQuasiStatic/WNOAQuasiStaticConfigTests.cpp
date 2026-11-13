#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using PriorType = Prior<Pose>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Vector12 = Eigen::Matrix<double, 12, 1>;

        Utilities::ConfigLoader loaderFor(const std::string &name, const std::string &contents)
        {
            const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
            std::ofstream(path) << contents;
            Utilities::ConfigLoader loader;
            EXPECT_TRUE(loader.loadConfig(path.string()));
            std::filesystem::remove(path);
            return loader;
        }

        // The quasi-static prior reads the WNOA prior's own section, so that one config
        // serves both estimators with the same spatial noise.
        const std::string kConfig = R"({
            "prior": {
                "type": "wnoa_quasi_static",
                "time_period": 0.05,
                "wnoa": {
                    "factor_type": "paper",
                    "fixed_noise_model": true,
                    "cache_noise_model": false,
                    "weights": {
                        "prior_factor": 2.0,
                        "Q0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                        "Q1": [1, 1, 1, 1, 1, 1],
                        "Q2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        "Q3": [1, 1, 1, 1, 1, 1],
                        "Q4": [1, 1, 1, 1, 1, 1]
                    },
                    "initial_condition": {
                        "T0": [0, 0, -1, 0,  0, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 1],
                        "epsilon0": [0, 0, 0, 0, 0, 1],
                        "varpi0": [9, 9, 9, 9, 9, 9]
                    }
                }
            }
        })";
    }

    TEST(WNOAQuasiStaticConfig, PriorTypeIsAccepted)
    {
        EXPECT_EQ(loaderFor("qs_type.json", kConfig).loadPriorType(), "wnoa_quasi_static");
        EXPECT_THROW(loaderFor("qs_bad_type.json", R"({ "prior": { "type": "wnoa_quasistatic" } })").loadPriorType(), std::invalid_argument);
    }

    TEST(WNOAQuasiStaticConfig, ReadsThePoseAndStrainPartsOfTheWNOASection)
    {
        const auto loader = loaderFor("qs_prior.json", kConfig);
        Values<Pose> base;
        double time_period = -1.0;
        PriorType::Matrix2N Q0 = PriorType::Matrix2N::Identity();
        PriorType::MatrixN Q2 = PriorType::MatrixN::Identity();
        loader.loadWNOAQuasiStaticPrior<Pose>(base, time_period, Q0, Q2);

        EXPECT_DOUBLE_EQ(time_period, 0.05);
        const Vector12 expected_Q0 = 2.0 * (Vector12() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).finished();
        EXPECT_TRUE(Q0.isApprox(PriorType::Matrix2N(expected_Q0.asDiagonal())));
        EXPECT_TRUE(Q2.isApprox(PriorType::MatrixN((2.0 * Vector6(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)).asDiagonal())));

        Eigen::Matrix4d T0;
        T0 << 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1;
        EXPECT_TRUE(base.pose().equals(Pose(T0)));
        EXPECT_TRUE(base.epsilon().isApprox((Vector6() << 0, 0, 0, 0, 0, 1).finished()));
    }

    TEST(WNOAQuasiStaticConfig, ReadsTheNoiseModelOptionsOfTheWNOASection)
    {
        const PriorOptions options = loaderFor("qs_options.json", kConfig).loadWNOAQuasiStaticPriorOptions();
        EXPECT_TRUE(options.fixed_noise_model);
        EXPECT_FALSE(options.cache_noise_model);

        const PriorOptions defaults = loaderFor("qs_no_options.json", R"({ "prior": { "type": "wnoa_quasi_static" } })").loadWNOAQuasiStaticPriorOptions();
        EXPECT_EQ(defaults.fixed_noise_model, PriorOptions().fixed_noise_model);
        EXPECT_EQ(defaults.cache_noise_model, PriorOptions().cache_noise_model);
    }

    TEST(WNOAQuasiStaticConfig, EstimatorQuasiStaticOptionIsRead)
    {
        EXPECT_FALSE(EstimatorOptions().quasi_static);
        EXPECT_TRUE(loaderFor("qs_on.json", R"({ "estimator": { "quasi_static": true } })").loadEstimatorOptions().quasi_static);
        EXPECT_FALSE(loaderFor("qs_off.json", R"({ "estimator": { "smootherLag": -1 } })").loadEstimatorOptions().quasi_static);
    }
} // namespace Spacetime::Priors::WNOAQuasiStatic
