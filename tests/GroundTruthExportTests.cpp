#include <filesystem>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/utilities/GroundTruthExport.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include "TestEstimatorOptions.hpp"

namespace Spacetime::Utilities
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::NodeValues<Pose>;

        std::vector<std::vector<double>> readRows(const std::filesystem::path &path, std::string &header)
        {
            std::ifstream file(path);
            std::getline(file, header);
            std::vector<std::vector<double>> rows;
            for (std::string line; std::getline(file, line);)
            {
                std::vector<double> row;
                std::stringstream stream(line);
                for (std::string cell; std::getline(stream, cell, ',');)
                    if (!cell.empty())
                        row.push_back(std::stod(cell));
                rows.push_back(row);
            }
            return rows;
        }

        /// A straight 1 m robot along z (strain [0,0,0, 0,0,1] in GTSAM order), solved in batch
        /// with covariances, plus a ground-truth sensor at the given (s, t) samples.
        struct ExportFixture
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 3);
            std::shared_ptr<Priors::WNOA::Prior<Pose>> prior;
            std::shared_ptr<Sensors::PoseSensor<ValueType>> ground_truth;
            Estimator estimator;
            // One folder per test: ctest -j runs each test in its own process, concurrently.
            std::filesystem::path logs_dir = std::filesystem::path(testing::TempDir()) /
                                             (std::string("ground_truth_export_logs_") + testing::UnitTest::GetInstance()->current_test_info()->name());

            static EstimatorOptions options()
            {
                EstimatorOptions o = Testing::estimatorOptions(Estimator::BATCH_LAG);
                o.compute_covariance = true;
                return o;
            }

            explicit ExportFixture(const std::vector<std::pair<double, double>> &samples) : estimator(options())
            {
                ValueType reference;
                reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, reference);
                estimator.addPrior(prior);
                const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
                ground_truth = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
                for (const auto &[s, t] : samples)
                    ground_truth->addMeasurement(std::make_shared<Types::MeasurementBase>(s, t, Eigen::Matrix4d::Identity(), R, Eigen::MatrixXi::Ones(6, 1)));
                for (double t : {0.0, 0.1, 0.2})
                    estimator.estimate(t);
                estimator.finalize();
                std::filesystem::remove_all(logs_dir);
            }

            std::size_t exportAll(bool with_covariance = true)
            {
                return exportGroundTruthPredictions(*prior, estimator.getCurrentValues(),
                                                          with_covariance ? estimator.getCurrentCovariance().get() : nullptr,
                                                          {ground_truth}, logs_dir.string());
            }
        };
    } // namespace

    TEST(GroundTruthExport, TangentOrderSwapExchangesHalvesAndIsAnInvolution)
    {
        const Eigen::Matrix<double, 6, 1> gtsam_order = (Eigen::Matrix<double, 6, 1>() << 1, 2, 3, 4, 5, 6).finished();
        const Eigen::Matrix<double, 6, 1> legacy_order = (Eigen::Matrix<double, 6, 1>() << 4, 5, 6, 1, 2, 3).finished();
        EXPECT_TRUE((tangentOrderSwap<1>() * gtsam_order).isApprox(legacy_order));
        EXPECT_TRUE((tangentOrderSwap<3>() * tangentOrderSwap<3>()).isIdentity());
    }

    TEST(GroundTruthExport, WritesTheLegacyFormat)
    {
        ExportFixture f({{0.5, 0.1}, {1.0, 0.2}});
        EXPECT_EQ(f.exportAll(), 2u);

        std::string header;
        const auto rows = readRows(f.logs_dir / "error_pred.txt", header);
        EXPECT_EQ(header, "time,arclength,x1,x2,x3,x4,x5,x6,e1,e2,e3,e4,e5,e6,v1,v2,v3,v4,v5,v6,");
        ASSERT_EQ(rows.size(), 2u);
        ASSERT_EQ(rows[0].size(), 20u);

        // Straight along z: Log(T) = [rho, phi] = [0,0,s, 0,0,0], strain [0,0,1, 0,0,0] (legacy order).
        EXPECT_DOUBLE_EQ(rows[0][0], 0.1);
        EXPECT_DOUBLE_EQ(rows[0][1], 0.5);
        const std::vector<double> expected_mean = {0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 18; i++)
            EXPECT_NEAR(rows[0][2 + i], expected_mean[i], 1e-9) << "column " << i;
        EXPECT_NEAR(rows[1][4], 1.0, 1e-9); // tip: rho_z = 1
    }

    TEST(GroundTruthExport, CovarianceIsReorderedToTheLegacyOrder)
    {
        // A sample on a node: the queried posterior is that node's marginal.
        ExportFixture f({{0.0, 0.1}});
        ASSERT_EQ(f.exportAll(), 1u);

        std::string header;
        const auto rows = readRows(f.logs_dir / "error_pred_cov.txt", header);
        EXPECT_EQ(header, "time,arclength,");
        ASSERT_EQ(rows.size(), 1u);
        ASSERT_EQ(rows[0].size(), 2u + 18u * 18u);
        Eigen::Matrix<double, 18, 18> legacy;
        for (int r = 0; r < 18; r++)
            for (int c = 0; c < 18; c++)
                legacy(r, c) = rows[0][2 + 18 * r + c];

        auto node = std::dynamic_pointer_cast<NodeType>(f.state->estimation_nodes.at(3)); // row 1 (t = 0.1), s = 0
        const gtsam::Matrix pose_cov = f.estimator.covariance(node->poseKey());
        const gtsam::Matrix strain_cov = f.estimator.covariance(node->strainKey());
        EXPECT_TRUE((legacy.block<3, 3>(0, 0).isApprox(pose_cov.block<3, 3>(3, 3), 1e-6))); // translation first
        EXPECT_TRUE((legacy.block<3, 3>(3, 3).isApprox(pose_cov.block<3, 3>(0, 0), 1e-6)));
        EXPECT_LT((legacy.block<3, 3>(0, 3) - pose_cov.block<3, 3>(3, 0)).norm(), 1e-6 * pose_cov.norm()); // may be ~0, so compared absolutely
        EXPECT_TRUE((legacy.block<3, 3>(6, 6).isApprox(strain_cov.block<3, 3>(3, 3), 1e-6)));
        EXPECT_TRUE(legacy.isApprox(legacy.transpose()));
    }

    TEST(GroundTruthExport, SkipsSamplesOutsideTheEstimate)
    {
        ExportFixture f({{0.5, -1.0}, {0.5, 0.1}});
        EXPECT_EQ(f.exportAll(), 1u);
        std::string header;
        EXPECT_EQ(readRows(f.logs_dir / "error_pred.txt", header).size(), 1u);
    }

    TEST(GroundTruthExport, WithoutPosteriorOnlyTheMeanIsWritten)
    {
        ExportFixture f({{0.5, 0.1}});
        EXPECT_EQ(f.exportAll(/*with_covariance=*/false), 1u);
        std::string header;
        EXPECT_TRUE(readRows(f.logs_dir / "error_pred_cov.txt", header).empty());
        EXPECT_EQ(header, "time,arclength,");
    }

    TEST(GroundTruthExport, DoesNotConsumeTheGroundTruth)
    {
        ExportFixture f({{0.5, 0.1}, {1.0, 0.2}});
        f.exportAll();
        EXPECT_EQ(f.ground_truth->size(), 2u);
    }

    TEST(GroundTruthExport, ThrowsWhenTheOutputCannotBeWritten)
    {
        ExportFixture f({{0.5, 0.1}});
        const std::filesystem::path blocker = std::filesystem::path(testing::TempDir()) / "ground_truth_export_blocker";
        std::ofstream(blocker) << "not a directory";
        EXPECT_ANY_THROW(exportGroundTruthPredictions(*f.prior, f.estimator.getCurrentValues(), nullptr, {f.ground_truth}, (blocker / "logs").string()));
        std::filesystem::remove(blocker);
    }

    TEST(GroundTruthExport, PoseOnlyPriorKeepsTheFormatWithNanRates)
    {
        // WNOV holds no strain or velocity: same 20 / 326 columns, with nan in their place.
        using WNOVNode = Priors::WNOV::Node<Pose>;
        using WNOVValues = Priors::WNOV::NodeValues<Pose>;
        auto state = std::make_shared<Types::ContinuumRobotState<WNOVNode>>(1.0, 3);
        auto prior = std::make_shared<Priors::WNOV::Prior<Pose>>(state, Pose(), (Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
        Estimator estimator(ExportFixture::options());
        estimator.addPrior(prior);
        const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
        auto ground_truth = std::make_shared<Sensors::PoseSensor<WNOVValues>>(prior);
        ground_truth->addMeasurement(std::make_shared<Types::MeasurementBase>(0.5, 0.15, Eigen::Matrix4d::Identity(), R, Eigen::MatrixXi::Ones(6, 1)));
        for (double t : {0.0, 0.1, 0.2})
            estimator.estimate(t);
        estimator.finalize();

        const std::filesystem::path logs_dir = std::filesystem::path(testing::TempDir()) / "ground_truth_export_wnov";
        std::filesystem::remove_all(logs_dir);
        ASSERT_EQ(exportGroundTruthPredictions(*prior, estimator.getCurrentValues(), estimator.getCurrentCovariance().get(), {ground_truth}, logs_dir.string()), 1u);

        std::string header;
        const auto mean_rows = readRows(logs_dir / "error_pred.txt", header);
        ASSERT_EQ(mean_rows.size(), 1u);
        ASSERT_EQ(mean_rows[0].size(), 20u);
        const std::vector<double> expected_pose = {0, 0, 0.5, 0, 0, 0}; // straight along z, legacy [rho, phi]
        for (int i = 0; i < 6; i++)
            EXPECT_NEAR(mean_rows[0][2 + i], expected_pose[i], 1e-9) << "column " << i;
        for (int i = 6; i < 18; i++)
            EXPECT_TRUE(std::isnan(mean_rows[0][2 + i])) << "column " << i;

        const auto cov_rows = readRows(logs_dir / "error_pred_cov.txt", header);
        ASSERT_EQ(cov_rows.size(), 1u);
        ASSERT_EQ(cov_rows[0].size(), 2u + 18u * 18u);
        Eigen::MatrixXd posterior;
        ASSERT_NE(prior->query(0.15, 0.5, estimator.getCurrentValues(), nullptr, estimator.getCurrentCovariance().get(), &posterior), nullptr);
        const Eigen::Matrix<double, 6, 6> legacy_pose = tangentOrderSwap<1>() * posterior * tangentOrderSwap<1>().transpose();
        for (int r = 0; r < 18; r++)
            for (int c = 0; c < 18; c++)
            {
                const double value = cov_rows[0][2 + 18 * r + c];
                if (r < 6 && c < 6)
                    EXPECT_NEAR(value, legacy_pose(r, c), 1e-12 + 1e-9 * std::abs(legacy_pose(r, c))) << r << ", " << c;
                else
                    EXPECT_TRUE(std::isnan(value)) << r << ", " << c;
            }
    }
    TEST(GroundTruthExport, PoseAndStrainPriorWritesStrainAndNanVelocity)
    {
        // The quasi-static prior holds a pose and a strain: velocity columns are nan, and
        // the covariance holds the 12 x 12 pose/strain block with nan elsewhere.
        using QSNode = Priors::WNOAQuasiStatic::Node<Pose>;
        using QSValues = Priors::WNOAQuasiStatic::NodeValues<Pose>;
        const Eigen::Vector<double, 6> strain = (Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished();
        auto state = std::make_shared<Types::ContinuumRobotState<QSNode>>(1.0, 3);
        auto prior = std::make_shared<Priors::WNOAQuasiStatic::Prior<Pose>>(state, QSValues(Pose(), strain));
        Estimator estimator(ExportFixture::options());
        estimator.addPrior(prior);
        const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
        auto ground_truth = std::make_shared<Sensors::PoseSensor<QSValues>>(prior);
        ground_truth->addMeasurement(std::make_shared<Types::MeasurementBase>(0.5, 0.12, Eigen::Matrix4d::Identity(), R, Eigen::MatrixXi::Ones(6, 1)));
        for (double t : {0.0, 0.1, 0.2})
            estimator.estimate(t);
        estimator.finalize();

        const std::filesystem::path logs_dir = std::filesystem::path(testing::TempDir()) / "ground_truth_export_quasi_static";
        std::filesystem::remove_all(logs_dir);
        ASSERT_EQ(exportGroundTruthPredictions(*prior, estimator.getCurrentValues(), estimator.getCurrentCovariance().get(), {ground_truth}, logs_dir.string()), 1u);

        std::string header;
        const auto mean_rows = readRows(logs_dir / "error_pred.txt", header);
        ASSERT_EQ(mean_rows.size(), 1u);
        ASSERT_EQ(mean_rows[0].size(), 20u);
        const std::vector<double> expected = {0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0}; // legacy [rho, phi] pose, then strain
        for (int i = 0; i < 12; i++)
            EXPECT_NEAR(mean_rows[0][2 + i], expected[i], 1e-9) << "column " << i;
        for (int i = 12; i < 18; i++)
            EXPECT_TRUE(std::isnan(mean_rows[0][2 + i])) << "column " << i;

        const auto cov_rows = readRows(logs_dir / "error_pred_cov.txt", header);
        ASSERT_EQ(cov_rows.size(), 1u);
        ASSERT_EQ(cov_rows[0].size(), 2u + 18u * 18u);
        Eigen::MatrixXd posterior;
        ASSERT_NE(prior->query(0.12, 0.5, estimator.getCurrentValues(), nullptr, estimator.getCurrentCovariance().get(), &posterior), nullptr);
        ASSERT_EQ(posterior.rows(), 12);
        const Eigen::Matrix<double, 12, 12> legacy = tangentOrderSwap<2>() * posterior * tangentOrderSwap<2>().transpose();
        for (int r = 0; r < 18; r++)
            for (int c = 0; c < 18; c++)
            {
                const double value = cov_rows[0][2 + 18 * r + c];
                if (r < 12 && c < 12)
                    EXPECT_NEAR(value, legacy(r, c), 1e-12) << r << "," << c;
                else
                    EXPECT_TRUE(std::isnan(value)) << r << "," << c;
            }
    }
} // namespace Spacetime::Utilities
