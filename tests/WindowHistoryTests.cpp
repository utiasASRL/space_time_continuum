#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/GroundTruthExport.hpp>
#include <spacetime/estimator/utilities/WindowHistory.hpp>
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

        constexpr unsigned int kSpatialDim = 3;
        constexpr double kPeriod = 0.1;
        constexpr int kTicks = 6;

        /// Parameters making the smoother's LM loop behave as Gauss-Newton (see the regression configs).
        gtsam::LevenbergMarquardtParams gaussNewtonParams()
        {
            gtsam::LevenbergMarquardtParams params;
            params.lambdaInitial = 1e-10;
            params.lambdaFactor = 1.0;
            params.lambdaUpperBound = 1e-10;
            params.relativeErrorTol = 1e-14;
            params.absoluteErrorTol = 1e-14;
            params.maxIterations = 100;
            return params;
        }

        EstimatorOptions options(double lag)
        {
            EstimatorOptions o = Testing::estimatorOptions(lag);
            o.lm_params = gaussNewtonParams();
            o.compute_covariance = true;
            return o;
        }

        /// A 1 m robot along z whose tip is measured at every node time, extending by 2 cm
        /// per tick (so every new measurement moves the older rows), plus a ground-truth
        /// sensor at three arclengths, on and between the node times.
        struct Problem
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, kSpatialDim);
            std::shared_ptr<Priors::WNOA::Prior<Pose>> prior;
            std::shared_ptr<Sensors::PoseSensor<ValueType>> ground_truth;
            Estimator estimator;
            WindowHistory history;
            std::vector<gtsam::Values> values_per_tick;
            std::vector<std::shared_ptr<const PosteriorCovariance>> posterior_per_tick;

            explicit Problem(double lag) : estimator(options(lag))
            {
                ValueType reference;
                reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, reference);
                estimator.addPrior(prior);

                const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
                auto sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
                ground_truth = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
                for (int k = 0; k < kTicks; k++)
                {
                    Eigen::Matrix4d tip = Eigen::Matrix4d::Identity();
                    tip(2, 3) = 1.0 + 0.02 * k;
                    sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(1.0, k * kPeriod, tip, R, Eigen::MatrixXi::Ones(6, 1)));
                    for (double s : {0.0, 0.5, 1.0})
                        for (double dt : {0.0, 0.05})
                            if (k * kPeriod + dt <= (kTicks - 1) * kPeriod)
                                ground_truth->addMeasurement(std::make_shared<Types::MeasurementBase>(s, k * kPeriod + dt, Eigen::Matrix4d::Identity(), R, Eigen::MatrixXi::Ones(6, 1)));
                }
                estimator.addSensor(sensor);
            }

            /// Run every tick, recording the history and each tick's window.
            void run()
            {
                for (int k = 0; k < kTicks; k++)
                {
                    estimator.estimate(k * kPeriod);
                    history.record(estimator);
                    values_per_tick.push_back(estimator.getCurrentValues());
                    posterior_per_tick.push_back(estimator.getCurrentCovariance());
                }
                estimator.finalize();
                history.record(estimator);
            }

            std::shared_ptr<NodeType> node(unsigned int row, unsigned int n) const
            {
                return std::dynamic_pointer_cast<NodeType>(state->estimation_nodes.at(row * kSpatialDim + n));
            }

            /// The last tick whose window still held `row`.
            int lastTickHolding(unsigned int row) const
            {
                int last = -1;
                for (int k = 0; k < static_cast<int>(values_per_tick.size()); k++)
                    if (values_per_tick[k].exists(node(row, 0)->poseKey()))
                        last = k;
                return last;
            }

            std::size_t rowsInWindow(int tick) const
            {
                std::size_t rows = 0;
                for (unsigned int row = 0; row < kTicks; row++)
                    rows += values_per_tick[tick].exists(node(row, 0)->poseKey());
                return rows;
            }
        };

        std::size_t countLines(const std::filesystem::path &path)
        {
            std::ifstream file(path);
            std::size_t lines = 0;
            for (std::string line; std::getline(file, line);)
                lines++;
            return lines;
        }
    } // namespace

    TEST(WindowHistory, ExtractionNamesParse)
    {
        EXPECT_EQ(parseExtraction("final"), Extraction::Final);
        EXPECT_EQ(parseExtraction("back"), Extraction::Back);
        EXPECT_THROW(parseExtraction("front"), std::invalid_argument);
        EXPECT_THROW(parseExtraction(""), std::invalid_argument);
    }

    TEST(WindowHistory, ExtractionIsReadFromTheEstimatorSection)
    {
        auto loaderFor = [](const std::string &contents)
        {
            const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / "window_history_config.json";
            std::ofstream(path) << contents;
            ConfigLoader loader;
            EXPECT_TRUE(loader.loadConfig(path.string()));
            std::filesystem::remove(path);
            return loader;
        };
        EXPECT_EQ(loaderFor(R"({"estimator": {"smootherLag": 0.1}})").loadExtraction(), Extraction::Final);
        EXPECT_EQ(loaderFor(R"({"other": 1})").loadExtraction(), Extraction::Final);
        EXPECT_EQ(loaderFor(R"({"estimator": {"extraction": "back"}})").loadExtraction(), Extraction::Back);
        EXPECT_THROW(loaderFor(R"({"estimator": {"extraction": "middle"}})").loadExtraction(), std::invalid_argument);
    }

    TEST(WindowHistory, LagKeepsOneRowPerPeriodPlusTheNewest)
    {
        // smootherLag = L keeps the rows within L of the newest one, i.e. L / period + 1 rows
        // (the legacy window of K = L / period + 1 rows; see the RoboSoft regression).
        for (const int periods : {0, 1, 2})
        {
            Problem problem(periods * kPeriod + 1e-6);
            problem.run();
            for (int k = 0; k < kTicks; k++)
                EXPECT_EQ(problem.rowsInWindow(k), static_cast<std::size_t>(std::min(k, periods) + 1)) << "lag " << periods << ", tick " << k;
        }
    }

    TEST(WindowHistory, RowsKeepTheirValuesFromTheLastWindowHoldingThem)
    {
        Problem problem(kPeriod + 1e-6);
        problem.run();

        ASSERT_EQ(problem.history.numRows(), static_cast<std::size_t>(kTicks));
        for (unsigned int row = 0; row < kTicks; row++)
        {
            const int tick = problem.lastTickHolding(row);
            ASSERT_GE(tick, 0);
            EXPECT_EQ(tick, std::min<int>(row + 1, kTicks - 1)) << "row " << row; // the back of the window, then the final window
            for (unsigned int n = 0; n < kSpatialDim; n++)
                for (const gtsam::Key key : problem.node(row, n)->keys())
                    EXPECT_TRUE(problem.history.values().at(key).equals_(problem.values_per_tick[tick].at(key), 0.0)) << "row " << row;
        }

        // The older rows did move after they were first estimated, so the history is not
        // simply each row's first estimate.
        const gtsam::Key first_tip = problem.node(0, kSpatialDim - 1)->poseKey();
        EXPECT_FALSE(problem.values_per_tick[0].at<Pose>(first_tip).equals(problem.history.values().at<Pose>(first_tip), 1e-9));
    }

    TEST(WindowHistory, PosteriorBetweenRowsComesFromTheEstimateThatLastHeldTheEarlierRow)
    {
        Problem problem(kPeriod + 1e-6);
        problem.run();

        EXPECT_EQ(problem.history.posteriorAt(-0.05), nullptr);
        for (unsigned int row = 0; row + 1 < kTicks; row++)
        {
            const int tick = problem.lastTickHolding(row);
            const PosteriorCovariance *expected = problem.posterior_per_tick[tick].get();
            ASSERT_NE(expected, nullptr);
            EXPECT_EQ(problem.history.posteriorAt(row * kPeriod), expected) << "row " << row;
            EXPECT_EQ(problem.history.posteriorAt(row * kPeriod + 0.05), expected) << "row " << row;

            // It holds the joint of the cell between this row and the next.
            Eigen::MatrixXd joint;
            EXPECT_TRUE(expected->joint({problem.node(row, 0)->poseKey(), problem.node(row + 1, 0)->poseKey(),
                                         problem.node(row, 1)->poseKey(), problem.node(row + 1, 1)->poseKey()},
                                        joint))
                << "row " << row;
        }
    }

    TEST(WindowHistory, FilterHasNoPosteriorBetweenRows)
    {
        Problem problem(0.0);
        problem.run();

        for (unsigned int row = 0; row + 1 < kTicks; row++)
        {
            const PosteriorCovariance *posterior = problem.history.posteriorAt(row * kPeriod + 0.05);
            ASSERT_NE(posterior, nullptr);
            EXPECT_TRUE(posterior->has(problem.node(row, 0)->poseKey()));
            EXPECT_FALSE(posterior->has(problem.node(row, 0)->poseKey(), problem.node(row + 1, 0)->poseKey()));

            Eigen::MatrixXd covariance;
            auto queried = problem.prior->query(row * kPeriod + 0.05, 0.5, problem.history.values(), nullptr, posterior, &covariance);
            ASSERT_NE(queried, nullptr); // the mean is still available
            EXPECT_EQ(covariance.size(), 0);
        }
    }

    TEST(WindowHistory, MatchesTheBatchWhenNothingIsMarginalized)
    {
        Problem window(100.0);
        window.run();
        Problem batch(Estimator::BATCH_LAG);
        batch.run();

        ASSERT_EQ(window.history.values().size(), batch.history.values().size());
        for (unsigned int row = 0; row < kTicks; row++)
            for (unsigned int n = 0; n < kSpatialDim; n++)
            {
                const ValueType a = window.node(row, n)->getValues(window.history.values());
                const ValueType b = batch.node(row, n)->getValues(batch.history.values());
                EXPECT_TRUE(a.pose().equals(b.pose(), 1e-6)) << "row " << row;
                EXPECT_TRUE(a.epsilon().isApprox(b.epsilon(), 1e-6)) << "row " << row;
                EXPECT_LT((a.varpi() - b.varpi()).norm(), 1e-6) << "row " << row;
            }
        // A single posterior serves every time.
        EXPECT_EQ(batch.history.posteriorAt(0.0), batch.estimator.getCurrentCovariance().get());
        EXPECT_EQ(batch.history.posteriorAt(0.45), batch.estimator.getCurrentCovariance().get());
    }

    TEST(WindowHistory, ExportCoversTheWholeRunWhereTheFinalWindowDoesNot)
    {
        Problem problem(kPeriod + 1e-6);
        problem.run();
        const std::size_t samples = problem.ground_truth->size();
        const std::filesystem::path logs = std::filesystem::path(testing::TempDir()) / "window_history_logs";
        std::filesystem::remove_all(logs);

        const std::size_t final_window = exportGroundTruthPredictions(
            *problem.prior, problem.estimator.getCurrentValues(), problem.estimator.getCurrentCovariance().get(),
            {problem.ground_truth}, logs.string());
        EXPECT_LT(final_window, samples);

        const std::size_t history = exportGroundTruthPredictions(
            *problem.prior, problem.history.values(), [&](double time)
            { return problem.history.posteriorAt(time); }, {problem.ground_truth}, logs.string());
        EXPECT_EQ(history, samples);
        EXPECT_EQ(countLines(logs / "error_pred.txt"), samples + 1);
        EXPECT_EQ(countLines(logs / "error_pred_cov.txt"), samples + 1); // a covariance for every sample
        std::filesystem::remove_all(logs);
    }
} // namespace Spacetime::Utilities
