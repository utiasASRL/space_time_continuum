#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/EstimatorRunner.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::NodeValues<Pose>;

        constexpr unsigned int kSpatialDim = 3;
        constexpr double kPeriod = 0.1;

        /// A straight 1 m robot along z whose tip is measured 5 cm further out, at every node time.
        struct Problem
        {
            std::shared_ptr<StateType> state;
            std::shared_ptr<Priors::WNOA::Prior<Pose>> prior;
            std::shared_ptr<Sensors::PoseSensor<ValueType>> sensor;

            explicit Problem(Estimator &estimator, int num_measurements = 4)
            {
                state = std::make_shared<StateType>(1.0, kSpatialDim);
                ValueType reference;
                reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, reference);
                estimator.addPrior(prior);

                const Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
                sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
                for (int k = 0; k < num_measurements; k++)
                {
                    Eigen::Matrix4d tip = Eigen::Matrix4d::Identity();
                    tip(2, 3) = 1.05;
                    sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(1.0, k * kPeriod, tip, covariance, Eigen::MatrixXi::Ones(6, 1)));
                }
                estimator.addSensor(sensor);
            }

            std::size_t numKeys() const { return state->estimation_nodes.size() * 3; }
        };

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

        /// A pose sensor recording every Types::SensorBase::onMarginalization call it receives.
        struct MarginalizationSpy : public Sensors::PoseSensor<ValueType>
        {
            using Sensors::PoseSensor<ValueType>::PoseSensor;
            std::vector<double> cutoffs;
            std::vector<gtsam::Values> values;

            void onMarginalization(double cutoff, const gtsam::Values &current_values) override
            {
                cutoffs.push_back(cutoff);
                values.push_back(current_values);
            }
        };

        std::shared_ptr<MarginalizationSpy> addSpy(Estimator &estimator, const Problem &problem)
        {
            auto spy = std::make_shared<MarginalizationSpy>(problem.prior);
            estimator.addSensor(spy);
            return spy;
        }

        void runTicks(Estimator &estimator, int ticks)
        {
            for (int k = 0; k < ticks; k++)
                estimator.estimate(k * kPeriod);
        }

        /// Sliding-window options using the incremental (iSAM2) smoother, relinearizing every variable at every update.
        EstimatorOptions incrementalOptions(double lag)
        {
            EstimatorOptions options = Testing::estimatorOptions(lag);
            options.smoother = EstimatorOptions::Smoother::Incremental;
            options.isam2_params.relinearizeThreshold = 0.0;
            options.isam2_params.relinearizeSkip = 1;
            return options;
        }

        /// Expects every node of `a` (estimated by `ea`) and `b` (by `eb`) to agree within `tolerance`.
        void expectSameNodes(const Problem &a, const Estimator &ea, const Problem &b, const Estimator &eb, double tolerance)
        {
            ASSERT_EQ(a.state->estimation_nodes.size(), b.state->estimation_nodes.size());
            for (std::size_t i = 0; i < a.state->estimation_nodes.size(); i++)
            {
                const auto node_a = std::dynamic_pointer_cast<NodeType>(a.state->estimation_nodes[i]);
                const auto node_b = std::dynamic_pointer_cast<NodeType>(b.state->estimation_nodes[i]);
                if (!ea.getCurrentValues().exists(node_a->poseKey()) || !eb.getCurrentValues().exists(node_b->poseKey()))
                    continue; // marginalized
                const ValueType va = node_a->getValues(ea.getCurrentValues());
                const ValueType vb = node_b->getValues(eb.getCurrentValues());
                EXPECT_TRUE(va.pose().equals(vb.pose(), tolerance)) << "node " << i;
                EXPECT_TRUE(va.epsilon().isApprox(vb.epsilon(), tolerance)) << "node " << i;
                EXPECT_LT((va.varpi() - vb.varpi()).norm(), tolerance) << "node " << i;
            }
        }
    } // namespace

    TEST(BatchEstimator, LagSelectsTheMode)
    {
        EXPECT_TRUE(Estimator(Testing::estimatorOptions(Estimator::BATCH_LAG)).isBatch());
        EXPECT_FALSE(Estimator(Testing::estimatorOptions(0.0)).isBatch());
        EXPECT_FALSE(Estimator(Testing::estimatorOptions(0.5)).isBatch());

        EstimatorOptions options = Testing::estimatorOptions(-1.0);
        EXPECT_TRUE(Estimator(options).isBatch());
    }

    TEST(BatchEstimator, OtherNegativeLagsAreRejected)
    {
        EXPECT_THROW(Estimator(Testing::estimatorOptions(-0.5)), std::invalid_argument);
        EXPECT_THROW(Estimator(Testing::estimatorOptions(-2.0)), std::invalid_argument);
    }

    TEST(BatchEstimator, EstimateOnlyAccumulatesUntilFinalize)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator);
        runTicks(estimator, 4);

        // The initial guess of every state is available, but nothing was solved yet.
        EXPECT_EQ(estimator.getCurrentValues().size(), problem.numKeys());
        EXPECT_EQ(estimator.getFactors().size(), 0u);
        const gtsam::Key tip_key = problem.state->estimation_nodes.back()->keys().front();
        EXPECT_THROW(estimator.covariance(tip_key), std::runtime_error);

        const Pose initial_tip = estimator.getCurrentValues().at<Pose>(tip_key);
        estimator.finalize();
        const Pose solved_tip = estimator.getCurrentValues().at<Pose>(tip_key);

        EXPECT_GT(estimator.getFactors().size(), 0u);
        EXPECT_NEAR(initial_tip.translation().z(), 1.0, 1e-12);
        EXPECT_GT(solved_tip.translation().z(), 1.0 + 1e-3); // pulled toward the 1.05 m measurements
        EXPECT_NO_THROW(estimator.covariance(tip_key));
    }

    TEST(BatchEstimator, NothingIsMarginalized)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator, 20);
        runTicks(estimator, 20); // 2 s, far longer than any window
        estimator.finalize();

        EXPECT_EQ(estimator.getCurrentValues().size(), problem.numKeys());
        for (const auto &node : problem.state->estimation_nodes)
            for (const gtsam::Key key : node->keys())
                EXPECT_TRUE(estimator.getCurrentValues().exists(key));
    }

    TEST(BatchEstimator, MatchesAnUnmarginalizedSlidingWindow)
    {
        // A window longer than the whole run marginalizes nothing, so after its last update
        // it has solved the same problem as the batch.
        Estimator batch(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem batch_problem(batch);
        runTicks(batch, 4);
        batch.finalize();

        Estimator window(Testing::estimatorOptions(100.0, gaussNewtonParams()));
        Problem window_problem(window);
        runTicks(window, 4);

        ASSERT_EQ(batch_problem.state->estimation_nodes.size(), window_problem.state->estimation_nodes.size());
        for (std::size_t i = 0; i < batch_problem.state->estimation_nodes.size(); i++)
        {
            auto batch_node = std::dynamic_pointer_cast<NodeType>(batch_problem.state->estimation_nodes[i]);
            auto window_node = std::dynamic_pointer_cast<NodeType>(window_problem.state->estimation_nodes[i]);
            const ValueType a = batch_node->getValues(batch.getCurrentValues());
            const ValueType b = window_node->getValues(window.getCurrentValues());
            EXPECT_TRUE(a.pose().equals(b.pose(), 1e-6)) << "node " << i;
            EXPECT_TRUE(a.epsilon().isApprox(b.epsilon(), 1e-6)) << "node " << i;
            EXPECT_LT((a.varpi() - b.varpi()).norm(), 1e-6) << "node " << i;
        }
    }

    TEST(BatchEstimator, FinalizeIsANoOpInSlidingWindowMode)
    {
        Estimator estimator(Testing::estimatorOptions(0.25));
        Problem problem(estimator);
        runTicks(estimator, 4);
        const gtsam::Values before = estimator.getCurrentValues();
        estimator.finalize();
        EXPECT_TRUE(before.equals(estimator.getCurrentValues()));
        EXPECT_DOUBLE_EQ(estimator.getBatchTimings().solve, 0.0);
    }

    TEST(BatchEstimator, SecondFinalizeWithoutNewDataIsANoOp)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator);
        runTicks(estimator, 4);
        estimator.finalize();
        const gtsam::Values solved = estimator.getCurrentValues();
        const std::size_t factors = estimator.getFactors().size();
        estimator.finalize();
        EXPECT_TRUE(solved.equals(estimator.getCurrentValues()));
        EXPECT_EQ(estimator.getFactors().size(), factors);
    }

    TEST(BatchEstimator, EstimatorRunnerSolvesTheBatchOnceItEnds)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator);
        Interface::Standalone::RunnerOptions options;
        options.end_time = 0.3 + 1e-9;
        options.period = kPeriod;
        Interface::Standalone::EstimatorRunner runner(estimator, std::make_shared<Interface::Standalone::SteppedClock>(), options);
        runner.run();

        EXPECT_EQ(problem.state->estimation_nodes.size(), 4u * kSpatialDim);
        EXPECT_GT(estimator.getFactors().size(), 0u);
    }

    TEST(BatchEstimator, EstimatorRunnerRejectsRealTimeBatch)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        Problem problem(estimator);
        Interface::Standalone::RunnerOptions options;
        options.end_time = 0.3;
        options.period = kPeriod;
        Interface::Standalone::EstimatorRunner runner(estimator, std::make_shared<Interface::Standalone::WallClock>(), options);
        EXPECT_THROW(runner.start(), std::invalid_argument);
        EXPECT_TRUE(problem.state->estimation_nodes.empty());
    }

    TEST(BatchEstimator, OrderingDoesNotChangeTheSolution)
    {
        gtsam::LevenbergMarquardtParams colamd = gaussNewtonParams(), metis = gaussNewtonParams();
        colamd.orderingType = gtsam::Ordering::COLAMD;
        metis.orderingType = gtsam::Ordering::METIS;

        Estimator a(Testing::estimatorOptions(Estimator::BATCH_LAG, colamd));
        Problem problem_a(a);
        runTicks(a, 4);
        a.finalize();
        Estimator b(Testing::estimatorOptions(Estimator::BATCH_LAG, metis));
        Problem problem_b(b);
        runTicks(b, 4);
        b.finalize();

        EXPECT_EQ(a.options().lm_params.orderingType, gtsam::Ordering::COLAMD);
        EXPECT_EQ(b.options().lm_params.orderingType, gtsam::Ordering::METIS);
        expectSameNodes(problem_a, a, problem_b, b, 1e-9);
    }

    TEST(BatchEstimator, SmootherStateIsAvailableAfterFinalize)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator);
        runTicks(estimator, 4);
        EXPECT_TRUE(estimator.timestamps().empty());
        estimator.finalize();

        EXPECT_EQ(estimator.timestamps().size(), problem.numKeys());
        EXPECT_DOUBLE_EQ(estimator.getCurrentTimestamp(), 3 * kPeriod);
        EXPECT_TRUE(std::isinf(estimator.smootherLag()));
        // The smoother holds the solution, not the initial guess.
        const gtsam::Key tip_key = problem.state->estimation_nodes.back()->keys().front();
        EXPECT_TRUE(estimator.getLinearizationPoint().at<Pose>(tip_key).equals(estimator.getCurrentValues().at<Pose>(tip_key), 1e-12));
        EXPECT_NE(dynamic_cast<const gtsam::BatchFixedLagSmoother *>(&estimator.smoother()), nullptr);
    }

    TEST(BatchEstimator, FinalizeReportsTheSolve)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator);
        runTicks(estimator, 4);
        estimator.finalize();

        const auto &result = estimator.getBatchResult();
        EXPECT_GE(result.iterations, 1u);
        EXPECT_GT(result.initial_error, 0.0);
        EXPECT_LT(result.final_error, result.initial_error);
        EXPECT_NEAR(result.final_error, estimator.getFactors().error(estimator.getLinearizationPoint()), 1e-9 * result.initial_error);
    }

    TEST(BatchEstimator, ConvergenceTolerancesAreHonored)
    {
        // Any decrease is relatively smaller than 1: the solve stops after one iteration.
        gtsam::LevenbergMarquardtParams loose = gaussNewtonParams();
        loose.relativeErrorTol = 1.0;
        Estimator a(Testing::estimatorOptions(Estimator::BATCH_LAG, loose));
        Problem problem_a(a);
        runTicks(a, 4);
        a.finalize();
        EXPECT_EQ(a.getBatchResult().iterations, 1u);

        // Damped steps only approach the minimum: tight tolerances need several of them.
        gtsam::LevenbergMarquardtParams tight = Testing::lmParams();
        tight.lambdaInitial = 1.0;
        tight.relativeErrorTol = 1e-14;
        tight.absoluteErrorTol = 1e-14;
        Estimator b(Testing::estimatorOptions(Estimator::BATCH_LAG, tight));
        Problem problem_b(b);
        runTicks(b, 4);
        b.finalize();
        EXPECT_GT(b.getBatchResult().iterations, 1u);
    }

    TEST(IncrementalEstimator, SmootherOptionSelectsTheBackend)
    {
        Estimator incremental(incrementalOptions(0.5));
        EXPECT_FALSE(incremental.isBatch());
        EXPECT_DOUBLE_EQ(incremental.smootherLag(), 0.5);
        EXPECT_NE(dynamic_cast<const gtsam::IncrementalFixedLagSmoother *>(&incremental.smoother()), nullptr);

        Estimator window(Testing::estimatorOptions(0.5));
        EXPECT_NE(dynamic_cast<const gtsam::BatchFixedLagSmoother *>(&window.smoother()), nullptr);
    }

    TEST(IncrementalEstimator, BatchModeIgnoresTheSmootherOption)
    {
        EstimatorOptions options = incrementalOptions(Estimator::BATCH_LAG);
        options.lm_params = gaussNewtonParams();
        Estimator incremental(options);
        EXPECT_TRUE(incremental.isBatch());
        Problem problem_a(incremental);
        runTicks(incremental, 4);
        incremental.finalize();

        Estimator batch(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem_b(batch);
        runTicks(batch, 4);
        batch.finalize();
        expectSameNodes(problem_a, incremental, problem_b, batch, 1e-9);
    }

    TEST(IncrementalEstimator, MatchesTheBatchSmootherWithoutMarginalization)
    {
        Estimator incremental(incrementalOptions(100.0));
        Problem problem_a(incremental);
        runTicks(incremental, 4);

        Estimator window(Testing::estimatorOptions(100.0, gaussNewtonParams()));
        Problem problem_b(window);
        runTicks(window, 4);

        EXPECT_EQ(incremental.getCurrentValues().size(), problem_a.numKeys());
        expectSameNodes(problem_a, incremental, problem_b, window, 1e-6);
    }

    TEST(IncrementalEstimator, FiniteLagMarginalizesOldStates)
    {
        Estimator estimator(incrementalOptions(1.5 * kPeriod));
        Problem problem(estimator, 6);
        runTicks(estimator, 6);

        // Rows older than 0.5 - 0.15 (rows 0 to 3) are gone from the smoother and the values.
        const double cutoff = estimator.getCurrentTimestamp() - estimator.smootherLag();
        EXPECT_DOUBLE_EQ(estimator.getCurrentTimestamp(), 5 * kPeriod);
        for (const auto &[key, time] : estimator.timestamps())
            EXPECT_GE(time, cutoff);
        for (unsigned int row = 0; row < 6; row++)
            for (unsigned int i = 0; i < kSpatialDim; i++)
                for (const gtsam::Key key : problem.state->estimation_nodes.at(row * kSpatialDim + i)->keys())
                    EXPECT_EQ(estimator.getCurrentValues().exists(key), row >= 4) << "row " << row;
    }

    TEST(IncrementalEstimator, MatchesTheBatchSmootherWithMarginalization)
    {
        Estimator incremental(incrementalOptions(1.5 * kPeriod));
        Problem problem_a(incremental, 6);
        runTicks(incremental, 6);

        Estimator window(Testing::estimatorOptions(1.5 * kPeriod, gaussNewtonParams()));
        Problem problem_b(window, 6);
        runTicks(window, 6);

        EXPECT_EQ(incremental.getCurrentValues().size(), window.getCurrentValues().size());
        expectSameNodes(problem_a, incremental, problem_b, window, 1e-4);
    }

    TEST(IncrementalEstimator, CovarianceIsAvailableInTheWindow)
    {
        EstimatorOptions options = incrementalOptions(1.5 * kPeriod);
        options.compute_covariance = true;
        Estimator incremental(options);
        Problem problem_a(incremental, 6);
        runTicks(incremental, 6);

        options.smoother = EstimatorOptions::Smoother::Batch;
        options.lm_params = gaussNewtonParams();
        Estimator window(options);
        Problem problem_b(window, 6);
        runTicks(window, 6);

        ASSERT_NE(incremental.getCurrentCovariance(), nullptr);
        const gtsam::Key key_a = problem_a.state->estimation_nodes.back()->keys().front();
        const gtsam::Key key_b = problem_b.state->estimation_nodes.back()->keys().front();
        const gtsam::Matrix a = incremental.covariance(key_a), b = window.covariance(key_b);
        EXPECT_TRUE(a.allFinite());
        EXPECT_GT(a.diagonal().minCoeff(), 0.0);
        EXPECT_LT((a - b).norm(), 1e-3 * b.norm());
    }

    TEST(SensorMarginalizationHook, SlidingWindowPassesTheWindowStartBeforeEachUpdate)
    {
        Estimator estimator(Testing::estimatorOptions(1.5 * kPeriod, gaussNewtonParams()));
        Problem problem(estimator);
        auto spy = addSpy(estimator, problem);
        runTicks(estimator, 4);

        // Each update keeps the states no older than the newest time minus the lag.
        const std::vector<double> expected = {0.0, 0.0, kPeriod, 2 * kPeriod};
        ASSERT_EQ(spy->cutoffs.size(), expected.size());
        for (std::size_t k = 0; k < expected.size(); k++)
            EXPECT_NEAR(spy->cutoffs[k], expected[k], 1e-9) << "tick " << k;

        // The values still hold the states about to be marginalized, i.e. row 0 at tick 2.
        for (unsigned int i = 0; i < kSpatialDim; i++)
            for (const gtsam::Key key : problem.state->estimation_nodes.at(i)->keys())
                EXPECT_TRUE(spy->values[2].exists(key));
    }

    TEST(SensorMarginalizationHook, BatchPassesAnInfiniteCutoffOnceAfterTheSolve)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem problem(estimator);
        auto spy = addSpy(estimator, problem);
        runTicks(estimator, 4);
        EXPECT_TRUE(spy->cutoffs.empty());

        estimator.finalize();
        ASSERT_EQ(spy->cutoffs.size(), 1u);
        EXPECT_TRUE(std::isinf(spy->cutoffs.front()));
        EXPECT_EQ(spy->values.front().size(), problem.numKeys());
    }

    TEST(EstimatorRuntime, SlidingWindowTimesEveryPhaseOfEveryStep)
    {
        EstimatorOptions options = incrementalOptions(0.25);
        options.compute_covariance = true;
        Estimator estimator(options);
        Problem problem(estimator);
        runTicks(estimator, 4);

        const RuntimeStatistics &stats = estimator.getRuntimeStatistics();
        EXPECT_EQ(stats.count(RuntimePhase::GraphConstruction), 4u);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 4u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 4u);
        const PhaseTimings last = stats.last();
        EXPECT_GT(last.graph_construction, 0.0);
        EXPECT_GT(last.solve, 0.0);
        EXPECT_GT(last.covariance, 0.0);
        // The phases are disjoint parts of the last estimate() call. Each phase is its own
        // difference of clock readings, so their sum can exceed the whole by a rounding error.
        EXPECT_LE(last.graph_construction + last.solve + last.covariance, estimator.getLastEstimateTime() * (1.0 + 1e-12));
    }

    TEST(EstimatorRuntime, CovarianceIsNotTimedWhenDisabled)
    {
        Estimator estimator(incrementalOptions(0.25));
        Problem problem(estimator);
        runTicks(estimator, 3);

        const RuntimeStatistics &stats = estimator.getRuntimeStatistics();
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 3u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 0u);
        EXPECT_DOUBLE_EQ(stats.total().covariance, 0.0);
        ASSERT_NE(estimator.getSnapshot(), nullptr);
        EXPECT_FALSE(estimator.getSnapshot()->covariance_enabled);
    }

    TEST(EstimatorRuntime, SnapshotRecordsTheEstimationMode)
    {
        Estimator batch(Testing::estimatorOptions(Estimator::BATCH_LAG, gaussNewtonParams()));
        Problem batch_problem(batch);
        runTicks(batch, 3);
        batch.finalize();
        ASSERT_NE(batch.getSnapshot(), nullptr);
        EXPECT_TRUE(batch.getSnapshot()->batch);

        Estimator sliding(incrementalOptions(0.25));
        Problem sliding_problem(sliding);
        runTicks(sliding, 3);
        ASSERT_NE(sliding.getSnapshot(), nullptr);
        EXPECT_FALSE(sliding.getSnapshot()->batch);
    }

    TEST(EstimatorRuntime, SnapshotCarriesTheStatisticsAtPublication)
    {
        EstimatorOptions options = incrementalOptions(0.25);
        options.compute_covariance = true;
        Estimator estimator(options);
        Problem problem(estimator);
        runTicks(estimator, 2);
        const auto first = estimator.getSnapshot();
        estimator.estimate(2 * kPeriod);
        const auto second = estimator.getSnapshot();

        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_TRUE(second->covariance_enabled);
        // Each snapshot is a copy: the first is unchanged by the later step.
        EXPECT_EQ(first->runtime.count(RuntimePhase::Solve), 2u);
        EXPECT_EQ(second->runtime.count(RuntimePhase::Solve), 3u);
        EXPECT_EQ(second->runtime.count(RuntimePhase::Covariance), 3u);
        EXPECT_DOUBLE_EQ(second->runtime.total().solve, estimator.getRuntimeStatistics().total().solve);
    }

    TEST(EstimatorRuntime, BatchTimesConstructionPerStepAndASingleSolve)
    {
        EstimatorOptions options = Testing::estimatorOptions(Estimator::BATCH_LAG);
        options.lm_params = gaussNewtonParams();
        options.compute_covariance = true;
        Estimator estimator(options);
        Problem problem(estimator);
        runTicks(estimator, 4);

        const RuntimeStatistics &stats = estimator.getRuntimeStatistics();
        EXPECT_EQ(stats.count(RuntimePhase::GraphConstruction), 4u);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 0u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 0u);
        EXPECT_GT(stats.total().graph_construction, 0.0);

        estimator.finalize();
        EXPECT_EQ(stats.count(RuntimePhase::GraphConstruction), 4u);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 1u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 1u);
        // getBatchTimings() reads the same measurements.
        EXPECT_DOUBLE_EQ(estimator.getBatchTimings().solve, stats.last().solve);
        EXPECT_DOUBLE_EQ(estimator.getBatchTimings().covariance, stats.last().covariance);

        ASSERT_NE(estimator.getSnapshot(), nullptr);
        EXPECT_EQ(estimator.getSnapshot()->runtime.count(RuntimePhase::Solve), 1u);
        EXPECT_EQ(estimator.getSnapshot()->runtime.count(RuntimePhase::GraphConstruction), 4u);
    }
} // namespace Spacetime
