#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/visualizer/default/objects/RuntimeWidget.hpp>

namespace Spacetime
{
    using namespace Visualization;
    using namespace Visualization::Default;

    static_assert(IsVisualizable<RuntimeWidget>);

    namespace
    {
        /// A snapshot after two steps of 1 and 3 ms construction, 4 and 6 ms solve, 0.5 and 1.5 ms covariance.
        EstimatorSnapshot twoSteps(bool covariance_enabled)
        {
            EstimatorSnapshot snapshot{1.0, gtsam::Values()};
            snapshot.covariance_enabled = covariance_enabled;
            for (const auto &[construction, solve, covariance] : {std::tuple{0.001, 0.004, 0.0005}, std::tuple{0.003, 0.006, 0.0015}})
            {
                snapshot.runtime.record(RuntimePhase::GraphConstruction, construction);
                snapshot.runtime.record(RuntimePhase::Solve, solve);
                if (covariance_enabled)
                    snapshot.runtime.record(RuntimePhase::Covariance, covariance);
            }
            return snapshot;
        }
    } // namespace

    TEST(RuntimeWidget, OnlineShowsTheLastStepAndTheRunningAverage)
    {
        const EstimatorSnapshot snapshot = twoSteps(true);
        EXPECT_EQ(formatRuntime(&snapshot, true),
                  "Graph Construction: 3.0 ms (avg 2.0 ms)   |   Solve: 6.0 ms (avg 5.0 ms)   |   Covariance Extraction: 1.5 ms (avg 1.0 ms)   |   Total: 10.5 ms (avg 8.0 ms)");
    }

    TEST(RuntimeWidget, OfflineShowsTheAverageAndTotal)
    {
        const EstimatorSnapshot snapshot = twoSteps(true);
        EXPECT_EQ(formatRuntime(&snapshot, false),
                  "Graph Construction: avg 2.0 ms, total 4.0 ms   |   Solve: avg 5.0 ms, total 10.0 ms   |   Covariance Extraction: avg 1.0 ms, total 2.0 ms   |   Total: avg 8.0 ms, total 16.0 ms");
    }

    TEST(RuntimeWidget, DisabledCovarianceShowsOff)
    {
        const EstimatorSnapshot snapshot = twoSteps(false);
        EXPECT_EQ(formatRuntime(&snapshot, true),
                  "Graph Construction: 3.0 ms (avg 2.0 ms)   |   Solve: 6.0 ms (avg 5.0 ms)   |   Covariance Extraction: off   |   Total: 9.0 ms (avg 7.0 ms)");
        EXPECT_EQ(formatRuntime(&snapshot, false),
                  "Graph Construction: avg 2.0 ms, total 4.0 ms   |   Solve: avg 5.0 ms, total 10.0 ms   |   Covariance Extraction: off   |   Total: avg 7.0 ms, total 14.0 ms");
    }

    TEST(RuntimeWidget, NoSnapshotShowsPlaceholders)
    {
        const std::string placeholders = "Graph Construction: --   |   Solve: --   |   Covariance Extraction: --   |   Total: --";
        EXPECT_EQ(formatRuntime(nullptr, true), placeholders);
        EXPECT_EQ(formatRuntime(nullptr, false), placeholders);
    }

    TEST(RuntimeWidget, UntimedPhaseShowsAPlaceholder)
    {
        // E.g. a batch snapshot before its solve: construction was timed, the solve not yet.
        EstimatorSnapshot snapshot{0.0, gtsam::Values()};
        snapshot.covariance_enabled = true;
        snapshot.runtime.record(RuntimePhase::GraphConstruction, 0.002);
        EXPECT_EQ(formatRuntime(&snapshot, false),
                  "Graph Construction: avg 2.0 ms, total 2.0 ms   |   Solve: --   |   Covariance Extraction: --   |   Total: avg 2.0 ms, total 2.0 ms");
    }

    TEST(RuntimeWidget, BatchShowsOnlyTheTotals)
    {
        // Construction is timed once per estimate(), the solve once by finalize(), so a per-phase average is not comparable across phases.
        EstimatorSnapshot snapshot = twoSteps(true);
        snapshot.batch = true;
        const std::string totals = "Graph Construction: 4.0 ms   |   Solve: 10.0 ms   |   Covariance Extraction: 2.0 ms   |   Total: 16.0 ms";
        EXPECT_EQ(formatRuntime(&snapshot, false), totals);
        EXPECT_EQ(formatRuntime(&snapshot, true), totals);
    }

    TEST(RuntimeWidget, BatchDisabledCovarianceShowsOff)
    {
        EstimatorSnapshot snapshot = twoSteps(false);
        snapshot.batch = true;
        EXPECT_EQ(formatRuntime(&snapshot, false),
                  "Graph Construction: 4.0 ms   |   Solve: 10.0 ms   |   Covariance Extraction: off   |   Total: 14.0 ms");
    }

    TEST(RuntimeWidget, BatchUntimedPhaseShowsAPlaceholder)
    {
        // A batch snapshot before finalize(): construction was timed, the solve not yet.
        EstimatorSnapshot snapshot{0.0, gtsam::Values()};
        snapshot.covariance_enabled = true;
        snapshot.batch = true;
        snapshot.runtime.record(RuntimePhase::GraphConstruction, 0.002);
        EXPECT_EQ(formatRuntime(&snapshot, false),
                  "Graph Construction: 2.0 ms   |   Solve: --   |   Covariance Extraction: --   |   Total: 2.0 ms");
    }

    TEST(RuntimeWidget, RoundsToOneDecimalInMilliseconds)
    {
        EstimatorSnapshot snapshot{0.0, gtsam::Values()};
        snapshot.runtime.record(RuntimePhase::GraphConstruction, 0.00126);
        snapshot.runtime.record(RuntimePhase::Solve, 0.00124);
        snapshot.runtime.record(RuntimePhase::Solve, 0.41176);
        EXPECT_EQ(formatRuntime(&snapshot, true),
                  "Graph Construction: 1.3 ms (avg 1.3 ms)   |   Solve: 411.8 ms (avg 206.5 ms)   |   Covariance Extraction: off   |   Total: 413.0 ms (avg 207.8 ms)");
        EXPECT_EQ(formatRuntime(&snapshot, false),
                  "Graph Construction: avg 1.3 ms, total 1.3 ms   |   Solve: avg 206.5 ms, total 413.0 ms   |   Covariance Extraction: off   |   Total: avg 207.8 ms, total 414.3 ms");
    }
} // namespace Spacetime
