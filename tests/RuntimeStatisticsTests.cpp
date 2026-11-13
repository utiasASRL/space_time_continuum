#include <stdexcept>

#include <gtest/gtest.h>

#include <spacetime/estimator/RuntimeStatistics.hpp>

namespace Spacetime
{
    namespace
    {
        void expectTimings(const PhaseTimings &actual, double graph_construction, double solve, double covariance)
        {
            EXPECT_DOUBLE_EQ(actual.graph_construction, graph_construction);
            EXPECT_DOUBLE_EQ(actual.solve, solve);
            EXPECT_DOUBLE_EQ(actual.covariance, covariance);
        }
    } // namespace

    TEST(RuntimeStatistics, EmptyIsAllZeros)
    {
        const RuntimeStatistics stats;
        expectTimings(stats.last(), 0.0, 0.0, 0.0);
        expectTimings(stats.average(), 0.0, 0.0, 0.0);
        expectTimings(stats.total(), 0.0, 0.0, 0.0);
        EXPECT_EQ(stats.count(RuntimePhase::GraphConstruction), 0u);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 0u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 0u);
    }

    TEST(RuntimeStatistics, SingleStep)
    {
        RuntimeStatistics stats;
        stats.record(RuntimePhase::GraphConstruction, 0.001);
        stats.record(RuntimePhase::Solve, 0.004);
        stats.record(RuntimePhase::Covariance, 0.002);
        expectTimings(stats.last(), 0.001, 0.004, 0.002);
        expectTimings(stats.average(), 0.001, 0.004, 0.002);
        expectTimings(stats.total(), 0.001, 0.004, 0.002);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 1u);
    }

    TEST(RuntimeStatistics, SeveralStepsAccumulate)
    {
        RuntimeStatistics stats;
        for (const double seconds : {1.0, 2.0, 6.0})
        {
            stats.record(RuntimePhase::GraphConstruction, seconds);
            stats.record(RuntimePhase::Solve, 2.0 * seconds);
            stats.record(RuntimePhase::Covariance, 3.0 * seconds);
        }
        expectTimings(stats.last(), 6.0, 12.0, 18.0);
        expectTimings(stats.average(), 3.0, 6.0, 9.0);
        expectTimings(stats.total(), 9.0, 18.0, 27.0);
        EXPECT_EQ(stats.count(RuntimePhase::GraphConstruction), 3u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 3u);
    }

    TEST(RuntimeStatistics, PhasesAreCountedIndependently)
    {
        // As in batch mode: one graph construction per estimate(), a single solve in finalize().
        RuntimeStatistics stats;
        stats.record(RuntimePhase::GraphConstruction, 1.0);
        stats.record(RuntimePhase::GraphConstruction, 3.0);
        stats.record(RuntimePhase::Solve, 5.0);
        EXPECT_EQ(stats.count(RuntimePhase::GraphConstruction), 2u);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 1u);
        EXPECT_EQ(stats.count(RuntimePhase::Covariance), 0u);
        expectTimings(stats.average(), 2.0, 5.0, 0.0);
        expectTimings(stats.total(), 4.0, 5.0, 0.0);
        expectTimings(stats.last(), 3.0, 5.0, 0.0);
    }

    TEST(RuntimeStatistics, ZeroDurationIsAccepted)
    {
        RuntimeStatistics stats;
        EXPECT_NO_THROW(stats.record(RuntimePhase::Solve, 0.0));
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 1u);
    }

    TEST(RuntimeStatistics, NegativeDurationThrows)
    {
        RuntimeStatistics stats;
        EXPECT_THROW(stats.record(RuntimePhase::Solve, -1e-6), std::invalid_argument);
        EXPECT_EQ(stats.count(RuntimePhase::Solve), 0u);
    }
} // namespace Spacetime
