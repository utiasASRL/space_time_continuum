#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime/interface/EstimationSchedule.hpp>

namespace Spacetime::Interface
{
    namespace
    {
        /// Every time an offline schedule estimates at, start to end.
        std::vector<double> offlineTimes(double start, double end, double period)
        {
            EstimationSchedule schedule(start, end, period, /*real_time=*/false);
            std::vector<double> times;
            for (double t = schedule.firstTime(); !schedule.isDone(t); t = schedule.next(t))
                times.push_back(t);
            return times;
        }
    } // namespace

    TEST(EstimationSchedule, RejectsNonPositivePeriod)
    {
        EXPECT_THROW(EstimationSchedule(0.0, 1.0, 0.0, false), std::invalid_argument);
        EXPECT_THROW(EstimationSchedule(0.0, 1.0, -0.1, true), std::invalid_argument);
    }

    TEST(EstimationSchedule, StartsAtTheStartTime)
    {
        EstimationSchedule schedule(2.5, 3.0, 0.1, false);
        EXPECT_DOUBLE_EQ(schedule.firstTime(), 2.5);
        EXPECT_EQ(schedule.tick(), 0);
    }

    TEST(EstimationSchedule, IsDoneOnlyPastTheEnd)
    {
        EstimationSchedule schedule(0.0, 1.0, 0.1, false);
        EXPECT_FALSE(schedule.isDone(1.0)); // the end time itself is estimated
        EXPECT_TRUE(schedule.isDone(1.0 + 1e-9));
    }

    // Offline, the time is computed as start + k * period, never accumulated, so many
    // steps of a period that is not exactly representable still land exactly.
    TEST(EstimationSchedule, OfflineStepsAreExact)
    {
        const auto times = offlineTimes(0.0, 3.0, 0.1);
        ASSERT_EQ(times.size(), 31u); // 0.0, 0.1, ..., 3.0
        for (std::size_t k = 0; k < times.size(); k++)
            EXPECT_EQ(times[k], 0.0 + static_cast<double>(k) * 0.1) << "step " << k;
    }

    TEST(EstimationSchedule, PeriodControlsTheOfflineStepCount)
    {
        EXPECT_EQ(offlineTimes(0.0, 0.1, 0.05).size(), 3u); // {0, 0.05, 0.1}
        EXPECT_EQ(offlineTimes(0.0, 0.1, 0.02).size(), 6u); // {0, 0.02, ..., 0.1}
    }

    TEST(EstimationSchedule, OfflineIgnoresTheCurrentTime)
    {
        EstimationSchedule schedule(0.0, 1.0, 0.1, false);
        EXPECT_DOUBLE_EQ(schedule.next(100.0), 0.1);
        EXPECT_DOUBLE_EQ(schedule.next(-5.0), 0.2);
    }

    // In real time, an estimate that finished early waits for the next boundary.
    TEST(EstimationSchedule, RealTimeWakesAtTheNextBoundary)
    {
        EstimationSchedule schedule(0.0, 10.0, 0.1, true);
        EXPECT_DOUBLE_EQ(schedule.next(0.05), 0.1);
        EXPECT_DOUBLE_EQ(schedule.next(0.12), 0.2);
        EXPECT_EQ(schedule.tick(), 2);
    }

    // An overrun runs again immediately, then resumes at the next future boundary:
    // the boundaries it missed are skipped, not caught up on back to back.
    TEST(EstimationSchedule, RealTimeOverrunDoesNotBurstToCatchUp)
    {
        EstimationSchedule schedule(0.0, 10.0, 0.1, true);
        EXPECT_DOUBLE_EQ(schedule.next(0.05), 0.1);
        EXPECT_DOUBLE_EQ(schedule.next(0.55), 0.55); // overran 0.2 .. 0.5: now
        EXPECT_EQ(schedule.tick(), 5);
        EXPECT_DOUBLE_EQ(schedule.next(0.56), 0.6); // back on the schedule
    }

    TEST(EstimationSchedule, RealTimeBoundariesAreRelativeToTheStart)
    {
        EstimationSchedule schedule(10.0, 20.0, 0.5, true);
        EXPECT_DOUBLE_EQ(schedule.next(10.1), 10.5);
    }

    // A gap between estimates beyond the period (by more than 10%) means estimation is
    // running too slowly, e.g. after a real-time overrun.
    TEST(EstimationSchedule, IsLateOnlyWhenTheGapExceedsThePeriod)
    {
        EstimationSchedule schedule(0.0, 10.0, 0.1, true);
        EXPECT_FALSE(schedule.isLate(0.0, 0.05)) << "faster than the period is fine";
        EXPECT_FALSE(schedule.isLate(0.05, 0.15)) << "exactly the period is fine";
        EXPECT_FALSE(schedule.isLate(0.0, 0.105)) << "within the 10% slack";
        EXPECT_TRUE(schedule.isLate(0.15, 0.3)) << "a 0.15 gap exceeds the 0.1 period";
    }

    TEST(EstimationSchedule, AccumulatedPeriodStepsAreNotLate)
    {
        // A loop that ACCUMULATES t += period (rather than computing start + k * period, as
        // the offline schedule does) drifts by a few ulps per step. 0.1 is not exactly
        // representable, so some gaps land a hair above the period; the slack absorbs that.
        EstimationSchedule schedule(0.0, 10.0, 0.1, true);
        double t = 0.0;
        for (int k = 0; k < 100; ++k, t += 0.1)
            EXPECT_FALSE(schedule.isLate(t, t + 0.1)) << "step " << k;
    }
} // namespace Spacetime::Interface
