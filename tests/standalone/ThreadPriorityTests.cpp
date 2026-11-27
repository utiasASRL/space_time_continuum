#include <algorithm>
#include <thread>

#include <gtest/gtest.h>

#include <spacetime/interface/standalone/ThreadPriority.hpp>

namespace Spacetime::Interface::Standalone
{
    namespace
    {
        /// Runs `fn` on a fresh thread (so lowering it leaves the test's thread, and the threads it creates later, alone).
        template <typename Fn>
        void onFreshThread(Fn fn)
        {
            std::thread thread(fn);
            thread.join();
        }
    } // namespace

    TEST(ThreadPriority, LowersOnlyTheCallingThread)
    {
        const int main_priority = currentThreadPriority();
        int before = 0, after = 0;
        bool lowered = false;
        onFreshThread([&]()
                      {
            before = currentThreadPriority();
            lowered = lowerCurrentThreadPriority(3);
            after = currentThreadPriority(); });

        EXPECT_TRUE(lowered);
        EXPECT_LT(after, before);
#if !defined(__APPLE__) && !defined(_WIN32)
        EXPECT_EQ(after, std::max(before - 3, -19)); // Linux: minus the nice value, which is capped at 19
#endif
        EXPECT_EQ(currentThreadPriority(), main_priority);
    }

    TEST(ThreadPriority, NonPositiveDeltaChangesNothing)
    {
        onFreshThread([]()
                      {
            const int before = currentThreadPriority();
            EXPECT_FALSE(lowerCurrentThreadPriority(0));
            EXPECT_EQ(currentThreadPriority(), before); });
    }

    // Restored on destruction where the OS allows: macOS and Windows always; Linux only with the
    // privilege to lower a nice value again (CAP_SYS_NICE or RLIMIT_NICE), which restore() reports.
    TEST(ThreadPriority, ScopedLoweringIsRestoredWhereTheOsAllows)
    {
        onFreshThread([]()
                      {
            const int before = currentThreadPriority();
            int during = before;
            bool restored = false;
            {
                ScopedLowerThreadPriority lowered(10);
                EXPECT_TRUE(lowered.lowered());
                during = currentThreadPriority();
                restored = lowered.restore();
                EXPECT_EQ(lowered.restore(), restored); // only the first call does anything
            }
            EXPECT_LT(during, before);
#if defined(__APPLE__) || defined(_WIN32)
            EXPECT_TRUE(restored);
#endif
            EXPECT_EQ(currentThreadPriority() == before, restored); });
    }

    // Raised above a default thread where that needs no privilege (macOS, Windows); left alone on Linux.
    TEST(ThreadPriority, EstimatorThreadIsRaisedWhereTheOsAllows)
    {
        int default_priority = 0, estimator_priority = 0;
        onFreshThread([&]()
                      { default_priority = currentThreadPriority(); });
        onFreshThread([&]()
                      {
            prioritizeEstimatorThread();
            estimator_priority = currentThreadPriority(); });
#if defined(__APPLE__) || defined(_WIN32)
        EXPECT_GT(estimator_priority, default_priority);
#else
        EXPECT_EQ(estimator_priority, default_priority);
#endif
    }

    // The estimator stays above a lowered thread (e.g. the visualizer's GUI thread), on every OS.
    TEST(ThreadPriority, EstimatorThreadIsAboveALoweredThread)
    {
        int lowered_priority = 0, estimator_priority = 0;
        onFreshThread([&]()
                      {
            lowerCurrentThreadPriority(10);
            lowered_priority = currentThreadPriority(); });
        onFreshThread([&]()
                      {
            prioritizeEstimatorThread();
            estimator_priority = currentThreadPriority(); });
        EXPECT_GT(estimator_priority, lowered_priority);
    }

    TEST(ThreadPriority, TimerResolutionScopeIsHarmless)
    {
        ScopedTimerResolution resolution; // Windows: 1 ms timers while alive; elsewhere nothing
        SUCCEED();
    }
} // namespace Spacetime::Interface::Standalone
