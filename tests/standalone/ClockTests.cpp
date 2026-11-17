#include <chrono>
#include <stop_token>
#include <thread>

#include <gtest/gtest.h>

#include <spacetime/interface/standalone/Clock.hpp>

namespace Spacetime::Standalone
{
    namespace
    {
        double secondsSince(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } // namespace

    // Offline, "sleeping" until a time just jumps the clock to it, without any wall-clock wait.
    TEST(SteppedClock, SleepUntilAdvancesInstantly)
    {
        SteppedClock clock;
        clock.start(0.0);
        const auto start = std::chrono::steady_clock::now();
        clock.sleepUntil(1000.0, std::stop_token());
        EXPECT_LT(secondsSince(start), 0.05);
        EXPECT_DOUBLE_EQ(clock.now(), 1000.0);
    }

    TEST(SteppedClock, NeverRunsBackwards)
    {
        SteppedClock clock;
        clock.start(1.0);
        clock.sleepUntil(0.5, std::stop_token());
        EXPECT_DOUBLE_EQ(clock.now(), 1.0);
    }

    TEST(WallClock, TracksElapsedTimeFromItsStart)
    {
        WallClock clock;
        clock.start(10.0);
        EXPECT_GE(clock.now(), 10.0);
        EXPECT_LT(clock.now(), 10.05);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_GE(clock.now(), 10.05);
        EXPECT_LT(clock.now(), 10.5);
    }

    TEST(WallClock, SleepUntilWaitsForTheTime)
    {
        WallClock clock;
        clock.start(1.0);
        const auto start = std::chrono::steady_clock::now();
        clock.sleepUntil(1.05, std::stop_token());
        EXPECT_GE(secondsSince(start), 0.045);
        EXPECT_GE(clock.now(), 1.05);
    }

    TEST(WallClock, SleepUntilAPastTimeReturnsImmediately)
    {
        WallClock clock;
        clock.start(1.0);
        const auto start = std::chrono::steady_clock::now();
        clock.sleepUntil(0.0, std::stop_token());
        EXPECT_LT(secondsSince(start), 0.01);
    }

    TEST(WallClock, SleepUntilReturnsPromptlyOnStopRequest)
    {
        WallClock clock;
        clock.start(0.0);
        std::stop_source stop;
        std::thread stopper([&stop]()
                            {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            stop.request_stop(); });

        const auto start = std::chrono::steady_clock::now();
        clock.sleepUntil(60.0, stop.get_token());
        stopper.join();
        EXPECT_LT(secondsSince(start), 1.0);
    }

    TEST(WallClock, SleepUntilWithAlreadyStoppedTokenReturnsImmediately)
    {
        WallClock clock;
        clock.start(0.0);
        std::stop_source stop;
        stop.request_stop();
        const auto start = std::chrono::steady_clock::now();
        clock.sleepUntil(60.0, stop.get_token());
        EXPECT_LT(secondsSince(start), 0.01);
    }

    // The visualizer reads the clock from its own thread while the estimator thread (re)starts it.
    TEST(WallClock, NowIsSafeToReadWhileStarting)
    {
        WallClock clock;
        std::thread reader([&clock]()
                           {
            for (int i = 0; i < 10000; i++)
                EXPECT_GE(clock.now(), 0.0); });
        for (int i = 0; i < 1000; i++)
            clock.start(static_cast<double>(i));
        reader.join();
    }
} // namespace Spacetime::Standalone
