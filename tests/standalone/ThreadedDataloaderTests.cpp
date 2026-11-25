#include <chrono>
#include <stop_token>
#include <thread>

#include <gtest/gtest.h>

#include <spacetime/interface/standalone/ThreadedDataloader.hpp>

namespace Spacetime::Interface::Standalone
{
    namespace
    {
        /// Exposes the protected watermark setters, so the tests can drive them directly.
        struct ManualDataloader : public ThreadedDataloader
        {
            void run(std::stop_token) override {}
            using ThreadedDataloader::finish;
            using ThreadedDataloader::setNextTime;
        };

        double secondsSince(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } // namespace

    TEST(ThreadedDataloader, WaitReturnsImmediatelyWhenAlreadyDelivered)
    {
        ManualDataloader loader;
        loader.setNextTime(1.0);
        const auto start = std::chrono::steady_clock::now();
        EXPECT_TRUE(loader.waitUntilDeliveredThrough(0.5, std::stop_token()));
        EXPECT_LT(secondsSince(start), 0.01);
    }

    TEST(ThreadedDataloader, WaitWakesWhenTheWatermarkPassesTheTime)
    {
        ManualDataloader loader;
        std::thread producer([&loader]()
                             {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            loader.setNextTime(0.2); // not enough yet
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            loader.setNextTime(0.6); });

        EXPECT_TRUE(loader.waitUntilDeliveredThrough(0.5, std::stop_token()));
        EXPECT_TRUE(loader.deliveredThrough(0.5));
        producer.join();
    }

    TEST(ThreadedDataloader, WaitWakesOnFinish)
    {
        ManualDataloader loader;
        std::thread producer([&loader]()
                             {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            loader.finish(); });
        EXPECT_TRUE(loader.waitUntilDeliveredThrough(100.0, std::stop_token()));
        producer.join();
    }

    TEST(ThreadedDataloader, WaitReturnsFalseOnStopRequest)
    {
        ManualDataloader loader;
        std::stop_source stop;
        std::thread stopper([&stop]()
                            {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            stop.request_stop(); });

        const auto start = std::chrono::steady_clock::now();
        EXPECT_FALSE(loader.waitUntilDeliveredThrough(1.0, stop.get_token()));
        EXPECT_LT(secondsSince(start), 1.0);
        stopper.join();
    }
} // namespace Spacetime::Interface::Standalone
