#include <chrono>
#include <cmath>
#include <memory>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/FileDataloader.hpp>

namespace Spacetime::Standalone
{
    namespace
    {
        using Types::MeasurementQueue;
        using Types::pMeasurementQueueType;

        std::shared_ptr<Types::MeasurementBase> makeMeasurement(double time)
        {
            return std::make_shared<Types::MeasurementBase>(0.0, time, Eigen::Matrix4d::Identity(),
                                                            Eigen::MatrixXd::Identity(6, 6), Eigen::MatrixXi::Ones(6, 1));
        }

        pMeasurementQueueType makeQueue(std::initializer_list<double> times)
        {
            auto queue = std::make_shared<MeasurementQueue>();
            for (double t : times)
                queue->addMeasurement(makeMeasurement(t));
            return queue;
        }

        std::vector<double> timesIn(const MeasurementQueue &queue)
        {
            std::vector<double> times;
            for (const auto &m : queue.peekAll())
                times.push_back(m->time);
            return times;
        }

        std::shared_ptr<SteppedClock> steppedClock()
        {
            auto clock = std::make_shared<SteppedClock>();
            clock->start(0.0);
            return clock;
        }

        double secondsSince(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } // namespace

    TEST(FileDataloader, WithholdsTheQueuesOnConstruction)
    {
        auto a = makeQueue({0.0, 0.2}), b = makeQueue({0.1});
        FileDataloader loader({a, b}, steppedClock());
        EXPECT_TRUE(a->empty());
        EXPECT_TRUE(b->empty());
        EXPECT_FALSE(loader.deliveredThrough(0.0));
    }

    TEST(FileDataloader, LastTimeIsTheLatestWithheldStamp)
    {
        auto a = makeQueue({0.0, 0.2}), b = makeQueue({0.5});
        FileDataloader loader({a, b}, steppedClock());
        ASSERT_TRUE(loader.lastTime().has_value());
        EXPECT_DOUBLE_EQ(*loader.lastTime(), 0.5);
    }

    // A recording with nothing in it is bounded (unlike a live source), and ends before any time.
    TEST(FileDataloader, EmptyQueuesEndBeforeAnyTimeAndFinishImmediately)
    {
        auto a = makeQueue({});
        FileDataloader loader({a}, steppedClock());
        ASSERT_TRUE(loader.lastTime().has_value());
        EXPECT_TRUE(std::isinf(*loader.lastTime()) && *loader.lastTime() < 0.0);
        loader.run(std::stop_token());
        EXPECT_TRUE(loader.deliveredThrough(1e9));
        EXPECT_TRUE(a->empty());
    }

    TEST(FileDataloader, OfflineRunDeliversEverythingAndFinishes)
    {
        auto a = makeQueue({0.0, 0.2}), b = makeQueue({0.1, 0.3});
        FileDataloader loader({a, b}, steppedClock());
        loader.run(std::stop_token());

        EXPECT_EQ(timesIn(*a), (std::vector<double>{0.0, 0.2}));
        EXPECT_EQ(timesIn(*b), (std::vector<double>{0.1, 0.3}));
        EXPECT_TRUE(loader.deliveredThrough(1e9));
    }

    // Measurements are delivered in global time order across every queue, and the
    // watermark only claims a time once every measurement stamped at it is delivered.
    TEST(FileDataloader, DeliversInTimeOrderAndAdvancesTheWatermarkBehindIt)
    {
        auto a = makeQueue({0.0, 0.2}), b = makeQueue({0.1, 0.2});
        FileDataloader loader({a, b}, steppedClock());

        std::vector<std::pair<double, bool>> deliveries; // (time, deliveredThrough(time) when it arrived)
        std::vector<bool> previous_complete;             // deliveredThrough(time - 0.05) when it arrived
        for (const auto &queue : {a, b})
            queue->addMeasurementCallback([&](const MeasurementQueue::pMeasurementType &m)
                                          {
                deliveries.emplace_back(m->time, loader.deliveredThrough(m->time));
                previous_complete.push_back(m->time < 0.05 || loader.deliveredThrough(m->time - 0.05)); });

        loader.run(std::stop_token());

        ASSERT_EQ(deliveries.size(), 4u);
        EXPECT_DOUBLE_EQ(deliveries[0].first, 0.0);
        EXPECT_DOUBLE_EQ(deliveries[1].first, 0.1);
        EXPECT_DOUBLE_EQ(deliveries[2].first, 0.2);
        EXPECT_DOUBLE_EQ(deliveries[3].first, 0.2);
        for (const auto &[time, complete] : deliveries)
            EXPECT_FALSE(complete) << "time " << time << " was claimed before its delivery finished";
        for (bool complete : previous_complete)
            EXPECT_TRUE(complete);
    }

    TEST(FileDataloader, EachMeasurementNotifiesCallbacksOnce)
    {
        auto a = makeQueue({0.0, 0.1});
        FileDataloader loader({a}, steppedClock());
        int calls = 0;
        a->addMeasurementCallback([&](const MeasurementQueue::pMeasurementType &)
                                  { calls++; });
        EXPECT_EQ(calls, 0); // withheld, so nothing is replayed on registration
        loader.run(std::stop_token());
        EXPECT_EQ(calls, 2);
    }

    TEST(FileDataloader, DestructionRestoresUndeliveredMeasurements)
    {
        auto a = makeQueue({0.0, 0.1, 0.2});
        {
            FileDataloader loader({a}, steppedClock());
        }
        EXPECT_EQ(timesIn(*a), (std::vector<double>{0.0, 0.1, 0.2}));
    }

    TEST(FileDataloader, RealTimeRunIsPacedByTheClock)
    {
        auto a = makeQueue({0.0, 0.05, 0.1});
        auto clock = std::make_shared<WallClock>();
        FileDataloader loader({a}, clock);

        clock->start(0.0);
        const auto start = std::chrono::steady_clock::now();
        loader.run(std::stop_token());
        EXPECT_GE(secondsSince(start), 0.09);
        EXPECT_EQ(a->size(), 3u);
        EXPECT_TRUE(loader.deliveredThrough(1e9));
    }

    TEST(FileDataloader, RealTimeDeliversPastMeasurementsImmediately)
    {
        auto a = makeQueue({0.0, 0.5, 1.0});
        auto clock = std::make_shared<WallClock>();
        FileDataloader loader({a}, clock);

        clock->start(5.0); // every measurement is already in the past
        const auto start = std::chrono::steady_clock::now();
        loader.run(std::stop_token());
        EXPECT_LT(secondsSince(start), 0.05);
        EXPECT_EQ(a->size(), 3u);
    }

    // As the old OfflineInterface did, a stopped run hands whatever it did not deliver back
    // to its queue, so no data is lost (the watermark does not claim it, though).
    TEST(FileDataloader, StopEndsARealTimeRunAndRestoresTheRest)
    {
        auto a = makeQueue({0.0, 60.0});
        auto clock = std::make_shared<WallClock>();
        FileDataloader loader({a}, clock);
        clock->start(0.0);

        std::vector<double> delivered;
        a->addMeasurementCallback([&](const MeasurementQueue::pMeasurementType &m)
                                  { delivered.push_back(m->time); });

        std::stop_source stop;
        std::thread runner([&]()
                           { loader.run(stop.get_token()); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto stop_time = std::chrono::steady_clock::now();
        stop.request_stop();
        runner.join();
        EXPECT_LT(secondsSince(stop_time), 1.0);

        EXPECT_EQ(timesIn(*a), (std::vector<double>{0.0, 60.0}));
        EXPECT_FALSE(loader.deliveredThrough(60.0));
        EXPECT_EQ(delivered, (std::vector<double>{0.0, 60.0})); // restoring is an ordinary add
    }

    TEST(FileDataloader, StoppedBeforeRunningRestoresEverything)
    {
        auto a = makeQueue({0.0, 0.1});
        FileDataloader loader({a}, steppedClock());
        std::stop_source stop;
        stop.request_stop();
        loader.run(stop.get_token());
        EXPECT_EQ(timesIn(*a), (std::vector<double>{0.0, 0.1}));
    }
} // namespace Spacetime::Standalone
