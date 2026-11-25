#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <spacetime/estimator/types/MeasurementQueue.hpp>

namespace Spacetime
{
    namespace
    {
        std::shared_ptr<Types::MeasurementBase> makeMeasurement(double time)
        {
            return std::make_shared<Types::MeasurementBase>(0.0, time, Eigen::Matrix4d::Identity(),
                                                            Eigen::MatrixXd::Identity(6, 6), Eigen::MatrixXi::Ones(6, 1));
        }

        std::vector<double> timesOf(const std::vector<Types::MeasurementQueue::pMeasurementType> &measurements)
        {
            std::vector<double> times;
            for (const auto &m : measurements)
                times.push_back(m->time);
            return times;
        }
    } // namespace

    TEST(MeasurementQueueAccessors, EmptyQueue)
    {
        Types::MeasurementQueue queue;
        EXPECT_TRUE(queue.empty());
        EXPECT_EQ(queue.size(), 0u);
        EXPECT_FALSE(queue.lastTime().has_value());
        EXPECT_TRUE(queue.peekAll().empty());
        EXPECT_TRUE(queue.takeAll().empty());
    }

    TEST(MeasurementQueueAccessors, SizeAndLastTimeTrackTheQueue)
    {
        Types::MeasurementQueue queue;
        for (double t : {0.0, 0.1, 0.2})
            queue.addMeasurement(makeMeasurement(t));

        EXPECT_FALSE(queue.empty());
        EXPECT_EQ(queue.size(), 3u);
        ASSERT_TRUE(queue.lastTime().has_value());
        EXPECT_DOUBLE_EQ(*queue.lastTime(), 0.2);

        queue.drainUpTo(0.1);
        EXPECT_EQ(queue.size(), 1u);
        EXPECT_DOUBLE_EQ(*queue.lastTime(), 0.2);
    }

    TEST(MeasurementQueueAccessors, FrontTimeIsTheEarliestQueuedMeasurement)
    {
        Types::MeasurementQueue queue;
        EXPECT_FALSE(queue.frontTime().has_value());
        for (double t : {0.0, 0.1, 0.2})
            queue.addMeasurement(makeMeasurement(t));

        ASSERT_TRUE(queue.frontTime().has_value());
        EXPECT_DOUBLE_EQ(*queue.frontTime(), 0.0);
        queue.drainUpTo(0.1);
        EXPECT_DOUBLE_EQ(*queue.frontTime(), 0.2);
        EXPECT_EQ(queue.size(), 1u); // not consumed
    }

    TEST(MeasurementQueueAccessors, TakeAllEmptiesTheQueueInOrder)
    {
        Types::MeasurementQueue queue;
        for (double t : {0.0, 0.1, 0.2})
            queue.addMeasurement(makeMeasurement(t));

        EXPECT_EQ(timesOf(queue.takeAll()), (std::vector<double>{0.0, 0.1, 0.2}));
        EXPECT_TRUE(queue.empty());
        EXPECT_TRUE(queue.takeAll().empty());
    }

    // Taking measurements out does not notify callbacks, and adding them back notifies each once.
    TEST(MeasurementQueueAccessors, TakeAllDoesNotNotifyCallbacks)
    {
        Types::MeasurementQueue queue;
        queue.addMeasurement(makeMeasurement(0.0));
        const auto taken = queue.takeAll();

        int calls = 0;
        queue.addMeasurementCallback([&](const Types::MeasurementQueue::pMeasurementType &)
                                     { calls++; });
        EXPECT_EQ(calls, 0); // nothing left to replay

        for (const auto &m : taken)
            queue.addMeasurement(m);
        EXPECT_EQ(calls, 1);
    }

    // A callback that saw a measurement while it was queued is not called again when it is
    // redelivered; one registered while it was withheld is called on redelivery.
    TEST(MeasurementQueueAccessors, RedeliveryNotifiesOnlyCallbacksRegisteredSinceWithholding)
    {
        Types::MeasurementQueue queue;
        queue.addMeasurement(makeMeasurement(0.0));

        int early = 0, late = 0;
        queue.addMeasurementCallback([&](const Types::MeasurementQueue::pMeasurementType &)
                                     { early++; });
        EXPECT_EQ(early, 1); // replayed on registration

        const auto withheld = queue.withholdAll();
        ASSERT_EQ(withheld.size(), 1u);
        EXPECT_EQ(withheld.front().notified, 1u);
        EXPECT_TRUE(queue.empty());
        queue.addMeasurementCallback([&](const Types::MeasurementQueue::pMeasurementType &)
                                     { late++; });
        EXPECT_EQ(late, 0); // nothing queued to replay

        for (const auto &w : withheld)
            queue.redeliver(w);
        EXPECT_EQ(early, 1);
        EXPECT_EQ(late, 1);
        EXPECT_EQ(timesOf(queue.peekAll()), (std::vector<double>{0.0}));

        queue.addMeasurement(makeMeasurement(0.1)); // a new measurement reaches every callback
        EXPECT_EQ(early, 2);
        EXPECT_EQ(late, 2);
    }

    TEST(MeasurementQueueAccessors, ConcurrentProducersAndReaders)
    {
        Types::MeasurementQueue queue;
        constexpr int kProducers = 4, kPerProducer = 500;

        std::vector<std::thread> producers;
        for (int p = 0; p < kProducers; p++)
            producers.emplace_back([&queue]()
                                   {
                for (int i = 0; i < kPerProducer; i++)
                    queue.addMeasurement(makeMeasurement(static_cast<double>(i))); });

        // Concurrent reads must never crash or observe more than was added.
        std::thread reader([&queue]()
                           {
            for (int i = 0; i < 2000; i++)
            {
                EXPECT_LE(queue.size(), static_cast<std::size_t>(kProducers * kPerProducer));
                (void)queue.lastTime();
                (void)queue.empty();
            } });

        for (auto &producer : producers)
            producer.join();
        reader.join();
        EXPECT_EQ(queue.size(), static_cast<std::size_t>(kProducers * kPerProducer));
    }
} // namespace Spacetime
