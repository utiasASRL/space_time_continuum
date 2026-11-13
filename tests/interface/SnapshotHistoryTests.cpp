#include <atomic>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include <spacetime/interface/SnapshotHistory.hpp>

namespace Spacetime::Interface
{
    namespace
    {
        std::shared_ptr<const EstimatorSnapshot> snapshotAt(double time)
        {
            return std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{time, gtsam::Values(), nullptr});
        }
    } // namespace

    TEST(SnapshotHistory, EmptyHistoryHasNothing)
    {
        SnapshotHistory history;
        EXPECT_EQ(history.size(), 0u);
        EXPECT_EQ(history.at(0.0), nullptr);
        EXPECT_EQ(history.latest(), nullptr);
    }

    TEST(SnapshotHistory, NothingBeforeTheFirstSnapshot)
    {
        SnapshotHistory history;
        history.record(snapshotAt(1.0));
        EXPECT_EQ(history.at(0.5), nullptr);
    }

    TEST(SnapshotHistory, ReturnsTheLatestSnapshotAtOrBeforeTheTime)
    {
        SnapshotHistory history;
        const auto first = snapshotAt(0.0), second = snapshotAt(0.1), third = snapshotAt(0.2);
        history.record(first);
        history.record(second);
        history.record(third);

        EXPECT_EQ(history.size(), 3u);
        EXPECT_EQ(history.at(0.0), first);   // exact
        EXPECT_EQ(history.at(0.15), second); // in between
        EXPECT_EQ(history.at(0.2), third);   // exact, last
        EXPECT_EQ(history.at(10.0), third);  // after the end
        EXPECT_EQ(history.latest(), third);
    }

    TEST(SnapshotHistory, IgnoresNullSnapshots)
    {
        SnapshotHistory history;
        history.record(nullptr);
        EXPECT_EQ(history.size(), 0u);
    }

    TEST(SnapshotHistory, ConcurrentRecordAndRead)
    {
        SnapshotHistory history;
        std::atomic<bool> done{false};
        std::thread writer([&]()
                           {
            for (int i = 0; i < 2000; i++)
                history.record(snapshotAt(0.01 * i));
            done = true; });

        while (!done)
        {
            const auto snapshot = history.at(5.0);
            if (snapshot)
                EXPECT_LE(snapshot->time, 5.0);
        }
        writer.join();
        EXPECT_EQ(history.size(), 2000u);
        EXPECT_DOUBLE_EQ(history.at(5.005)->time, 0.01 * 500);
    }
} // namespace Spacetime::Interface
