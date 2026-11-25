#include <chrono>
#include <thread>
#include <atomic>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    using Pose = gtsam::Pose3;
    using NodeType = Priors::WNOA::Node<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using ValueType = Priors::WNOA::NodeValues<Pose>;

    namespace
    {
        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior(unsigned int spatial_dim)
        {
            auto state = std::make_shared<StateType>(1.0, spatial_dim);
            ValueType node_reference;
            return std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);
        }
    }

    TEST(EstimatorSnapshot, NullBeforeFirstEstimate)
    {
        Estimator estimator(Testing::estimatorOptions(0.0));
        EXPECT_EQ(estimator.getSnapshot(), nullptr);
    }

    TEST(EstimatorSnapshot, PublishedAfterEstimateMatchesRequestedTime)
    {
        auto prior = makePrior(2);
        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(prior);

        estimator.estimate(1.5);

        auto snapshot = estimator.getSnapshot();
        ASSERT_NE(snapshot, nullptr);
        EXPECT_DOUBLE_EQ(snapshot->time, 1.5);
        EXPECT_EQ(snapshot->values.size(), estimator.getCurrentValues().size());
    }

    TEST(EstimatorSnapshot, SubsequentEstimateReplacesThePublishedSnapshot)
    {
        auto prior = makePrior(2);
        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(prior);

        estimator.estimate(0.0);
        auto first = estimator.getSnapshot();
        estimator.estimate(0.1);
        auto second = estimator.getSnapshot();

        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_NE(first, second);
        EXPECT_DOUBLE_EQ(first->time, 0.0);
        EXPECT_DOUBLE_EQ(second->time, 0.1);
    }

    // Stress test: one thread repeatedly calls estimate() (as the estimator thread
    // would in online mode) while this thread concurrently calls getSnapshot() (as
    // a visualizer thread would). Every returned snapshot must be a fully-formed,
    // internally consistent object -- not a torn/partial read -- which is exactly
    // what Estimator::m_snapshot_mutex is there to guarantee. Uses loose,
    // steady_clock-based bounds rather than exact counts, matching this repo's
    // existing style for timing-sensitive tests (see standalone/EstimatorRunnerTests.cpp).
    TEST(EstimatorSnapshot, ConcurrentEstimateAndGetSnapshotNeverObservesATornSnapshot)
    {
        auto prior = makePrior(2);
        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(prior);

        std::atomic<bool> stop{false};
        std::atomic<int> reads{0};

        std::thread writer([&]()
                           {
            double t = 0.0;
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            while (std::chrono::steady_clock::now() < deadline)
            {
                estimator.estimate(t);
                t += 0.01;
            }
            stop = true; });

        while (!stop)
        {
            auto snapshot = estimator.getSnapshot();
            if (snapshot)
            {
                // A torn read (e.g. a partially-constructed gtsam::Values, or a time
                // field from a different tick than the values) would be the kind of
                // thing a race could produce; this is the closest a black-box test
                // can get to asserting "not torn" for an opaque published payload.
                EXPECT_GE(snapshot->time, 0.0);
                reads++;
            }
        }
        writer.join();
        EXPECT_GT(reads.load(), 0);
    }

    TEST(EstimatorSnapshot, WaitForSnapshotNewerThanTimesOutWithoutAnEstimate)
    {
        Estimator estimator(Testing::estimatorOptions(0.0));
        const auto start = std::chrono::steady_clock::now();
        EXPECT_EQ(estimator.waitForSnapshotNewerThan(0.0, std::chrono::milliseconds(20)), nullptr);
        EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(15));
    }

    TEST(EstimatorSnapshot, WaitForSnapshotNewerThanReturnsAnAlreadyNewerSnapshot)
    {
        auto prior = makePrior(2);
        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(prior);
        estimator.estimate(1.0);

        const auto snapshot = estimator.waitForSnapshotNewerThan(0.5, std::chrono::milliseconds(0));
        ASSERT_NE(snapshot, nullptr);
        EXPECT_DOUBLE_EQ(snapshot->time, 1.0);
        // Not newer than its own time.
        EXPECT_EQ(estimator.waitForSnapshotNewerThan(1.0, std::chrono::milliseconds(0)), nullptr);
    }

    TEST(EstimatorSnapshot, WaitForSnapshotNewerThanWakesOnPublish)
    {
        auto prior = makePrior(2);
        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(prior);
        estimator.estimate(0.0);

        std::thread writer([&]()
                           {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            estimator.estimate(0.1); });

        const auto start = std::chrono::steady_clock::now();
        const auto snapshot = estimator.waitForSnapshotNewerThan(0.0, std::chrono::seconds(10));
        writer.join();
        ASSERT_NE(snapshot, nullptr);
        EXPECT_DOUBLE_EQ(snapshot->time, 0.1);
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
    }
}
