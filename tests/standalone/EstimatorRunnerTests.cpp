#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/EstimatorRunner.hpp>
#include <spacetime/interface/standalone/FileDataloader.hpp>
#include <spacetime/interface/standalone/ThreadedDataloader.hpp>

namespace Spacetime::Standalone
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;

        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior()
        {
            auto state = std::make_shared<StateType>(1.0, 2);
            return std::make_shared<Priors::WNOA::Prior<Pose>>(state, ValueType());
        }

        std::shared_ptr<Types::MeasurementBase> makeMeasurement(double time, double arclength = 0.0, double x = 0.0)
        {
            Eigen::Matrix4d value = Eigen::Matrix4d::Identity();
            value(0, 3) = x;
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
            return std::make_shared<Types::MeasurementBase>(arclength, time, value, covariance, Eigen::MatrixXi::Ones(6, 1));
        }

        std::shared_ptr<Sensors::PoseSensor<ValueType>> addPoseSensor(Estimator &estimator, const std::shared_ptr<Priors::WNOA::Prior<Pose>> &prior)
        {
            const Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
            auto sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
            estimator.addSensor(sensor);
            return sensor;
        }

        RunnerOptions options(double start_time, double end_time, double period)
        {
            RunnerOptions opts;
            opts.start_time = start_time;
            opts.end_time = end_time;
            opts.period = period;
            return opts;
        }

        double secondsSince(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }

        /// Delivers its withheld measurements in time order, but slowly, so an estimator
        /// that did not wait for it would run ahead of the data.
        struct SlowDataloader : public ThreadedDataloader
        {
            SlowDataloader(const Types::pMeasurementQueueType &queue) : m_queue(queue)
            {
                for (const auto &m : queue->takeAll())
                    m_pending.push_back(m);
            }

            void run(std::stop_token stop) override
            {
                while (!m_pending.empty() && !stop.stop_requested())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    m_queue->addMeasurement(m_pending.front());
                    m_pending.pop_front();
                    setNextTime(m_pending.empty() ? std::numeric_limits<double>::infinity() : m_pending.front()->time);
                }
                finish();
            }

            std::optional<double> lastTime() const override { return 1.0; }

            Types::pMeasurementQueueType m_queue;
            std::deque<Types::MeasurementQueue::pMeasurementType> m_pending;
        };

        /// Never delivers anything until stopped, like a live source with no data.
        struct SilentDataloader : public ThreadedDataloader
        {
            void run(std::stop_token stop) override
            {
                while (!stop.stop_requested())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        };

        struct ThrowingDataloader : public ThreadedDataloader
        {
            void run(std::stop_token) override { throw std::runtime_error("dataloader failure"); }
        };

        /// The pose problem used by the determinism test: two sensors with distinct, noisy-ish data.
        struct TwoSensorProblem
        {
            std::shared_ptr<Priors::WNOA::Prior<Pose>> prior = makePrior();
            Estimator estimator = Estimator(/*smootherLag=*/0.35);
            std::shared_ptr<Sensors::PoseSensor<ValueType>> a, b;

            TwoSensorProblem()
            {
                estimator.addPrior(prior);
                a = addPoseSensor(estimator, prior);
                b = addPoseSensor(estimator, prior);
                for (int k = 0; k <= 10; k++)
                {
                    a->addMeasurement(makeMeasurement(0.1 * k, 0.0, 0.01 * k));
                    b->addMeasurement(makeMeasurement(0.1 * k + 0.05, 1.0, 1.0 + 0.02 * k));
                }
            }
        };
    } // namespace

    // ---- Behaviour ported from the old OfflineInterface tests ----

    TEST(EstimatorRunner, EndTimeCutoffPreservesUndeliveredMeasurements)
    {
        auto prior = makePrior();
        Estimator estimator;
        estimator.addPrior(prior);
        auto sensor = addPoseSensor(estimator, prior);
        sensor->addMeasurement(makeMeasurement(0.0));
        sensor->addMeasurement(makeMeasurement(0.5));

        auto clock = std::make_shared<SteppedClock>();
        EstimatorRunner runner(estimator, clock, options(0.0, 0.0, 0.1)); // only t = 0.0 becomes due
        runner.addDataloader(std::make_shared<FileDataloader>(estimator.getMeasurementQueues(), clock));
        runner.run();

        const auto left = sensor->peekAll();
        ASSERT_EQ(left.size(), 1u);
        EXPECT_DOUBLE_EQ(left.front()->time, 0.5);
    }

    TEST(EstimatorRunner, RealTimeIsPacedByPeriod)
    {
        Estimator estimator; // no priors/sensors: isolates the loop's own pacing
        const double period = 0.02, end_time = 0.2;
        EstimatorRunner runner(estimator, std::make_shared<WallClock>(), options(0.0, end_time, period));
        int calls = 0;
        runner.setTickCallback([&](double)
                               { calls++; });

        const auto start = std::chrono::steady_clock::now();
        runner.run();
        const double elapsed = secondsSince(start);

        // ~end_time / period + 1 = 11 calls. Loose bounds tolerate scheduler jitter.
        EXPECT_GE(calls, 5);
        EXPECT_LE(calls, 15);
        EXPECT_GE(elapsed, end_time * 0.8);
    }

    TEST(EstimatorRunner, RealTimeOverrunDoesNotBurstToCatchUp)
    {
        Estimator estimator;
        const double period = 0.02, end_time = 0.3;
        EstimatorRunner runner(estimator, std::make_shared<WallClock>(), options(0.0, end_time, period));
        int calls = 0;
        runner.setTickCallback([&](double)
                               {
            if (calls++ == 0)
                std::this_thread::sleep_for(std::chrono::duration<double>(0.2)); }); // overrun ten periods
        testing::internal::CaptureStdout();
        runner.run();
        const std::string output = testing::internal::GetCapturedStdout();

        // 1 slow call, 1 immediate call, then ~5 paced ones. Catching up would make ~16.
        EXPECT_LE(calls, 11);
        EXPECT_GE(calls, 3);
        EXPECT_NE(output.find("estimation is being called too slowly"), std::string::npos) << output;
    }

    TEST(EstimatorRunner, OfflineNodesAreSpacedExactlyOnePeriod)
    {
        const double period = 0.1;
        auto state = std::make_shared<StateType>(1.0, 2);
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, ValueType());

        Estimator estimator(/*smootherLag=*/100.0);
        estimator.addPrior(prior);
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), options(0.0, 3.0, period));

        testing::internal::CaptureStdout();
        runner.run();
        const std::string output = testing::internal::GetCapturedStdout();
        EXPECT_EQ(output.find("called too slowly"), std::string::npos) << output;

        const auto nodes = state->snapshotEstimationNodes();
        const std::size_t rows = nodes.size() / 2;
        ASSERT_EQ(rows, 31u); // t = 0.0, 0.1, ..., 3.0
        for (std::size_t k = 0; k < rows; ++k)
            EXPECT_DOUBLE_EQ(nodes[k * 2]->time, 0.0 + static_cast<double>(k) * period) << "row " << k;
    }

    TEST(EstimatorRunner, QuasiStaticOfflineRowsAreAtTheUniqueMeasurementTimes)
    {
        auto prior = makePrior();
        EstimatorOptions estimator_options;
        estimator_options.smootherLag = Estimator::BATCH_LAG;
        estimator_options.quasi_static = true;
        Estimator estimator(estimator_options);
        estimator.addPrior(prior);
        auto a = addPoseSensor(estimator, prior);
        auto b = addPoseSensor(estimator, prior);
        for (const double t : {0.013, 0.05, 0.27, 0.6})
            a->addMeasurement(makeMeasurement(t));
        b->addMeasurement(makeMeasurement(0.05, 1.0));
        b->addMeasurement(makeMeasurement(0.31, 1.0));

        auto clock = std::make_shared<SteppedClock>();
        EstimatorRunner runner(estimator, clock, options(0.0, 0.5, 0.1));
        runner.addDataloader(std::make_shared<FileDataloader>(estimator.getMeasurementQueues(), clock));
        runner.run();

        const auto nodes = prior->state()->snapshotEstimationNodes();
        std::vector<double> row_times;
        for (std::size_t i = 0; i < nodes.size(); i += 2)
            row_times.push_back(nodes[i]->time);
        EXPECT_EQ(row_times, (std::vector<double>{0.013, 0.05, 0.27, 0.31})); // 0.6 is past the end time
    }

    // The old OfflineInterface replayed every model's queue (sensors and actuators alike).
    TEST(EstimatorRunner, ReplaysEveryMeasurementQueue)
    {
        struct QueuedModel : public Types::Model, public Types::MeasurementQueue
        {
            std::vector<double> seen_at;
            Factors getNewFactors(double time, gtsam::Values &) override
            {
                for (const auto &m : drainUpTo(time))
                {
                    EXPECT_LE(m->time, time);
                    seen_at.push_back(m->time);
                }
                return {};
            }
        };
        auto model = std::make_shared<QueuedModel>();
        for (double t : {0.0, 0.2, 0.4})
            model->addMeasurement(makeMeasurement(t));

        Estimator estimator;
        estimator.addPrior(makePrior());
        estimator.addModel(model);

        auto clock = std::make_shared<SteppedClock>();
        EstimatorRunner runner(estimator, clock, options(0.0, -1.0, 0.2)); // until the last measurement
        runner.addDataloader(std::make_shared<FileDataloader>(estimator.getMeasurementQueues(), clock));
        runner.run();
        EXPECT_EQ(model->seen_at, (std::vector<double>{0.0, 0.2, 0.4}));
    }

    // ---- New behaviour ----

    TEST(EstimatorRunner, EndTimeDefaultsToTheLatestDataloaderTime)
    {
        Estimator estimator;
        auto queue = std::make_shared<Types::MeasurementQueue>();
        queue->addMeasurement(makeMeasurement(0.3));
        auto other = std::make_shared<Types::MeasurementQueue>();
        other->addMeasurement(makeMeasurement(0.5));

        auto clock = std::make_shared<SteppedClock>();
        EstimatorRunner runner(estimator, clock, options(0.0, -1.0, 0.1));
        runner.addDataloader(std::make_shared<FileDataloader>(std::vector<Types::pMeasurementQueueType>{queue}, clock));
        runner.addDataloader(std::make_shared<FileDataloader>(std::vector<Types::pMeasurementQueueType>{other}, clock));
        std::vector<double> ticks;
        runner.setTickCallback([&](double t)
                               { ticks.push_back(t); });
        runner.run();

        EXPECT_DOUBLE_EQ(runner.endTime(), 0.5);
        ASSERT_EQ(ticks.size(), 6u); // 0.0 ... 0.5
        EXPECT_DOUBLE_EQ(ticks.back(), 0.5);
    }

    // A live source (no known last time) with a negative end time runs until stopped.
    TEST(EstimatorRunner, EndTimeIsUnboundedWithALiveDataloader)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<WallClock>(), options(0.0, -1.0, 0.01));
        runner.addDataloader(std::make_shared<SilentDataloader>());
        runner.start();
        EXPECT_TRUE(std::isinf(runner.endTime()));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_FALSE(runner.finished());
        runner.requestStop();
        runner.join();
        EXPECT_TRUE(runner.finished());
    }

    // The offline estimate does not depend on how the data is split across dataloaders,
    // nor on how fast they deliver: the estimator waits for each one's watermark.
    TEST(EstimatorRunner, OfflineResultIsIndependentOfTheDataloaders)
    {
        auto runWith = [](auto add_dataloaders)
        {
            auto problem = std::make_unique<TwoSensorProblem>();
            auto clock = std::make_shared<SteppedClock>();
            EstimatorRunner runner(problem->estimator, clock, options(0.0, 1.0, 0.1));
            add_dataloaders(runner, *problem, clock);
            runner.run();
            return problem;
        };

        auto single = runWith([](EstimatorRunner &runner, TwoSensorProblem &p, const std::shared_ptr<SteppedClock> &clock)
                              { runner.addDataloader(std::make_shared<FileDataloader>(p.estimator.getMeasurementQueues(), clock)); });
        auto split = runWith([](EstimatorRunner &runner, TwoSensorProblem &p, const std::shared_ptr<SteppedClock> &clock)
                             {
            runner.addDataloader(std::make_shared<FileDataloader>(std::vector<Types::pMeasurementQueueType>{p.a}, clock));
            runner.addDataloader(std::make_shared<FileDataloader>(std::vector<Types::pMeasurementQueueType>{p.b}, clock)); });
        auto slow = runWith([](EstimatorRunner &runner, TwoSensorProblem &p, const std::shared_ptr<SteppedClock> &clock)
                            {
            runner.addDataloader(std::make_shared<SlowDataloader>(p.a));
            runner.addDataloader(std::make_shared<FileDataloader>(std::vector<Types::pMeasurementQueueType>{p.b}, clock)); });

        const gtsam::Values &reference = single->estimator.getCurrentValues();
        ASSERT_GT(reference.size(), 0u);
        EXPECT_TRUE(reference.equals(split->estimator.getCurrentValues(), 1e-12));
        EXPECT_TRUE(reference.equals(slow->estimator.getCurrentValues(), 1e-12));
    }

    // Real time never waits on a dataloader, so a silent live source cannot stall it.
    TEST(EstimatorRunner, RealTimeDoesNotWaitForDataloaders)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<WallClock>(), options(0.0, 0.05, 0.01));
        runner.addDataloader(std::make_shared<SilentDataloader>());
        int calls = 0;
        runner.setTickCallback([&](double)
                               { calls++; });
        const auto start = std::chrono::steady_clock::now();
        runner.run(); // also stops the silent dataloader once the estimator reaches the end
        EXPECT_LT(secondsSince(start), 1.0);
        EXPECT_GE(calls, 3);
    }

    // ...whereas offline, synchronizing with it would block until stopped.
    TEST(EstimatorRunner, OfflineWaitsForDataloadersUntilStopped)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), options(0.0, 1.0, 0.1));
        runner.addDataloader(std::make_shared<SilentDataloader>());
        std::atomic<int> calls{0};
        runner.setTickCallback([&](double)
                               { calls++; });
        runner.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_EQ(calls.load(), 0);
        runner.requestStop();
        runner.join();
        EXPECT_EQ(calls.load(), 0);
    }

    TEST(EstimatorRunner, SynchronizationCanBeOverridden)
    {
        Estimator estimator;
        RunnerOptions opts = options(0.0, 0.2, 0.1);
        opts.synchronize = false;
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), opts);
        runner.addDataloader(std::make_shared<SilentDataloader>());
        int calls = 0;
        runner.setTickCallback([&](double)
                               { calls++; });
        runner.run();
        EXPECT_EQ(calls, 3);
    }

    TEST(EstimatorRunner, DataloaderExceptionIsRethrownByJoin)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), options(0.0, 1.0, 0.1));
        runner.addDataloader(std::make_shared<ThrowingDataloader>());
        runner.start();
        EXPECT_THROW(runner.join(), std::runtime_error);
    }

    TEST(EstimatorRunner, TickCallbackExceptionIsRethrownByJoin)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), options(0.0, 1.0, 0.1));
        runner.setTickCallback([](double)
                               { throw std::runtime_error("tick failure"); });
        runner.start();
        EXPECT_THROW(runner.join(), std::runtime_error);
    }

    TEST(EstimatorRunner, EstimatesOnItsOwnThread)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), options(0.0, 0.0, 0.1));
        std::thread::id tick_thread;
        runner.setTickCallback([&](double)
                               { tick_thread = std::this_thread::get_id(); });
        runner.run();
        EXPECT_NE(tick_thread, std::thread::id());
        EXPECT_NE(tick_thread, std::this_thread::get_id());
    }

    TEST(EstimatorRunner, StartsTheClockAtTheStartTime)
    {
        Estimator estimator;
        auto clock = std::make_shared<SteppedClock>();
        EstimatorRunner runner(estimator, clock, options(2.0, 2.2, 0.1));
        std::vector<double> clock_at_tick;
        runner.setTickCallback([&](double)
                               { clock_at_tick.push_back(clock->now()); });
        runner.run();
        ASSERT_EQ(clock_at_tick.size(), 3u);
        EXPECT_DOUBLE_EQ(clock_at_tick.front(), 2.0);
    }

    TEST(EstimatorRunner, DestructorStopsARunningRunner)
    {
        Estimator estimator;
        const auto start = std::chrono::steady_clock::now();
        {
            EstimatorRunner runner(estimator, std::make_shared<WallClock>(), options(0.0, 60.0, 0.5));
            runner.start();
        }
        EXPECT_LT(secondsSince(start), 1.0);
    }

    TEST(EstimatorRunner, CannotStartTwice)
    {
        Estimator estimator;
        EstimatorRunner runner(estimator, std::make_shared<SteppedClock>(), options(0.0, 0.0, 0.1));
        runner.start();
        EXPECT_THROW(runner.start(), std::logic_error);
        runner.join();
    }
} // namespace Spacetime::Standalone
