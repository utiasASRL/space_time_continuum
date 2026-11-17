#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/Interfaces.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/EstimatorRunner.hpp>
#include <spacetime/interface/standalone/FileDataloader.hpp>

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

        std::shared_ptr<Types::MeasurementBase> makeMeasurement(double time, double x = 0.0)
        {
            Eigen::Matrix4d value = Eigen::Matrix4d::Identity();
            value(0, 3) = x;
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
            return std::make_shared<Types::MeasurementBase>(0.0, time, value, covariance, Eigen::MatrixXi::Ones(6, 1));
        }

        /// A sliding-window pose problem with one sensor.
        struct Problem
        {
            std::shared_ptr<Priors::WNOA::Prior<Pose>> prior = makePrior();
            Estimator estimator = Estimator(/*smootherLag=*/0.35);
            std::shared_ptr<Sensors::PoseSensor<ValueType>> sensor;

            explicit Problem(std::initializer_list<double> times)
            {
                estimator.addPrior(prior);
                sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
                estimator.addSensor(sensor);
                for (double t : times)
                    sensor->addMeasurement(makeMeasurement(t, 0.1 * t));
            }
        };

        Interface::SessionOptions sessionOptions(double start_time, double end_time, double period, bool real_time = false)
        {
            Interface::SessionOptions options;
            options.start_time = start_time;
            options.end_time = end_time;
            options.period = period;
            options.real_time = real_time;
            return options;
        }

        /// Generic downstream code: only the concept is known here.
        template <Interface::InterfaceType I>
        double runGeneric(Estimator &estimator, const Interface::SessionOptions &options, std::vector<double> &ticks,
                          std::thread::id &tick_thread)
        {
            typename I::Session session(estimator, options);
            session.addMeasurementQueues(estimator.getMeasurementQueues());
            session.setTickCallback([&](double t)
                                    {
                ticks.push_back(t);
                tick_thread = std::this_thread::get_id(); });
            session.start();
            session.join();
            EXPECT_TRUE(session.finished());
            return session.endTime();
        }

        double secondsSince(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } // namespace

    // The session is only a facade: the same estimate as driving the runner directly.
    TEST(StandaloneInterface, OfflineResultMatchesTheRunner)
    {
        Problem through_session({0.0, 0.1, 0.2, 0.3, 0.4, 0.5});
        std::vector<double> ticks;
        std::thread::id tick_thread;
        runGeneric<Interfaces::Standalone>(through_session.estimator, sessionOptions(0.0, 0.5, 0.1), ticks, tick_thread);

        Problem through_runner({0.0, 0.1, 0.2, 0.3, 0.4, 0.5});
        auto clock = std::make_shared<SteppedClock>();
        RunnerOptions runner_options;
        runner_options.start_time = 0.0;
        runner_options.end_time = 0.5;
        runner_options.period = 0.1;
        EstimatorRunner runner(through_runner.estimator, clock, runner_options);
        runner.addDataloader(std::make_shared<FileDataloader>(through_runner.estimator.getMeasurementQueues(), clock));
        runner.run();

        ASSERT_GT(through_runner.estimator.getCurrentValues().size(), 0u);
        EXPECT_TRUE(through_runner.estimator.getCurrentValues().equals(through_session.estimator.getCurrentValues(), 1e-12));
    }

    TEST(StandaloneInterface, OfflineTimeFollowsTheEstimates)
    {
        Estimator estimator;
        Interfaces::Standalone::Session session(estimator, sessionOptions(2.0, 2.2, 0.1));
        EXPECT_FALSE(session.isRealTime());
        std::vector<double> now_at_tick;
        session.setTickCallback([&](double)
                                { now_at_tick.push_back(session.now()); });
        session.start();
        session.join();
        ASSERT_EQ(now_at_tick.size(), 3u);
        EXPECT_DOUBLE_EQ(now_at_tick.front(), 2.0);
    }

    TEST(StandaloneInterface, RealTimeFollowsTheWallClockAndStopsPromptly)
    {
        Estimator estimator;
        Interfaces::Standalone::Session session(estimator, sessionOptions(0.0, 60.0, 0.5, /*real_time=*/true));
        EXPECT_TRUE(session.isRealTime());
        session.start();
        const double before = session.now();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_GE(session.now() - before, 0.02);
        EXPECT_FALSE(session.finished());

        const auto stop_time = std::chrono::steady_clock::now();
        session.requestStop();
        session.join();
        EXPECT_LT(secondsSince(stop_time), 1.0);
        EXPECT_TRUE(session.finished());
    }
} // namespace Spacetime::Standalone
