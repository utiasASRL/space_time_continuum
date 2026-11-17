/**
 * @file Interface.hpp
 * @brief The standalone interface's bundle, Interfaces::Standalone (see
 *        Interface::InterfaceType): file playback on its own estimator and dataloader
 *        threads, as fast as possible (SteppedClock) or in real time (WallClock).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_INTERFACE_HPP
#define SPACETIME_STANDALONE_INTERFACE_HPP

#include <memory>
#include <string_view>
#include <vector>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/EstimatorRunner.hpp>
#include <spacetime/interface/standalone/FileDataloader.hpp>

namespace Spacetime::Standalone
{
    /// The standalone Session (see Interface::InterfaceType for each hook's contract): a thin wrapper
    /// over an EstimatorRunner with one FileDataloader per addMeasurementQueues() call.
    class Session
    {
    public:
        /// Real time on a WallClock, otherwise as fast as possible on a SteppedClock.
        Session(Estimator &estimator, const Interface::SessionOptions &options)
            : mp_clock(makeClock(options.real_time)), m_runner(estimator, mp_clock, runnerOptions(options)) {}

        /// Replays what is already loaded into `queues` (see FileDataloader), on its own dataloader thread.
        /// Withholds (empties) the queues immediately; call before start() and before any measurement observer subscribes.
        void addMeasurementQueues(const std::vector<Types::pMeasurementQueueType> &queues)
        {
            m_runner.addDataloader(std::make_shared<FileDataloader>(queues, mp_clock));
        }

        /// See EstimatorRunner::setTickCallback. The callback runs on the estimator thread.
        void setTickCallback(Interface::EstimatorDriver::TickCallback callback) { m_runner.setTickCallback(std::move(callback)); }
        /// Starts the estimator and dataloader threads (see EstimatorRunner::start; throws as it does).
        void start() { m_runner.start(); }
        /// Asks every thread to stop. Safe from any thread.
        void requestStop() { m_runner.requestStop(); }
        /// Blocks until the run ends; rethrows the first error raised on any thread.
        void join() { m_runner.join(); }
        /// True once the run has ended (always false if never started).
        bool finished() const { return m_runner.finished(); }
        /// The resolved end time, in seconds. Valid after start().
        double endTime() const { return m_runner.endTime(); }
        /// The clock's current time, in seconds. Safe from any thread.
        double now() const { return mp_clock->now(); }
        /// Whether SessionOptions::real_time selected a WallClock.
        bool isRealTime() const { return mp_clock->isRealTime(); }

    private:
        std::shared_ptr<PacingClock> mp_clock;
        EstimatorRunner m_runner;

        static std::shared_ptr<PacingClock> makeClock(bool real_time)
        {
            if (real_time)
                return std::make_shared<WallClock>();
            return std::make_shared<SteppedClock>();
        }

        static RunnerOptions runnerOptions(const Interface::SessionOptions &options)
        {
            RunnerOptions runner_options;
            runner_options.start_time = options.start_time;
            runner_options.end_time = options.end_time;
            runner_options.period = options.period;
            return runner_options;
        }
    };
} // namespace Spacetime::Standalone

namespace Spacetime::Interfaces
{
    /// The standalone interface: replays recorded data (see Standalone::Session).
    struct Standalone
    {
        static constexpr std::string_view name = "standalone"; ///< Interface name.
        using Session = Spacetime::Standalone::Session;         ///< See Standalone::Session.
    };
    static_assert(Interface::InterfaceType<Standalone>);
} // namespace Spacetime::Interfaces

#endif // SPACETIME_STANDALONE_INTERFACE_HPP
