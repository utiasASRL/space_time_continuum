/**
 * @file Interface.hpp
 * @brief The standalone ("default") interface, selected by Spacetime::Session (i.e.
 *        Spacetime::Session<INTERFACE_STANDALONE>): file playback on its own estimator and
 *        dataloader threads, as fast as possible (SteppedClock) or in real time (WallClock).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_INTERFACE_HPP
#define SPACETIME_STANDALONE_INTERFACE_HPP

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/EstimatorRunner.hpp>
#include <spacetime/interface/standalone/FileDataloader.hpp>

namespace Spacetime::Interface::Standalone
{
    /// The standalone Session (see Interface::InterfaceType for each hook's contract): a thin wrapper
    /// over an EstimatorRunner, built by start(), with one FileDataloader over all of the estimator's
    /// measurement queues.
    class Session
    {
    public:
        /// Real time on a WallClock, otherwise as fast as possible on a SteppedClock.
        explicit Session(const Interface::SessionOptions &options = Interface::SessionOptions())
            : m_options(options), mp_clock(makeClock(options.real_time)) {}

        Session(const Session &) = delete;
        Session &operator=(const Session &) = delete;

        /// See EstimatorRunner::setTickCallback. The callback runs on the estimator thread. Call before start().
        void setTickCallback(Interface::EstimatorDriver::TickCallback callback) { m_tick_callback = std::move(callback); }

        /**
         * @brief Starts estimating with `estimator`, which must outlive the session: replays
         *        what is loaded into its measurement queues (Estimator::getMeasurementQueues,
         *        see FileDataloader) on a dataloader thread, and starts the estimator thread
         *        (see EstimatorRunner::start).
         * @throws std::logic_error if already started; otherwise as EstimatorRunner::start
         *         does, leaving the queues' data in place.
         */
        void start(Estimator &estimator)
        {
            std::lock_guard<std::mutex> lock(m_runner_mutex);
            if (mp_runner)
                throw std::logic_error("Standalone::Session::start: already started.");
            mp_runner = std::make_unique<EstimatorRunner>(estimator, mp_clock, runnerOptions(m_options));
            mp_runner->addDataloader(std::make_shared<FileDataloader>(estimator.getMeasurementQueues(), mp_clock));
            mp_runner->setTickCallback(m_tick_callback);
            if (m_stop_requested)
                mp_runner->requestStop();
            try
            {
                mp_runner->start();
            }
            catch (...)
            {
                mp_runner.reset(); // hands the withheld measurements back to their queues
                throw;
            }
        }

        /// Asks every thread to stop. Safe from any thread, at any time (also before start()).
        void requestStop()
        {
            std::lock_guard<std::mutex> lock(m_runner_mutex);
            m_stop_requested = true;
            if (mp_runner)
                mp_runner->requestStop();
        }
        /// Blocks until the run ends; rethrows the first error raised on any thread. Returns at once if never started.
        void join()
        {
            if (mp_runner) // only start(), on this same control thread, sets it
                mp_runner->join();
        }
        /// True once the run has ended (always false if never started).
        bool finished() const
        {
            std::lock_guard<std::mutex> lock(m_runner_mutex);
            return mp_runner && mp_runner->finished();
        }
        /// The time of the first estimate (SessionOptions::start_time), in seconds.
        double startTime() const { return m_options.start_time; }
        /// The resolved end time, in seconds. Valid after start().
        double endTime() const
        {
            std::lock_guard<std::mutex> lock(m_runner_mutex);
            return mp_runner ? mp_runner->endTime() : m_options.end_time;
        }
        /// The clock's current time, in seconds. Safe from any thread.
        double now() const { return mp_clock->now(); }
        /// Whether SessionOptions::real_time selected a WallClock.
        bool isRealTime() const { return mp_clock->isRealTime(); }

    private:
        Interface::SessionOptions m_options;
        std::shared_ptr<PacingClock> mp_clock;
        Interface::EstimatorDriver::TickCallback m_tick_callback;
        mutable std::mutex m_runner_mutex; // guards mp_runner's creation against concurrent readers and m_stop_requested
        bool m_stop_requested = false;
        std::unique_ptr<EstimatorRunner> mp_runner; // built by start()

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
} // namespace Spacetime::Interface::Standalone

namespace Spacetime::Interface
{
    /// The standalone ("default") interface: replays recorded data (see Standalone::Session).
    template <>
    struct Traits<INTERFACE_STANDALONE>
    {
        static constexpr std::string_view name = "default"; ///< Interface name.
        using Session = Standalone::Session;                ///< See Standalone::Session.
    };
    static_assert(InterfaceType<Traits<INTERFACE_STANDALONE>>);
} // namespace Spacetime::Interface

#endif // SPACETIME_STANDALONE_INTERFACE_HPP
