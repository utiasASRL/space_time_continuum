/**
 * @file EstimatorRunner.hpp
 * @brief The standalone interface's threads: one estimator thread and one thread per
 *        dataloader. The estimator thread steps an Interface::EstimatorDriver on the times
 *        of an Interface::EstimationSchedule, read from and paced by a PacingClock:
 *        - Offline (e.g. SteppedClock), it runs as fast as possible, and before each
 *          estimate waits until every dataloader has delivered all of its measurements up to
 *          that time (see Interface::Dataloader), so the result does not depend on thread
 *          timing.
 *        - In real time (e.g. WallClock), it sleeps until each scheduled time and never waits
 *          on data; whatever has arrived is used.
 *
 *        Each dataloader thread fills the models' measurement queues. Results are read
 *        through Estimator::getSnapshot() (or waitForSnapshotNewerThan) from any other thread.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_ESTIMATORRUNNER_HPP
#define SPACETIME_STANDALONE_ESTIMATORRUNNER_HPP

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/interface/EndTime.hpp>
#include <spacetime/interface/EstimationSchedule.hpp>
#include <spacetime/interface/EstimatorDriver.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/ThreadedDataloader.hpp>

namespace Spacetime::Standalone
{
    /// How an EstimatorRunner runs (see its file comment).
    struct RunnerOptions
    {
        double start_time = 0.0; ///< Time of the first estimate, in seconds.
        /// Time to stop estimating at. If negative: until the data ends (see Interface::resolveEndTime).
        double end_time = -1.0;
        double period = 1.0 / 30.0; ///< Time between estimates, and so between nodes, in seconds.
        /// Whether to wait on the dataloaders' watermarks before each estimate. Defaults to !clock->isRealTime().
        std::optional<bool> synchronize;
    };

    /// Owns the estimator thread and the dataloader threads (see the file comment). Configure
    /// (addDataloader, setTickCallback) and start() from one thread; requestStop() is safe from any.
    class EstimatorRunner
    {
    public:
        /// Called on the estimator thread with the time of each estimate, right after it. Keep it cheap.
        using TickCallback = Interface::EstimatorDriver::TickCallback;

        /// Drives `estimator`, which must outlive the runner, on `clock`.
        EstimatorRunner(Estimator &estimator, std::shared_ptr<PacingClock> clock, RunnerOptions options = RunnerOptions())
            : m_estimator(estimator), mp_clock(std::move(clock)), m_options(options) {}

        /// Stops and joins any running threads (discarding their errors).
        ~EstimatorRunner()
        {
            requestStop();
            if (m_estimator_thread.joinable())
                m_estimator_thread.join();
        }

        EstimatorRunner(const EstimatorRunner &) = delete;
        EstimatorRunner &operator=(const EstimatorRunner &) = delete;

        /// Adds a dataloader to run on its own thread. Call before start().
        void addDataloader(std::shared_ptr<ThreadedDataloader> dataloader) { mp_dataloaders.push_back(std::move(dataloader)); }

        /// Sets the per-estimate callback (see TickCallback). Call before start().
        void setTickCallback(TickCallback callback) { m_tick_callback = std::move(callback); }

        /**
         * @brief Starts the clock at the start time, then the dataloader threads and the estimator thread.
         * @throws std::invalid_argument if the period is not positive, or a batch-mode
         *         estimator (see Estimator::isBatch) is given a real-time clock.
         * @throws std::logic_error if already started.
         * @note A batch-mode estimator only accumulates factors during the loop, and is solved
         *       by Estimator::finalize() once it ends (also when it ends on a stop request), on
         *       the estimator thread.
         */
        void start()
        {
            if (m_started)
                throw std::logic_error("EstimatorRunner::start: already started.");
            const bool real_time = mp_clock->isRealTime();
            m_end_time = Interface::resolveEndTime(m_options.end_time, mp_dataloaders);
            // Both validate their arguments, before any thread starts.
            Interface::EstimationSchedule schedule(m_options.start_time, m_end_time, m_options.period, real_time);
            Interface::EstimatorDriver driver(m_estimator, real_time);
            driver.setTickCallback(m_tick_callback);
            m_started = true;

            const bool synchronize = m_options.synchronize.value_or(!real_time);
            debug() << "Running estimation from time: " << m_options.start_time << " to time: " << m_end_time
                    << (real_time ? " in real time" : " as fast as possible")
                    << " (period=" << m_options.period << ")";

            mp_clock->start(m_options.start_time);
            for (const auto &dataloader : mp_dataloaders)
                m_dataloader_threads.emplace_back([this, dataloader]()
                                                  { guarded([&]()
                                                            { dataloader->run(m_dataloader_stop.get_token()); }); });
            m_estimator_thread = std::thread([this, schedule, driver = std::move(driver), synchronize]() mutable
                                             { estimatorThread(schedule, driver, synchronize); });
        }

        /// Asks every thread to stop as soon as possible. Safe to call from any thread, at any time.
        void requestStop()
        {
            m_stop.request_stop();
            m_dataloader_stop.request_stop();
        }

        /// Waits for the estimator (and so every dataloader) thread to end. Rethrows the first error raised on any of them.
        void join()
        {
            if (m_estimator_thread.joinable())
                m_estimator_thread.join();
            std::lock_guard<std::mutex> lock(m_error_mutex);
            if (m_error)
                std::rethrow_exception(std::exchange(m_error, nullptr));
        }

        /// start() then join(): blocks the calling thread until the run ends. Throws as start() and join() do.
        void run()
        {
            start();
            join();
        }

        /// True once the estimator thread has finished (reached the end time, stopped, or failed) and every dataloader has ended.
        bool finished() const { return m_finished; }

        /// The end time actually used (see RunnerOptions::end_time). Valid after start().
        double endTime() const { return m_end_time; }

        /// The clock the estimator is paced by, e.g. to show live time in a visualizer.
        const std::shared_ptr<PacingClock> &clock() const { return mp_clock; }

    private:
        Estimator &m_estimator;
        std::shared_ptr<PacingClock> mp_clock;
        RunnerOptions m_options;
        std::vector<std::shared_ptr<ThreadedDataloader>> mp_dataloaders;
        TickCallback m_tick_callback;
        double m_end_time = 0.0;

        std::stop_source m_stop;            // ends the estimator loop (and its watermark waits)
        std::stop_source m_dataloader_stop; // ends the dataloaders; also requested once the estimator loop ends
        std::thread m_estimator_thread;
        std::vector<std::thread> m_dataloader_threads; // joined by the estimator thread as it ends
        bool m_started = false;
        std::atomic<bool> m_finished{false};

        std::mutex m_error_mutex;
        std::exception_ptr m_error; // the first error raised on any thread

        /// Runs `fn`, recording any error it raises and stopping every thread.
        template <typename Fn>
        void guarded(Fn &&fn)
        {
            try
            {
                fn();
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex> lock(m_error_mutex);
                    if (!m_error)
                        m_error = std::current_exception();
                }
                requestStop();
            }
        }

        /// Body of the estimator thread: the estimate loop, finalize, then stops and joins the dataloader threads.
        void estimatorThread(Interface::EstimationSchedule &schedule, Interface::EstimatorDriver &driver, bool synchronize)
        {
            guarded([&]()
                    {
                estimateLoop(schedule, driver, synchronize, m_stop.get_token());
                driver.finish(); });

            // Live dataloaders never end on their own, and recorded ones may hold data past the end time.
            m_dataloader_stop.request_stop();
            for (auto &thread : m_dataloader_threads)
                thread.join();
            m_finished = true;
        }

        /// Steps `driver` on the schedule's times until done or stopped; returns early if stopped during a watermark wait.
        void estimateLoop(Interface::EstimationSchedule &schedule, Interface::EstimatorDriver &driver, bool synchronize, std::stop_token stop)
        {
            std::optional<double> previous_time;
            for (double time = schedule.firstTime(); !schedule.isDone(time) && !stop.stop_requested();)
            {
                if (previous_time && schedule.isLate(*previous_time, time))
                    warning() << "EstimatorRunner: the estimate at t=" << time << " is " << time - *previous_time
                              << " after the previous one, more than the period of " << schedule.period()
                              << "; estimation is being called too slowly.";
                previous_time = time;

                if (synchronize)
                    for (const auto &dataloader : mp_dataloaders)
                        if (!dataloader->waitUntilDeliveredThrough(time, stop))
                            return;

                driver.step(time);

                // Offline, the clock just jumps to the next time; in real time, the next
                // estimate is at whatever the clock reads once the wait is over.
                const double wake_time = schedule.next(mp_clock->now());
                mp_clock->sleepUntil(wake_time, stop);
                time = schedule.isRealTime() ? mp_clock->now() : wake_time;
            }
        }
    };
} // namespace Spacetime::Standalone

#endif // SPACETIME_STANDALONE_ESTIMATORRUNNER_HPP
