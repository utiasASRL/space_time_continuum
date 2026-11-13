/**
 * @file VisualizerThread.hpp
 * @brief Runs the default (Qt/VTK) visualizer on its own low-priority thread, so the
 *        process main thread stays free to be the control thread. Every Qt object (the
 *        QApplication, the Visualizer and every view) is created, used and destroyed on
 *        that thread only; other threads talk to it through this class.
 *
 *        Qt's GUI thread does not have to be the process main thread on Linux, but it
 *        does on macOS, which this does not support.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_VISUALIZER_DEFAULT_VISUALIZERTHREAD_HPP
#define SPACETIME_VISUALIZER_DEFAULT_VISUALIZERTHREAD_HPP

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <spacetime/visualizer/VisualizerOptions.hpp>

class QApplication;

namespace Spacetime::Visualization::Default
{
    class Visualizer;

    /// The visualizer's thread (see the file comment). Lifecycle: construct, waitReady(), beginPlayback(), then join() or requestQuit().
    class VisualizerThread
    {
    public:
        /// Adds the views to the visualizer and binds their data sources. Runs on the visualizer thread.
        using BuildFn = std::function<void(Visualizer &)>;

        /**
         * @brief Starts the thread, which lowers its own priority by `nice_delta`, creates
         *        the QApplication and the Visualizer, runs `build`, then waits for
         *        beginPlayback().
         * @param argc, argv Passed to QApplication; must outlive this object (e.g. main's).
         * @param on_closed Called on the visualizer thread once its event loop ends, for
         *        whatever reason (the user quit or closed the window, or requestQuit()),
         *        e.g. to stop the estimator.
         */
        VisualizerThread(int &argc, char **argv, VisualizerOptions options, std::string output_path, BuildFn build,
                         std::function<void()> on_closed = nullptr, int nice_delta = 10);

        /// Quits the event loop (if running) and joins the thread.
        ~VisualizerThread();

        VisualizerThread(const VisualizerThread &) = delete;
        VisualizerThread &operator=(const VisualizerThread &) = delete;

        /// Blocks until `build` has run. Rethrows anything it (or the visualizer's construction) threw.
        void waitReady();

        /**
         * @brief Shows the visualizer and starts its event loop, playing over [start_time, end_time].
         * @param live_time_source Online mode only: the current estimation time (see
         *        Visualizer::setLiveTimeSource), read on the visualizer thread, e.g. the
         *        runner's clock.
         */
        void beginPlayback(double start_time, double end_time, std::function<double()> live_time_source = nullptr);

        /// Asks the visualizer to quit, whether or not playback has begun. Safe to call from any thread.
        void requestQuit();

        /// Waits for the thread to end. Returns the event loop's exit code (0 if it never ran).
        int join();

    private:
        /// What beginPlayback() hands to the visualizer thread.
        struct Playback
        {
            double start_time, end_time;
            std::function<double()> live_time_source;
        };

        int &m_argc;
        char **m_argv;
        VisualizerOptions m_options;
        std::string m_output_path;
        BuildFn m_build;
        std::function<void()> m_on_closed;
        int m_nice_delta;

        std::mutex m_mutex;
        std::condition_variable m_changed;
        bool m_ready = false;
        bool m_quit_requested = false;
        std::optional<Playback> m_playback;
        std::exception_ptr m_build_error;
        QApplication *mp_app = nullptr; // set while the event loop can receive a quit
        int m_exit_code = 0;

        std::thread m_thread;

        void threadMain();
    };
} // namespace Spacetime::Visualization::Default

#endif // SPACETIME_VISUALIZER_DEFAULT_VISUALIZERTHREAD_HPP
