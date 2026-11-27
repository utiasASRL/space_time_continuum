/**
 * @file VisualizerHost.hpp
 * @brief Hosts the default (Qt/VTK) visualizer: owns the QApplication, the Visualizer window and
 *        every view, and runs its event loop. Every Qt object is created, used and destroyed
 *        on one thread, the GUI thread; other threads talk to it through this class. It is an
 *        Interface::Viewer, so a session runs it: `session.start(estimator, host); session.join();`.
 *
 *        Where the GUI thread is (HostMode):
 *        - WorkerThread (the default on Linux): a thread of its own, which lowers its priority
 *          for good, leaving the calling thread free (e.g. to be the control thread).
 *        - CallingThread (the default on macOS and Windows): the thread calling open() and
 *          run(). macOS only allows a GUI on the process's main thread (open() throws
 *          elsewhere), and Windows prefers it there. The thread is lowered only while run()
 *          shows the window (by when the session has started its estimator), and restored
 *          after. On Linux restoring needs privileges (see ScopedLowerThreadPriority): without
 *          them the thread stays lowered, and so do threads it creates later (e.g. a later
 *          session's estimator); run() warns then. Hence WorkerThread there.
 *        Either way the estimator thread stays above the GUI thread (see ThreadPriority.hpp).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_VISUALIZER_DEFAULT_VISUALIZERHOST_HPP
#define SPACETIME_VISUALIZER_DEFAULT_VISUALIZERHOST_HPP

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/Viewer.hpp>
#include <spacetime/visualizer/VisualizerOptions.hpp>

class QApplication;

namespace Spacetime::Visualization::Default
{
    class Visualizer;

    /// Where a VisualizerHost runs Qt (see the file comment).
    enum class HostMode
    {
        WorkerThread,  ///< A thread of the host's own, lowered in priority for good.
        CallingThread, ///< The thread calling open() and run() (on macOS: the main thread).
    };

    /// WorkerThread on Linux; CallingThread on macOS (where it is the only one possible) and Windows.
    HostMode defaultHostMode();

    /// The visualizer's host (see the file comment). Lifecycle: construct, open(), beginPlayback(), run(); requestClose() at any time.
    class VisualizerHost : public Interface::Viewer
    {
    public:
        /// Adds the views to the visualizer and binds their data sources. Runs on the GUI thread.
        using BuildFn = std::function<void(Visualizer &)>;

        /**
         * @brief Stores what open() needs; creates nothing yet.
         * @param argc, argv Passed to QApplication; must outlive this object (e.g. main's).
         * @param options, output_path Passed to the Visualizer (see its constructor);
         *        `options.online` is isOnline().
         * @param build Run once on the GUI thread by open(), after the Visualizer is built.
         * @param mode Where the GUI thread is (see HostMode).
         * @param nice_delta How much the GUI thread is lowered (see
         *        Interface::Standalone::lowerCurrentThreadPriority).
         */
        VisualizerHost(int &argc, char **argv, VisualizerOptions options, std::string output_path, BuildFn build,
                       HostMode mode = defaultHostMode(), int nice_delta = 10);

        /// Closes the window (if open) and, in WorkerThread mode, joins the thread. Destroy it on the thread that called open().
        ~VisualizerHost() override;

        /**
         * @brief Creates the QApplication and the Visualizer on the GUI thread, and runs `build`
         *        there; returns once that is done.
         * @param on_closed Called on the GUI thread once the host is done, for whatever reason
         *        (the user quit or closed the window, requestClose(), or `build` throwing), e.g.
         *        to stop the estimator.
         * @throws std::logic_error if already opened, or (macOS, CallingThread) off the main
         *         thread; whatever `build` (or the QApplication's or Visualizer's construction)
         *         threw, after which the host is done.
         */
        void open(ClosedCallback on_closed = nullptr);

        /// Interface::Viewer: as open(on_closed); `build` reads the estimator itself.
        void open(Estimator &estimator, ClosedCallback on_closed) override;

        /// Interface::Viewer: VisualizerOptions::online.
        bool isOnline() const override;

        /**
         * @brief What run() plays: [start_time, end_time] (see Visualizer::setTimeRange). Ignored
         *        if requestClose() came first. In WorkerThread mode the window shows at once.
         * @param live_time_source Online mode only: the current estimation time (see
         *        Visualizer::setLiveTimeSource), read on the GUI thread, e.g. a session's now().
         */
        void beginPlayback(double start_time, double end_time, LiveTimeSource live_time_source = nullptr) override;

        /**
         * @brief Shows the window and blocks until it is closed: in CallingThread mode, runs the
         *        event loop on this thread (lowered meanwhile); in WorkerThread mode, waits for
         *        the thread. Returns the event loop's exit code (0 if it never ran, e.g. after
         *        requestClose()). Call after open(), from the thread that called it.
         * @throws std::logic_error in CallingThread mode without beginPlayback() or requestClose() first.
         */
        int run() override;

        /// Asks the visualizer to quit, whether or not playback has begun. Safe to call from any thread.
        void requestClose() override;

        /// Where the GUI thread is.
        HostMode mode() const { return m_mode; }

        VisualizerHost(const VisualizerHost &) = delete;
        VisualizerHost &operator=(const VisualizerHost &) = delete;

    private:
        /// What beginPlayback() hands to the GUI thread.
        struct Playback
        {
            double start_time, end_time;     ///< Playback range [s].
            LiveTimeSource live_time_source; ///< See beginPlayback.
        };

        // Constructor arguments.
        int &m_argc;
        char **m_argv;
        VisualizerOptions m_options;
        std::string m_output_path;
        BuildFn m_build;
        HostMode m_mode;
        int m_nice_delta;
        ClosedCallback m_on_closed; ///< Set by open().

        // Qt objects, only touched on the GUI thread.
        std::unique_ptr<QApplication> mp_application;
        std::unique_ptr<Visualizer> mp_visualizer;

        /// Guards every field below.
        std::mutex m_mutex;
        /// Signalled when m_ready, m_quit_requested or m_playback changes.
        std::condition_variable m_changed;
        bool m_opened = false;         ///< Set by open().
        bool m_ready = false;          ///< Set once `build` ran (or construction threw).
        bool m_done = false;           ///< Set once torn down (on_closed called).
        bool m_quit_requested = false; ///< Set by requestClose().
        std::optional<Playback> m_playback;
        std::exception_ptr m_build_error; ///< What construction threw, rethrown by open().
        QApplication *mp_app = nullptr;   ///< Set while the event loop can receive a quit.
        int m_exit_code = 0;

        std::thread m_thread; ///< WorkerThread mode: the GUI thread.

        /// Creates the QApplication and the Visualizer and runs `build`; records any error and marks the host ready.
        void construct();
        /// Shows the window and runs the event loop over `playback`; returns its exit code.
        int runEventLoop(Playback playback);
        /// Destroys the Qt objects and calls on_closed (once).
        void teardown();
        /// WorkerThread mode: the thread's body.
        void workerMain();
    };
} // namespace Spacetime::Visualization::Default

#endif // SPACETIME_VISUALIZER_DEFAULT_VISUALIZERHOST_HPP
