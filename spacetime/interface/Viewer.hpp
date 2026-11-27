/**
 * @file Viewer.hpp
 * @brief Interface::Viewer: something that shows an estimator's results while or after a
 *        session runs it, e.g. the default visualizer's window
 *        (Visualization::Default::VisualizerHost). A session runs it with
 *        `session.start(estimator, viewer); session.join();` (see ViewerSession), which calls
 *        every hook below from the thread calling start() and join(): on macOS that must be
 *        the process's main thread, the only one a GUI may run on.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_VIEWER_HPP
#define SPACETIME_INTERFACE_VIEWER_HPP

#include <functional>

#include <spacetime/estimator/Estimator.hpp>

namespace Spacetime::Interface
{
    template <typename Base>
    class ViewerSession;

    /**
     * @brief Shows an estimator's results; run by a session (see ViewerSession, and the file
     *        comment). Its lifecycle, driven by the session: open() (before estimation starts),
     *        beginPlayback() (right after it starts if isOnline(), else once it has ended), then
     *        run() until closed. requestClose() may come at any point, from any thread.
     */
    class Viewer
    {
    public:
        /// Called once the viewer closes, for whatever reason (e.g. the user closed its window): the session then stops.
        using ClosedCallback = std::function<void()>;
        /// The current estimation time (the session's now()), for online playback.
        using LiveTimeSource = std::function<double()>;

        Viewer() = default;
        Viewer(const Viewer &) = delete;
        Viewer &operator=(const Viewer &) = delete;
        /// Detaches the viewer from the session running it, if any (see ViewerSession).
        virtual ~Viewer()
        {
            if (m_detach)
                m_detach();
        }

        /**
         * @brief Sets the viewer up to show `estimator`, which outlives it, e.g. builds its views.
         *        Called before estimation starts.
         * @param on_closed See ClosedCallback; may be called from any thread.
         * @throws Whatever setting up threw; the session then does not start.
         */
        virtual void open(Estimator &estimator, ClosedCallback on_closed) = 0;

        /// Whether playback follows the live estimate (beginPlayback right after estimation starts) or replays it once it has ended.
        virtual bool isOnline() const = 0;

        /**
         * @brief Sets what to play: [start_time, end_time] [s].
         * @param live_time_source Online only: the current estimation time; empty when replaying.
         */
        virtual void beginPlayback(double start_time, double end_time, LiveTimeSource live_time_source) = 0;

        /// Shows the results and blocks until the viewer is closed; returns its exit code. Returns at once if requestClose() came first.
        virtual int run() = 0;

        /// Asks the viewer to close (so run() returns) as soon as possible. Safe from any thread, at any time.
        virtual void requestClose() = 0;

    private:
        template <typename Base>
        friend class ViewerSession;
        /// Set by the session running this viewer, to forget it if the viewer is destroyed first.
        std::function<void()> m_detach;
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_VIEWER_HPP
