/**
 * @file ViewerSession.hpp
 * @brief Interface::ViewerSession: runs a Viewer (e.g. the default visualizer's window) along
 *        with any interface's session, so a run with a window is
 *        @code
 *        Spacetime::Session session(options);
 *        session.start(estimator, viewer);
 *        session.join(); // estimates and shows the viewer until it is closed
 *        @endcode
 *        Interface::Session (see Interfaces.hpp) is a ViewerSession over the selected
 *        interface's session, so every interface gets this without implementing it.
 *
 *        Call start() and join() from one thread, the "control" thread: the viewer is opened
 *        and run on it (on macOS this must be the process's main thread, where a GUI must run).
 *        The order keeps the estimator the highest-priority thread: the viewer opens before
 *        the session starts its estimator, and only lowers its own thread while it runs.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_VIEWERSESSION_HPP
#define SPACETIME_INTERFACE_VIEWERSESSION_HPP

#include <cmath>
#include <stdexcept>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/Viewer.hpp>

namespace Spacetime::Interface
{
    /**
     * @brief `Base` (a session, see InterfaceType) that can also run a Viewer: start(estimator,
     *        viewer), joinEstimation() and join() (see the file comment). Without a viewer it
     *        behaves exactly as `Base`.
     * @tparam Base The interface's session type.
     */
    template <typename Base>
    class ViewerSession : public Base
    {
    public:
        using Base::Base;

        /// Closes a viewer still open (run() never reached), before `Base` stops the run.
        ~ViewerSession()
        {
            if (!mp_viewer)
                return;
            mp_viewer->m_detach = nullptr;
            if (!m_viewer_ran)
            {
                mp_viewer->requestClose();
                try
                {
                    mp_viewer->run();
                }
                catch (...)
                {
                    // Being destroyed: nothing to report it to.
                }
            }
        }

        /// Starts estimating without a viewer (see the interface's session).
        void start(Estimator &estimator) { Base::start(estimator); }

        /**
         * @brief Opens `viewer` on `estimator` (on the calling thread), then starts estimating.
         *        An online viewer begins playback at once, following now(); an offline one
         *        replays [startTime(), endTime()] once estimation has ended, in join(). Closing
         *        the viewer stops the session. Both must outlive the session's join().
         * @throws std::logic_error if already started with a viewer; whatever the viewer's open()
         *         or the session's start() throws (the viewer is then closed).
         */
        void start(Estimator &estimator, Viewer &viewer)
        {
            if (mp_viewer)
                throw std::logic_error("Session::start: already started with a viewer.");
            viewer.open(estimator, [this]()
                        { this->requestStop(); });
            mp_viewer = &viewer;
            viewer.m_detach = [this]()
            { mp_viewer = nullptr; };
            try
            {
                Base::start(estimator);
            }
            catch (...)
            {
                closeViewer();
                throw;
            }
            if (viewer.isOnline())
                viewer.beginPlayback(this->startTime(), this->endTime(), [this]()
                                     { return this->now(); });
        }

        /**
         * @brief Waits for estimation to end, without running the viewer: e.g. to report or
         *        save results before an offline viewer replays them in join(). On an error the
         *        viewer is closed, and the error rethrown.
         */
        void joinEstimation()
        {
            if (!mp_viewer)
            {
                Base::join();
                return;
            }
            if (m_estimation_joined)
                return;
            m_estimation_joined = true;
            try
            {
                Base::join();
            }
            catch (...)
            {
                closeViewer();
                throw;
            }
        }

        /**
         * @brief Waits for the run to end, running the viewer (if any) on this thread until it is
         *        closed: online, while estimating, after which estimation is stopped and joined;
         *        offline, once estimation has ended (see joinEstimation), replaying it. Rethrows
         *        the first error of the run, or the viewer's.
         */
        void join()
        {
            if (!mp_viewer)
            {
                Base::join();
                return;
            }
            if (mp_viewer->isOnline())
            {
                runViewer();
                joinEstimation();
                return;
            }
            joinEstimation();
            if (!m_viewer_ran && mp_viewer)
            {
                double end_time = this->endTime();
                if (!std::isfinite(end_time))
                    end_time = this->now(); // unbounded (e.g. live data): replay up to where it stopped
                mp_viewer->beginPlayback(this->startTime(), end_time, nullptr);
                runViewer();
            }
        }

        /// The viewer's exit code (see Viewer::run), once join() has run it; 0 before or without one.
        int viewerExitCode() const { return m_viewer_exit_code; }

    private:
        Viewer *mp_viewer = nullptr;      ///< Set by start(estimator, viewer); cleared if the viewer is destroyed first.
        bool m_viewer_ran = false;        ///< Whether its run() has been called (it then needs no closing).
        bool m_estimation_joined = false; ///< Whether joinEstimation() has run (with a viewer).
        int m_viewer_exit_code = 0;

        /// Runs the viewer until closed; if that throws, stops and joins estimation and rethrows.
        void runViewer()
        {
            if (m_viewer_ran || !mp_viewer)
                return;
            m_viewer_ran = true;
            try
            {
                m_viewer_exit_code = mp_viewer->run();
            }
            catch (...)
            {
                this->requestStop();
                try
                {
                    joinEstimation();
                }
                catch (...)
                {
                    // The viewer's error is reported instead.
                }
                throw;
            }
        }

        /// Closes the viewer without showing it (run() returns at once after requestClose()).
        void closeViewer()
        {
            if (m_viewer_ran || !mp_viewer)
                return;
            m_viewer_ran = true;
            mp_viewer->requestClose();
            try
            {
                mp_viewer->run();
            }
            catch (...)
            {
                // Closing because of another error, which is the one reported.
            }
        }
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_VIEWERSESSION_HPP
