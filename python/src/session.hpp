#ifndef SPACETIME_PYTHON_SESSION_HPP
#define SPACETIME_PYTHON_SESSION_HPP

/**
 * @file session.hpp
 * @brief What every interface's Python Session shares: the owning wrapper (stopped and joined
 *        without the GIL when Python drops it) and the run methods, including running a viewer
 *        (e.g. spacetime.visualizer.Visualizer) with start(estimator, viewer) and join().
 */

#include <memory>
#include <string>
#include <utility>

#include <spacetime/interface/Interfaces.hpp>
#include <spacetime/interface/Viewer.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    /**
     * @brief Owns an Interface::Session<I>. Dropping it stops the run, joins its estimation and
     *        closes its viewer, all without the GIL: its threads may be waiting for the GIL.
     */
    template <int I>
    class PySession
    {
    public:
        using SessionType = Spacetime::Interface::Session<I>;

        template <typename... Args>
        explicit PySession(Args &&...args) : m_session(std::make_unique<SessionType>(std::forward<Args>(args)...)) {}

        ~PySession()
        {
            py::gil_scoped_release release;
            m_session->requestStop();
            try
            {
                m_session->joinEstimation(); // not join(): that would show the viewer
            }
            catch (...)
            {
                // Already reported by join(), or abandoned with the session.
            }
            m_session.reset(); // closes the viewer, if any, without showing it
        }

        PySession(const PySession &) = delete;
        PySession &operator=(const PySession &) = delete;

        SessionType &operator*() { return *m_session; }
        SessionType *operator->() { return m_session.get(); }

    private:
        std::unique_ptr<SessionType> m_session;
    };

    /**
     * @brief Binds the run methods every session has on `cls`: start (with an optional viewer),
     *        requestStop, joinEstimation, join, viewerExitCode, finished, startTime, endTime,
     *        now and isRealTime.
     * @param start_doc What start() does, for this interface.
     */
    template <int I>
    void bindSessionRun(py::class_<PySession<I>> &cls, const std::string &start_doc)
    {
        using Session = PySession<I>;
        using Spacetime::Interface::Viewer;
        cls.def(
               "start", [](Session &self, Spacetime::Estimator &estimator, Viewer *viewer)
               {
                   if (viewer)
                       self->start(estimator, *viewer);
                   else
                       self->start(estimator); },
               py::arg("estimator"), py::arg("viewer") = nullptr, py::call_guard<py::gil_scoped_release>(),
               py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
               (start_doc +
                " With a `viewer` (e.g. a visualizer.Visualizer), opens it first, on this thread (on macOS this must be the main "
                "thread); join() then shows it until it is closed, and closing it stops the run. Both are kept alive by the session.")
                   .c_str())
            .def("requestStop", [](Session &self)
                 { self->requestStop(); }, "Asks the run to end as soon as possible (and the viewer, if any, then closes); safe from any thread.")
            .def("joinEstimation", [](Session &self)
                 { self->joinEstimation(); }, py::call_guard<py::gil_scoped_release>(),
                 "Waits for estimation to end, without showing the viewer: e.g. to report results before an offline viewer replays "
                 "them in join(). Raises the first error of the run (the viewer is then closed).")
            .def("join", [](Session &self)
                 { self->join(); }, py::call_guard<py::gil_scoped_release>(),
                 "Waits for the run to end, showing the viewer (if any) until it is closed: online, while estimating; offline, once "
                 "estimation has ended, replaying it. Raises the first error of the run or the viewer (KeyboardInterrupt if Ctrl-C "
                 "closed the viewer).")
            .def("viewerExitCode", [](Session &self)
                 { return self->viewerExitCode(); }, "The viewer's exit code once join() has shown it; 0 before, or without one.")
            .def("finished", [](Session &self)
                 { return self->finished(); }, "Whether the run has ended (False if never started).")
            .def("startTime", [](Session &self)
                 { return self->startTime(); }, "The time of the first estimate [s] (SessionOptions.start_time).")
            .def("endTime", [](Session &self)
                 { return self->endTime(); }, "The end time actually used [s] (possibly inf); valid after start().")
            .def("now", [](Session &self)
                 { return self->now(); }, "The current estimation time [s]; safe from any thread.")
            .def("isRealTime", [](Session &self)
                 { return self->isRealTime(); }, "Whether estimation follows a real-time clock.");
    }
} // namespace SpacetimePython

#endif // SPACETIME_PYTHON_SESSION_HPP
