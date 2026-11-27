/**
 * @file bind_interface.cpp
 * @brief spacetime.interface: the standalone session that runs an estimator over recorded (or
 *        synthetic) measurements, and the snapshot history.
 */

#include <memory>

#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/Interfaces.hpp>
#include <spacetime/interface/SnapshotHistory.hpp>
#include <spacetime/interface/Viewer.hpp>

#include "common.hpp"
#include "session.hpp"

namespace SpacetimePython
{
    namespace
    {
        using Spacetime::Interface::Viewer;

        /// Lets a Python class be a Viewer (e.g. a custom window, or a test's fake): every hook takes the GIL.
        struct PyViewer : Viewer, py::trampoline_self_life_support
        {
            void open(Spacetime::Estimator &estimator, ClosedCallback on_closed) override
            {
                py::gil_scoped_acquire gil;
                PYBIND11_OVERRIDE_PURE(void, Viewer, open, estimator, std::move(on_closed));
            }
            bool isOnline() const override
            {
                py::gil_scoped_acquire gil;
                PYBIND11_OVERRIDE_PURE(bool, Viewer, isOnline);
            }
            void beginPlayback(double start_time, double end_time, LiveTimeSource live_time_source) override
            {
                py::gil_scoped_acquire gil;
                PYBIND11_OVERRIDE_PURE(void, Viewer, beginPlayback, start_time, end_time, std::move(live_time_source));
            }
            int run() override
            {
                py::gil_scoped_acquire gil;
                PYBIND11_OVERRIDE_PURE(int, Viewer, run);
            }
            void requestClose() override
            {
                py::gil_scoped_acquire gil;
                PYBIND11_OVERRIDE_PURE(void, Viewer, requestClose);
            }
        };

        using Session = PySession<Spacetime::Interface::INTERFACE_STANDALONE>;
    } // namespace

    void bindInterface(py::module_ &m)
    {
        using Spacetime::Interface::SessionOptions;

        py::class_<SessionOptions>(m, "SessionOptions", "What every interface's Session is constructed with.")
            .def(py::init<>())
            .def_readwrite("start_time", &SessionOptions::start_time, "Time of the first estimate [s].")
            .def_readwrite("end_time", &SessionOptions::end_time, "Time to stop estimating at [s]. Negative: until the data ends (unbounded for live data).")
            .def_readwrite("period", &SessionOptions::period, "Between estimates, and so between node rows [s].")
            .def_readwrite("real_time", &SessionOptions::real_time, "Paced by the wall clock, or as fast as possible (deterministic).");

        py::classh<Viewer, PyViewer>(m, "Viewer",
                                     "Shows an estimator's results while or after a session runs it: Session.start(estimator, viewer) opens "
                                     "it, and Session.join() shows it until it is closed (see visualizer.Visualizer). Subclass it to write "
                                     "your own; the session calls every method from the thread calling start() and join().")
            .def(py::init<>())
            .def("open", &Viewer::open, py::arg("estimator"), py::arg("on_closed"),
                 "Sets up to show `estimator`, before estimation starts. Call `on_closed()` once closed, for whatever reason: the session then stops.")
            .def("isOnline", &Viewer::isOnline, "Whether playback follows the live estimate (True) or replays it once estimation has ended (False).")
            .def("beginPlayback", &Viewer::beginPlayback, py::arg("start_time"), py::arg("end_time"), py::arg("live_time_source") = nullptr,
                 "What to play: [start_time, end_time] [s]; online, `live_time_source()` is the current estimation time.")
            .def("run", &Viewer::run, py::call_guard<py::gil_scoped_release>(),
                 "Shows the results until closed; returns an exit code. Returns at once if requestClose() came first.")
            .def("requestClose", &Viewer::requestClose, py::call_guard<py::gil_scoped_release>(),
                 "Asks the viewer to close as soon as possible; safe from any thread.");

        py::class_<Session> session(m, "Session",
                                    "Runs an estimator, given to start(), on its own estimator and dataloader threads, delivering the "
                                    "measurements loaded into its queues as if live: as fast as possible, waiting for every dataloader, or in real "
                                    "time. Optionally shows it in a viewer: session.start(estimator, visualizer); session.join(). Dropping the "
                                    "session stops and joins it.");
        session.def(py::init<const SessionOptions &>(), py::arg("options") = SessionOptions())
            .def(
                "setTickCallback", [](Session &self, std::function<void(double)> callback) { self->setTickCallback(std::move(callback)); },
                py::arg("callback"), "Calls `callback(time)` on the estimator thread after every estimate.");
        bindSessionRun(session,
                       "Starts estimating with `estimator` (kept alive by the session): replays what is loaded into its measurement queues "
                       "on a dataloader thread, and starts the estimator thread. Raises ValueError for invalid options (e.g. a non-positive "
                       "period, or a batch-mode estimator in real time), RuntimeError if already started.");

        using Spacetime::Interface::SnapshotHistory;
        py::classh<SnapshotHistory>(m, "SnapshotHistory",
                                     "Estimator snapshots by time, e.g. recorded from a session's tick callback, to look up the estimate as "
                                     "of any past time. Thread-safe; keeps every snapshot recorded.")
            .def(py::init<>())
            .def(
                "record", [](SnapshotHistory &self, std::shared_ptr<Spacetime::EstimatorSnapshot> snapshot) { self.record(snapshot); },
                py::arg("snapshot"), "Appends `snapshot` (None is ignored); record in non-decreasing time order (not checked).")
            .def(
                "at", [](const SnapshotHistory &self, double time) { return std::const_pointer_cast<Spacetime::EstimatorSnapshot>(self.at(time)); },
                py::arg("time"), "The latest snapshot stamped at or before `time`, or None.")
            .def(
                "latest", [](const SnapshotHistory &self) { return std::const_pointer_cast<Spacetime::EstimatorSnapshot>(self.latest()); },
                "The most recently recorded snapshot, or None.")
            .def("size", &SnapshotHistory::size, "The number of recorded snapshots.");
    }
} // namespace SpacetimePython
