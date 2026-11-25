/**
 * @file bind_interface.cpp
 * @brief spacetime.interface: the standalone session that runs an estimator over recorded (or
 *        synthetic) measurements, and the snapshot history.
 */

#include <memory>

#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/Interfaces.hpp>
#include <spacetime/interface/SnapshotHistory.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        /// A standalone session, stopped and joined without the GIL when Python drops it: its threads may be waiting for the GIL.
        class Session
        {
        public:
            explicit Session(const Spacetime::Interface::SessionOptions &options)
                : m_session(std::make_unique<Spacetime::Interface::Standalone::Session>(options)) {}

            ~Session()
            {
                py::gil_scoped_release release;
                m_session->requestStop();
                try
                {
                    m_session->join();
                }
                catch (...)
                {
                    // Already reported by join(), or abandoned with the session.
                }
                m_session.reset();
            }

            Spacetime::Interface::Standalone::Session &operator*() { return *m_session; }
            Spacetime::Interface::Standalone::Session *operator->() { return m_session.get(); }

        private:
            std::unique_ptr<Spacetime::Interface::Standalone::Session> m_session;
        };
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

        py::class_<Session>(m, "Session",
                            "Runs an estimator, given to start(), on its own estimator and dataloader threads, delivering the "
                            "measurements loaded into its queues as if live: as fast as possible, waiting for every dataloader, or in real time. "
                            "Dropping the session stops and joins it.")
            .def(py::init<const SessionOptions &>(), py::arg("options") = SessionOptions())
            .def(
                "setTickCallback", [](Session &self, std::function<void(double)> callback) { self->setTickCallback(std::move(callback)); },
                py::arg("callback"), "Calls `callback(time)` on the estimator thread after every estimate.")
            .def("start", [](Session &self, Spacetime::Estimator &estimator) { self->start(estimator); }, py::arg("estimator"),
                 py::call_guard<py::gil_scoped_release>(), py::keep_alive<1, 2>(),
                 "Starts estimating with `estimator` (kept alive by the session): replays what is loaded into its measurement queues "
                 "on a dataloader thread, and starts the estimator thread. Raises ValueError for invalid options (e.g. a non-positive "
                 "period, or a batch-mode estimator in real time), RuntimeError if already started.")
            .def("requestStop", [](Session &self) { self->requestStop(); }, "Asks the run to end as soon as possible; safe from any thread.")
            .def("join", [](Session &self) { self->join(); }, py::call_guard<py::gil_scoped_release>(),
                 "Waits for the session to finish; raises the first error of any of its threads.")
            .def("finished", [](Session &self) { return self->finished(); }, "Whether the run has ended (False if never started).")
            .def("endTime", [](Session &self) { return self->endTime(); }, "The end time actually used [s] (possibly inf); valid after start().")
            .def("now", [](Session &self) { return self->now(); }, "The current estimation time [s]; safe from any thread.")
            .def("isRealTime", [](Session &self) { return self->isRealTime(); }, "Whether SessionOptions.real_time selected the wall clock.");

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
