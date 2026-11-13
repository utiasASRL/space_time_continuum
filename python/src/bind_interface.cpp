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
            Session(Spacetime::Estimator &estimator, const Spacetime::Interface::SessionOptions &options)
                : m_session(std::make_unique<Spacetime::Standalone::Session>(estimator, options)) {}

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

            Spacetime::Standalone::Session &operator*() { return *m_session; }
            Spacetime::Standalone::Session *operator->() { return m_session.get(); }

        private:
            std::unique_ptr<Spacetime::Standalone::Session> m_session;
        };
    } // namespace

    void bindInterface(py::module_ &m)
    {
        using Spacetime::Interface::SessionOptions;

        py::class_<SessionOptions>(m, "SessionOptions")
            .def(py::init<>())
            .def_readwrite("start_time", &SessionOptions::start_time)
            .def_readwrite("end_time", &SessionOptions::end_time, "Negative: until the last measurement.")
            .def_readwrite("period", &SessionOptions::period, "Between estimates [s].")
            .def_readwrite("real_time", &SessionOptions::real_time, "Paced by the wall clock, or as fast as possible.");

        py::class_<Session>(m, "Session",
                            "Runs `estimator` (which must outlive it) on its own threads, delivering its measurement queues as if live.")
            .def(py::init<Spacetime::Estimator &, const SessionOptions &>(), py::arg("estimator"), py::arg("options"), py::keep_alive<1, 2>())
            .def(
                "addMeasurementQueues", [](Session &self, const std::vector<Spacetime::Types::pMeasurementQueueType> &queues)
                { self->addMeasurementQueues(queues); },
                py::arg("queues"))
            .def(
                "setTickCallback", [](Session &self, std::function<void(double)> callback) { self->setTickCallback(std::move(callback)); },
                py::arg("callback"), "Calls `callback(time)` on the estimator thread after every estimate.")
            .def("start", [](Session &self) { self->start(); }, py::call_guard<py::gil_scoped_release>())
            .def("requestStop", [](Session &self) { self->requestStop(); })
            .def("join", [](Session &self) { self->join(); }, py::call_guard<py::gil_scoped_release>(),
                 "Waits for the session to finish; raises the first error of any of its threads.")
            .def("finished", [](Session &self) { return self->finished(); })
            .def("endTime", [](Session &self) { return self->endTime(); })
            .def("now", [](Session &self) { return self->now(); })
            .def("isRealTime", [](Session &self) { return self->isRealTime(); });

        using Spacetime::Interface::SnapshotHistory;
        py::classh<SnapshotHistory>(m, "SnapshotHistory", "Estimator snapshots by time.")
            .def(py::init<>())
            .def(
                "record", [](SnapshotHistory &self, std::shared_ptr<Spacetime::EstimatorSnapshot> snapshot) { self.record(snapshot); },
                py::arg("snapshot"))
            .def(
                "at", [](const SnapshotHistory &self, double time) { return std::const_pointer_cast<Spacetime::EstimatorSnapshot>(self.at(time)); },
                py::arg("time"))
            .def("latest", [](const SnapshotHistory &self) { return std::const_pointer_cast<Spacetime::EstimatorSnapshot>(self.latest()); })
            .def("size", &SnapshotHistory::size);
    }
} // namespace SpacetimePython
