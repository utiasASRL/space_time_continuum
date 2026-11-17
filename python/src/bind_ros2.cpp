/**
 * @file bind_ros2.cpp
 * @brief spacetime.interface.ros2: live (or rosbag) estimation on ROS2 nodes. The nodes are
 *        rclcpp nodes, so rclcpp is initialized here (init/shutdown), side by side with any
 *        rclpy context in the same process.
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

// Before the bindings' common header: Python's headers define macros that clash with the library's.
#include <rclcpp/rclcpp.hpp>

#include <spacetime/interface/ros2/Interface.hpp>
#include <spacetime/interface/ros2/WaitForTransform.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        namespace Ros2 = Spacetime::Ros2;

        /// A ROS2 session, stopped and joined without the GIL when Python drops it: its threads may be waiting for the GIL.
        class Session
        {
        public:
            Session(Spacetime::Estimator &estimator, const Spacetime::Interface::SessionOptions &options, const Ros2::SessionRosOptions &ros_options)
                : m_session(std::make_unique<Ros2::Session>(estimator, options, ros_options)) {}

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

            Ros2::Session &operator*() { return *m_session; }
            Ros2::Session *operator->() { return m_session.get(); }

        private:
            std::unique_ptr<Ros2::Session> m_session;
        };

        bool useSimTime(const Ros2::SessionRosOptions &options)
        {
            for (const auto &parameter : options.node_options.parameter_overrides())
                if (parameter.get_name() == "use_sim_time")
                    return parameter.as_bool();
            return false;
        }

        void setUseSimTime(Ros2::SessionRosOptions &options, bool use_sim_time)
        {
            auto &overrides = options.node_options.parameter_overrides();
            std::erase_if(overrides, [](const rclcpp::Parameter &parameter)
                          { return parameter.get_name() == "use_sim_time"; });
            overrides.emplace_back("use_sim_time", use_sim_time);
        }
    } // namespace

    void bindRos2(py::module_ &m)
    {
        m.def(
            "init", [](const std::vector<std::string> &args)
            {
                if (rclcpp::ok())
                    return;
                std::vector<const char *> argv{"spacetime"};
                for (const auto &arg : args)
                    argv.push_back(arg.c_str());
                rclcpp::init(static_cast<int>(argv.size()), argv.data()); },
            py::arg("args") = std::vector<std::string>(),
            "Initializes rclcpp (once; ROS arguments such as --ros-args -p use_sim_time:=true apply to every node). Its SIGINT handler shuts it down, ending a running session.");
        m.def("shutdown", []()
              { rclcpp::shutdown(); }, "Shuts rclcpp down: running sessions end.");
        m.def("ok", []()
              { return rclcpp::ok(); }, "Whether rclcpp is initialized and not shut down.");

        py::classh<rclcpp::Node>(m, "Node", "An rclcpp node (run by a Session, see Session.addNode).")
            .def("get_name", [](const rclcpp::Node &self)
                 { return std::string(self.get_name()); }, "The node's name.");

        py::classh<Ros2::RosClock>(m, "RosClock", "Estimation time from a ROS clock (live, or a replayed /clock), relative to its first reading.")
            .def("started", &Ros2::RosClock::started, "Whether the origin is set (the first non-zero reading; under sim time, after the first /clock).")
            .def("now", &Ros2::RosClock::now, "The current estimation time [s]: start time + (ROS time - origin); the start time until started.")
            .def("startTime", &Ros2::RosClock::startTime, "The estimation time of the origin [s].")
            .def("isRealTime", &Ros2::RosClock::isRealTime, "Always True: estimation follows ROS time, live or replayed.");

        py::class_<Ros2::PoseSensorOptions>(m, "PoseSensorOptions", "The TF edge a pose sensor node listens to, and the measurements it makes of it.")
            .def(py::init<>())
            .def_readwrite("parent_frame", &Ros2::PoseSensorOptions::parent_frame, "The TF edge's parent frame, e.g. 'aurora_field'.")
            .def_readwrite("child_frame", &Ros2::PoseSensorOptions::child_frame, "The sensor's frame.")
            .def_readwrite("offset", &Ros2::PoseSensorOptions::offset, "T_bs: the sensor in the robot's frame at `arclength`.")
            .def_readwrite("arclength", &Ros2::PoseSensorOptions::arclength, "Where along the robot the sensor is attached [m].")
            .def_readwrite("covariance", &Ros2::PoseSensorOptions::covariance, "6x6, [rot, trans] in the robot's frame.")
            .def_readwrite("mask", &Ros2::PoseSensorOptions::mask, "[rot, trans] components measured (1) or ignored (0), e.g. [1, 1, 0, 1, 1, 1] for a 5DoF sensor.")
            .def_readwrite("body_frame", &Ros2::PoseSensorOptions::body_frame, "If set, the robot's pose is broadcast on TF as parent_frame -> body_frame.")
            .def_readwrite("topic", &Ros2::PoseSensorOptions::topic, "The TFMessage topic listened (and broadcast) on.");

        py::classh<Ros2::PoseSensorNode, rclcpp::Node>(m, "PoseSensorNode", "Queues a pose measurement of the robot for each transform of its TF edge (direct edge only, no chain lookup), "
                                                          "stamped in estimation time. Stamps before the clock starts or its start time, or not newer than the last, are dropped.")
            .def_static("bodyPose", &Ros2::PoseSensorNode::bodyPose, py::arg("T_is"), py::arg("T_bs"), "T_ib = T_is * T_bs^-1.")
            .def("options", &Ros2::PoseSensorNode::options, "The node's options.")
            .def("queue", &Ros2::PoseSensorNode::queue, "The measurement queue it feeds.")
            .def("messagesReceived", &Ros2::PoseSensorNode::messagesReceived, "TF messages received, matching or not.")
            .def("measurementsAdded", &Ros2::PoseSensorNode::measurementsAdded, "Measurements queued.")
            .def("measurementsDropped", &Ros2::PoseSensorNode::measurementsDropped, "Transforms of the edge dropped (see the class doc).");

        py::class_<Ros2::SessionRosOptions>(m, "SessionRosOptions", "ROS-specific options of a Session.")
            .def(py::init<>())
            .def_readwrite("node_prefix", &Ros2::SessionRosOptions::node_prefix, "The estimator node is <node_prefix>_estimator.")
            .def_readwrite("nice_delta", &Ros2::SessionRosOptions::nice_delta, "Added to the nice value of every node's thread but the estimator's.")
            .def_readwrite("realtime_estimator", &Ros2::SessionRosOptions::realtime_estimator, "Request SCHED_FIFO for the estimator's thread.")
            .def_property("use_sim_time", &useSimTime, &setUseSimTime, "Every node the session creates follows /clock (e.g. a rosbag played with --clock).");

        py::class_<Session>(m, "Session",
                            "Runs `estimator` (kept alive by the session) on ROS2 nodes: the estimator's node on a high-priority thread, the "
                            "sensor nodes and any other node (e.g. an rviz SnapshotPublisher) on another. Always real time: estimation follows "
                            "the ROS clock (live, or a replayed /clock with use_sim_time). Needs init(). Dropping the session stops and joins it.")
            .def(py::init<Spacetime::Estimator &, const Spacetime::Interface::SessionOptions &, const Ros2::SessionRosOptions &>(),
                 py::arg("estimator"), py::arg("options"), py::arg("ros_options") = Ros2::SessionRosOptions(), py::keep_alive<1, 2>())
            .def(
                "addMeasurementQueues", [](Session &self, const std::vector<Spacetime::Types::pMeasurementQueueType> &queues)
                { self->addMeasurementQueues(queues); },
                py::arg("queues"), "Records the queues to feed, to warn at start() about any no node feeds.")
            .def(
                "addPoseSensor", [](Session &self, Spacetime::Types::pMeasurementQueueType queue, const Ros2::PoseSensorOptions &options)
                { return self->addPoseSensor(std::move(queue), options); },
                py::arg("sensor"), py::arg("options"),
                "Feeds `sensor` (e.g. a sensors.PoseSensor) from a TF edge; returns its PoseSensorNode. "
                "Raises ValueError for invalid options (e.g. an empty frame, or a covariance that is not symmetric with a positive diagonal).")
            .def(
                "addNode", [](Session &self, const std::shared_ptr<rclcpp::Node> &node)
                { self->addNode(node); },
                py::arg("node"), "Runs `node` (e.g. an rviz SnapshotPublisher) on the lower-priority thread; raises ValueError for None.")
            .def("unfedQueues", [](Session &self)
                 { return self->unfedQueues(); }, "The queues given to addMeasurementQueues that no sensor node feeds.")
            .def(
                "clock", [](Session &self)
                { return std::const_pointer_cast<Ros2::RosClock>(self->clock()); },
                "The estimation clock (RosClock), e.g. for a SnapshotPublisher.")
            .def(
                "setTickCallback", [](Session &self, std::function<void(double)> callback)
                { self->setTickCallback(std::move(callback)); },
                py::arg("callback"), "Calls `callback(time)` on the estimator thread after every estimate.")
            .def("start", [](Session &self)
                 { self->start(); }, py::call_guard<py::gil_scoped_release>(),
                 "Starts the node threads. Raises ValueError for a non-positive period or a batch-mode estimator.")
            .def("requestStop", [](Session &self)
                 { self->requestStop(); }, "Asks the run to end as soon as possible; safe from any thread.")
            .def("join", [](Session &self)
                 { self->join(); }, py::call_guard<py::gil_scoped_release>(),
                 "Waits for the run to end (end time, requestStop, or rclcpp shutdown, e.g. Ctrl-C); raises the first error raised while estimating.")
            .def("finished", [](Session &self)
                 { return self->finished(); }, "Whether the run has ended.")
            .def("endTime", [](Session &self)
                 { return self->endTime(); }, "The end time [s]: inf if SessionOptions.end_time is negative.")
            .def("now", [](Session &self)
                 { return self->now(); }, "The current estimation time [s] (see RosClock); safe from any thread.")
            .def("isRealTime", [](Session &self)
                 { return self->isRealTime(); }, "Always True.");

        m.def(
            "waitForTransform", [](const std::string &parent, const std::string &child, double timeout)
            { return Ros2::waitForTransform(parent, child, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(timeout))); },
            py::arg("parent"), py::arg("child"), py::arg("timeout"), py::call_guard<py::gil_scoped_release>(),
            "The latest pose of `child` in `parent` from TF (/tf and /tf_static), waiting up to `timeout` seconds on a temporary node "
            "(so call it before the session starts); None on timeout or rclcpp shutdown.");
    }
} // namespace SpacetimePython
