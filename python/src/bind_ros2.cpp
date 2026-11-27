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
#include "session.hpp"

namespace SpacetimePython
{
    namespace
    {
        namespace Ros2 = Spacetime::Interface::Ros2;

        using Session = PySession<Spacetime::Interface::INTERFACE_ROS2>;

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

        py::classh<Ros2::SensorNode, rclcpp::Node>(m, "SensorNode", "A node feeding one measurement queue from ROS messages, stamped in estimation time. Stamps before the clock "
                                                                  "starts or its start time, or not newer than the last, are dropped.")
            .def("queue", &Ros2::SensorNode::queue, "The measurement queue it feeds.")
            .def("messagesReceived", &Ros2::SensorNode::messagesReceived, "Messages received, used or not.")
            .def("measurementsAdded", &Ros2::SensorNode::measurementsAdded, "Measurements queued.")
            .def("measurementsDropped", &Ros2::SensorNode::measurementsDropped, "Relevant messages dropped (see the node's doc).");

        py::classh<Ros2::PoseSensorNode, Ros2::SensorNode>(m, "PoseSensorNode", "Queues a pose measurement of the robot for each transform of its TF edge (direct edge only, no chain lookup), "
                                                                "stamped in estimation time. Stamps before the clock starts or its start time, or not newer than the last, are dropped.")
            .def_static("bodyPose", &Ros2::PoseSensorNode::bodyPose, py::arg("T_is"), py::arg("T_bs"), "T_ib = T_is * T_bs^-1.")
            .def("options", &Ros2::PoseSensorNode::options, "The node's options.");

        py::enum_<Ros2::ImuMeasurement>(m, "ImuMeasurement", "Which measurement an ImuSensorNode makes of each sensor_msgs/Imu message.")
            .value("Gyro", Ros2::ImuMeasurement::Gyro, "3x1 angular_velocity [rad/s], for a gyroscope sensor.")
            .value("Accelerometer", Ros2::ImuMeasurement::Accelerometer, "3x1 linear_acceleration (specific force) [m/s^2], for an accelerometer sensor.")
            .value("IMU", Ros2::ImuMeasurement::IMU, "6x1 [angular_velocity; linear_acceleration], for an IMU sensor.");

        py::class_<Ros2::ImuSensorOptions>(m, "ImuSensorOptions", "The sensor_msgs/Imu topic an IMU, gyroscope or accelerometer node listens to, and the measurements it makes of it.")
            .def(py::init<>())
            .def_readwrite("topic", &Ros2::ImuSensorOptions::topic, "The sensor_msgs/Imu topic listened on.")
            .def_readwrite("frame_id", &Ros2::ImuSensorOptions::frame_id, "If set, only messages of this header.frame_id are used.")
            .def_readwrite("arclength", &Ros2::ImuSensorOptions::arclength, "Where along the robot the sensor is attached [m].")
            .def_readwrite("gyro_covariance", &Ros2::ImuSensorOptions::gyro_covariance, "3x3 angular velocity covariance, sensor frame [(rad/s)^2].")
            .def_readwrite("accelerometer_covariance", &Ros2::ImuSensorOptions::accelerometer_covariance, "3x3 specific force covariance, sensor frame [(m/s^2)^2].")
            .def_readwrite("use_message_covariance", &Ros2::ImuSensorOptions::use_message_covariance,
                           "Use each message's covariance instead, unless unknown (first entry -1, or all zeros) or invalid.")
            .def_readwrite("mask", &Ros2::ImuSensorOptions::mask, "[angular velocity; specific force] components measured (1) or ignored (0); a gyro or accelerometer node uses its half.");

        py::classh<Ros2::ImuSensorNode, Ros2::SensorNode>(m, "ImuSensorNode", "Queues a gyroscope, accelerometer or IMU measurement (see ImuMeasurement) for each sensor_msgs/Imu message, "
                                                              "in the message's (sensor) frame, stamped in estimation time.")
            .def("measurement", &Ros2::ImuSensorNode::measurement, "The ImuMeasurement it makes.")
            .def("options", &Ros2::ImuSensorNode::options, "The node's options.");

        py::class_<Ros2::KinematicTendonOptions>(m, "KinematicTendonOptions", "The sensor_msgs/JointState joint a tendon node listens to, and the tendon lengths it measures from it.")
            .def(py::init<>())
            .def_readwrite("topic", &Ros2::KinematicTendonOptions::topic, "The sensor_msgs/JointState topic listened on.")
            .def_readwrite("joint_name", &Ros2::KinematicTendonOptions::joint_name, "The joint (in JointState.name) whose position measures the tendon.")
            .def_readwrite("variance", &Ros2::KinematicTendonOptions::variance, "Variance of the measured tendon length [m^2].")
            .def_readwrite("scale", &Ros2::KinematicTendonOptions::scale, "The tendon length is scale * position + length_offset [m].")
            .def_readwrite("length_offset", &Ros2::KinematicTendonOptions::length_offset, "See scale [m].");

        py::classh<Ros2::KinematicTendonNode, Ros2::SensorNode>(m, "KinematicTendonNode", "Queues the tendon length of its joint for each sensor_msgs/JointState message naming it, "
                                                                    "stamped in estimation time; a message naming it without a position is dropped.")
            .def("options", &Ros2::KinematicTendonNode::options, "The node's options.");

        py::class_<Ros2::SessionRosOptions>(m, "SessionRosOptions", "ROS-specific options of a Session.")
            .def(py::init<>())
            .def_readwrite("node_prefix", &Ros2::SessionRosOptions::node_prefix, "The estimator node is <node_prefix>_estimator.")
            .def_readwrite("nice_delta", &Ros2::SessionRosOptions::nice_delta, "How much to lower the thread of every node but the estimator's (a nice-value increment on Linux).")
            .def_readwrite("realtime_estimator", &Ros2::SessionRosOptions::realtime_estimator, "Request real-time scheduling for the estimator's thread (SCHED_FIFO on Linux, which needs the privilege); only warns if refused.")
            .def_property("use_sim_time", &useSimTime, &setUseSimTime, "Every node the session creates follows /clock (e.g. a rosbag played with --clock).");

        py::class_<Session> session(m, "Session",
                            "Runs an estimator, given to start(), on ROS2 nodes: the estimator's node on a high-priority thread, the "
                            "sensor nodes and any other node (e.g. an rviz SnapshotPublisher) on another. Always real time: estimation follows "
                            "the ROS clock (live, or a replayed /clock with use_sim_time). Needs init(). Dropping the session stops and joins it. Optionally shows it in a viewer: session.start(estimator, visualizer); session.join().");
        session.def(py::init<const Spacetime::Interface::SessionOptions &, const Ros2::SessionRosOptions &>(),
                 py::arg("options") = Spacetime::Interface::SessionOptions(), py::arg("ros_options") = Ros2::SessionRosOptions())
            .def(
                "addPoseSensor", [](Session &self, Spacetime::Types::pMeasurementQueueType queue, const Ros2::PoseSensorOptions &options)
                { return self->addPoseSensor(std::move(queue), options); },
                py::arg("sensor"), py::arg("options"),
                "Feeds `sensor` (e.g. a sensors.PoseSensor) from a TF edge; returns its PoseSensorNode. "
                "Raises ValueError for invalid options (e.g. an empty frame, or a covariance that is not symmetric with a positive diagonal).")
            .def(
                "addGyro", [](Session &self, Spacetime::Types::pMeasurementQueueType queue, const Ros2::ImuSensorOptions &options)
                { return self->addGyro(std::move(queue), options); },
                py::arg("sensor"), py::arg("options"),
                "Feeds `sensor` (e.g. a sensors.GyroSensor) with the angular velocity of sensor_msgs/Imu messages; returns its ImuSensorNode. "
                "Raises ValueError for invalid options.")
            .def(
                "addAccelerometer", [](Session &self, Spacetime::Types::pMeasurementQueueType queue, const Ros2::ImuSensorOptions &options)
                { return self->addAccelerometer(std::move(queue), options); },
                py::arg("sensor"), py::arg("options"),
                "Feeds `sensor` (e.g. a sensors.AccelerometerSensor) with the linear acceleration of sensor_msgs/Imu messages; returns its "
                "ImuSensorNode. Raises ValueError for invalid options.")
            .def(
                "addIMU", [](Session &self, Spacetime::Types::pMeasurementQueueType queue, const Ros2::ImuSensorOptions &options)
                { return self->addIMU(std::move(queue), options); },
                py::arg("sensor"), py::arg("options"),
                "Feeds `sensor` (e.g. a sensors.IMUSensor) with the angular velocity and linear acceleration of sensor_msgs/Imu messages; "
                "returns its ImuSensorNode. Raises ValueError for invalid options.")
            .def(
                "addKinematicTendon", [](Session &self, Spacetime::Types::pMeasurementQueueType queue, const Ros2::KinematicTendonOptions &options)
                { return self->addKinematicTendon(std::move(queue), options); },
                py::arg("actuator"), py::arg("options"),
                "Feeds `actuator` (e.g. an actuators.KinematicTendon) with one joint of sensor_msgs/JointState messages; returns its "
                "KinematicTendonNode. Raises ValueError for invalid options (e.g. no joint name).")
            .def(
                "addNode", [](Session &self, const std::shared_ptr<rclcpp::Node> &node)
                { self->addNode(node); },
                py::arg("node"), "Runs `node` (e.g. an rviz SnapshotPublisher) on the lower-priority thread; raises ValueError for None.")
            .def("unfedQueues", [](Session &self, const Spacetime::Estimator &estimator)
                 { return self->unfedQueues(estimator); }, py::arg("estimator"),
                 "`estimator`'s measurement queues that no sensor node feeds; start() warns about them.")
            .def(
                "clock", [](Session &self)
                { return std::const_pointer_cast<Ros2::RosClock>(self->clock()); },
                "The estimation clock (RosClock), e.g. for a SnapshotPublisher.")
            .def(
                "setTickCallback", [](Session &self, std::function<void(double)> callback)
                { self->setTickCallback(std::move(callback)); },
                py::arg("callback"), "Calls `callback(time)` on the estimator thread after every estimate.");
        bindSessionRun(session,
                       "Starts estimating with `estimator` (kept alive by the session) on the node threads. Raises ValueError for a "
                       "non-positive period or a batch-mode estimator, RuntimeError if already started. The run ends at the end time, "
                       "on requestStop, or on rclcpp shutdown (e.g. Ctrl-C).");

        m.def(
            "waitForTransform", [](const std::string &parent, const std::string &child, double timeout)
            { return Ros2::waitForTransform(parent, child, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(timeout))); },
            py::arg("parent"), py::arg("child"), py::arg("timeout"), py::call_guard<py::gil_scoped_release>(),
            "The latest pose of `child` in `parent` from TF (/tf and /tf_static), waiting up to `timeout` seconds on a temporary node "
            "(so call it before the session starts); None on timeout or rclcpp shutdown.");
    }
} // namespace SpacetimePython
