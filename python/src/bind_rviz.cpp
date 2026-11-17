/**
 * @file bind_rviz.cpp
 * @brief spacetime.visualizer.rviz: publishing the estimator's snapshots for rviz2.
 */

#include <memory>
#include <string>

// Before the bindings' common header: Python's headers define macros that clash with the library's.
#include <rclcpp/rclcpp.hpp>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/visualizer/rviz/SnapshotPublisherNode.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        namespace Rviz = Spacetime::Visualization::Rviz;
    } // namespace

    void bindRviz(py::module_ &m)
    {
        py::class_<Rviz::SnapshotPublisherOptions>(m, "SnapshotPublisherOptions", "Where and how often a SnapshotPublisher publishes.")
            .def(py::init<>())
            .def_readwrite("rate", &Rviz::SnapshotPublisherOptions::rate, "Checks for a new snapshot this many times per second (wall time); <= 0 publishes only on publishLatest().")
            .def_readwrite("topic_prefix", &Rviz::SnapshotPublisherOptions::topic_prefix, "Topics are <prefix>/<robot>/poses (PoseArray) and <prefix>/markers (MarkerArray), relative to the node's namespace unless absolute.")
            .def_readwrite("tf_topic", &Rviz::SnapshotPublisherOptions::tf_topic, "The TFMessage topic node frames are broadcast on.");

        py::class_<Rviz::RobotDisplayOptions>(m, "RobotDisplayOptions", "How one robot is published.")
            .def(py::init<>())
            .def_readwrite("frame_id", &Rviz::RobotDisplayOptions::frame_id, "The frame the robot's poses are in, e.g. 'aurora_field'.")
            .def_readwrite("radius", &Rviz::RobotDisplayOptions::radius, "Of the drawn tube [m].")
            .def_readwrite("extra_points", &Rviz::RobotDisplayOptions::extra_points, "Extra backbone points per node-to-node segment, for a smoother tube.")
            .def_readwrite("node_frames", &Rviz::RobotDisplayOptions::node_frames, "Broadcast each node on TF as <robot>/node_<i>.")
            .def_readwrite("covariance", &Rviz::RobotDisplayOptions::covariance,
                           "Draw the 99.7% position-covariance ellipsoids (needs EstimatorOptions.compute_covariance).");

        py::classh<Rviz::SnapshotPublisherNode, rclcpp::Node>(
            m, "SnapshotPublisher",
            "Publishes the estimator's latest snapshot for rviz2: each robot's backbone (PoseArray), tube, covariance ellipsoids "
            "and joints (MarkerArray), and node frames (TF). Run it on a session (Session.addNode).")
            .def(py::init([](Spacetime::Estimator &estimator, const std::shared_ptr<Spacetime::Ros2::RosClock> &clock, const Rviz::SnapshotPublisherOptions &options)
                          { return std::make_shared<Rviz::SnapshotPublisherNode>([&estimator]()
                                                                                 { return estimator.getSnapshot(); },
                                                                                 clock, options); }),
                 py::arg("estimator"), py::arg("clock"), py::arg("options") = Rviz::SnapshotPublisherOptions(), py::keep_alive<1, 2>(),
                 "Publishes `estimator`'s getSnapshot() (the estimator is kept alive). `clock` is the session's (Session.clock()), which stamps "
                 "the messages; snapshots are skipped until it has started. Raises ValueError for a null clock.")
            .def(
                "addRobot", [](Rviz::SnapshotPublisherNode &self, const std::string &name, const pStateful &prior, const Rviz::RobotDisplayOptions &options)
                {
                    if (auto wnoa = std::dynamic_pointer_cast<WnoaPrior>(prior))
                        self.addRobot(name, wnoa, options);
                    else if (auto wnov = std::dynamic_pointer_cast<WnovPrior>(prior))
                        self.addRobot(name, wnov, options);
                    else if (auto quasi_static = std::dynamic_pointer_cast<QuasiStaticPrior>(prior))
                        self.addRobot(name, quasi_static, options);
                    else if (auto end_effector = std::dynamic_pointer_cast<EndEffectorPrior>(prior))
                        self.addRobot(name, end_effector, options);
                    else
                        throw py::type_error("SnapshotPublisher.addRobot: the prior must be a built-in one (WNOA, WNOV, quasi-static or an end effector's).");
                },
                py::arg("name"), py::arg("prior"), py::arg("options") = Rviz::RobotDisplayOptions(),
                "Publishes the robot `prior` extends as `name` (a topic token, [A-Za-z][A-Za-z0-9_]*). Add robots and joints before the session starts. "
                "Raises ValueError for a taken or invalid name or a robot added already.")
            .def(
                "addJoint", [](Rviz::SnapshotPublisherNode &self, const std::string &name, const RobotJoint &joint, double radius)
                { self.addJoint(name, joint, radius); },
                py::arg("name"), py::arg("joint"), py::arg("radius") = 0.015,
                "Draws `joint` between its robots, added before it: a cylinder of `radius` [m] if its points are apart, else a sphere. "
                "Raises ValueError for a taken or invalid name, a robot not added, or robots in different frames.")
            .def(
                "addJoint", [](Rviz::SnapshotPublisherNode &self, const std::string &name, const EndEffectorJoint &joint, double radius)
                { self.addJoint(name, joint, radius); },
                py::arg("name"), py::arg("joint"), py::arg("radius") = 0.015, "Draws an end effector's joint; see the RobotJoint overload.")
            .def("publishLatest", &Rviz::SnapshotPublisherNode::publishLatest, py::call_guard<py::gil_scoped_release>(),
                 "Publishes the latest snapshot if it is new (and the clock has started); returns whether it did.");
    }
} // namespace SpacetimePython
