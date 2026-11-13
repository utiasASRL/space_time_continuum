#ifndef SPACETIME_PYTHON_BIND_ROBOT_HPP
#define SPACETIME_PYTHON_BIND_ROBOT_HPP

/**
 * @file bind_robot.hpp
 * @brief Binds Types::ContinuumRobotState<NodeT> for one node type (WNOA, WNOV, quasi-static,
 *        end effector).
 */

#include <optional>
#include <string>
#include <vector>

#include "common.hpp"

namespace SpacetimePython
{
    /// `target.lock(type, s, value, time)` with value a gtsam.Pose3 or a 6-vector, the two variable types of the nodes.
    template <typename Target>
    void lockByType(Target &target, const std::string &type, double s, const py::object &value, std::optional<double> time)
    {
        if (py::isinstance<gtsam::Pose3>(value))
            return target.template lock<gtsam::Pose3>(type, s, value.cast<gtsam::Pose3>(), time);
        Vector6 vector;
        try
        {
            vector = value.cast<Vector6>();
        }
        catch (const py::cast_error &)
        {
            throw py::type_error("lock: the value must be a gtsam.Pose3 or a 6-vector.");
        }
        target.template lock<Vector6>(type, s, vector, time);
    }

    /// Binds the grid state of node type NodeT into `m` as `name`.
    template <typename NodeT>
    void bindContinuumRobotState(py::module_ &m, const char *name)
    {
        using StateT = Spacetime::Types::ContinuumRobotState<NodeT>;

        py::classh<StateT, Spacetime::Types::State>(m, name, "A robot's grid of nodes: the same arclengths at every estimation time.")
            .def(py::init<double, unsigned int>(), py::arg("length"), py::arg("spatial_dim"))
            .def(py::init<std::vector<double>>(), py::arg("arclengths"))
            .def_readonly("length", &StateT::length)
            .def_readonly("spatial_dim", &StateT::spatial_dim)
            .def_readonly("arclengths", &StateT::arclengths)
            .def("lock", &lockByType<StateT>, py::arg("type"), py::arg("s"), py::arg("value"), py::arg("t") = std::nullopt,
                 "Locks variable `type` (e.g. \"pose\", \"strain\") of the node at arclength `s` (-1 for the tip) to `value`, "
                 "at time `t` or at every time.")
            .def("snapshotEstimationNodes", &StateT::snapshotEstimationNodes)
            .def("getNextSpatialNode", &StateT::getNextSpatialNode, py::arg("node"))
            .def("getPreviousSpatialNode", &StateT::getPreviousSpatialNode, py::arg("node"))
            .def("getNextTemporalNode", &StateT::getNextTemporalNode, py::arg("node"))
            .def("getPreviousTemporalNode", &StateT::getPreviousTemporalNode, py::arg("node"))
            .def("getClosestShiftedNode", &StateT::getClosestShiftedNode, py::arg("arclength"), py::arg("time"), py::arg("spatial_direction"),
                 py::arg("temporal_direction"))
            .def("getNodeIndex", &StateT::getNodeIndex, py::arg("spatial_index"), py::arg("temporal_index"))
            .def("getNodeArclengthTime", &StateT::getNodeArclengthTime, py::arg("index"));
    }
} // namespace SpacetimePython

#endif // SPACETIME_PYTHON_BIND_ROBOT_HPP
