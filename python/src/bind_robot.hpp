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
    /// `target.lock(type, s, value, time)` with value a gtsam.Pose3 or a 6-vector, the two variable types of the nodes,
    /// or None for the initial guess (see ContinuumRobotState::lockInitial).
    template <typename Target>
    void lockByType(Target &target, const std::string &type, double s, const py::object &value, std::optional<double> time)
    {
        if (value.is_none())
            return target.lockInitial(type, s, time);
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
            .def(py::init<double, unsigned int>(), py::arg("length"), py::arg("spatial_dim"),
                 "`spatial_dim` (>= 2, not checked) nodes per row, evenly spaced over [0, length] [m].")
            .def(py::init<std::vector<double>>(), py::arg("arclengths"), "Nodes at the given ascending, non-empty `arclengths` [m]; the length is the last one.")
            .def_readonly("length", &StateT::length, "Robot length [m].")
            .def_readonly("spatial_dim", &StateT::spatial_dim, "Nodes per row.")
            .def_readonly("arclengths", &StateT::arclengths, "Node arclengths [m], ascending.")
            .def("lock", &lockByType<StateT>, py::arg("type"), py::arg("s"), py::arg("value") = py::none(), py::arg("t") = std::nullopt,
                 "Locks variable `type` (e.g. \"pose\", \"strain\") of the node at arclength `s` (-1 for the tip) to `value` "
                 "(a gtsam.Pose3 or a 6-vector), in the row at time `t` or in every row; applies to rows created after the call. "
                 "Without a value, it is locked at the initial guess the prior gives it when the node is first added (e.g. the "
                 "boundary NodeValues' strain, straight by default). "
                 "Raises ValueError for an `s` that is not a node arclength, an unknown variable or a value of the wrong type.")
            .def("snapshotEstimationNodes", &StateT::snapshotEstimationNodes,
                 "A copy of every node, by flat index; safe to call while the estimator thread adds rows.")
            .def("getNextSpatialNode", &StateT::getNextSpatialNode, py::arg("node"),
                 "The next node along the robot, or None at the tip; raises RuntimeError if `node` is not in the state.")
            .def("getPreviousSpatialNode", &StateT::getPreviousSpatialNode, py::arg("node"), "The previous node along the robot, or None at the base.")
            .def("getNextTemporalNode", &StateT::getNextTemporalNode, py::arg("node"), "The node at the same arclength in the next row, or None in the latest row.")
            .def("getPreviousTemporalNode", &StateT::getPreviousTemporalNode, py::arg("node"),
                 "The node at the same arclength in the previous row, or None in the first row.")
            .def("getClosestShiftedNode", &StateT::getClosestShiftedNode, py::arg("arclength"), py::arg("time"), py::arg("spatial_direction"),
                 py::arg("temporal_direction"),
                 "The node nearest (L1 in arclength + time) to (arclength, time) on the side each direction selects "
                 "(< 0 at or below, > 0 at or above, 0 either), or None.")
            .def("getNodeIndex", &StateT::getNodeIndex, py::arg("spatial_index"), py::arg("temporal_index"),
                 "The flat index of the node at a grid index, or 2**32 - 1 if there is none.")
            .def("getNodeArclengthTime", &StateT::getNodeArclengthTime, py::arg("index"), "(arclength, time) of the node at flat `index`, or (-1, -1) if there is none.");
    }
} // namespace SpacetimePython

#endif // SPACETIME_PYTHON_BIND_ROBOT_HPP
