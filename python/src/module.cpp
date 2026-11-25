/**
 * @file module.cpp
 * @brief The _spacetime extension: one submodule per part of the library (see python/spacetime/__init__.py).
 */

#include "common.hpp"

PYBIND11_MODULE(_spacetime, m)
{
    // GTSAM's Python module registers every GTSAM type the bindings take or return.
    py::module_::import("gtsam");

    m.doc() = "Continuum robot state estimation in space and time, on GTSAM.";
    auto types = m.def_submodule("types", "Base classes of the library: models, stateful models, sensors, actuators, states, nodes, robots.");
    auto priors = m.def_submodule("priors", "Gaussian-process priors.");
    auto wnoa = priors.def_submodule("wnoa", "The white-noise-on-acceleration prior, its nodes and grid states.");
    auto wnov = priors.def_submodule("wnov", "The white-noise-on-velocity prior, its nodes and grid states.");
    priors.def_submodule("wnoa_quasi_static", "The quasi-static prior (WNOA in arclength only), its nodes and grid states.");
    auto joints = m.def_submodule("joints", "Joints between robots, and end effectors.");
    auto sensors = m.def_submodule("sensors", "Built-in sensors and their factors.");
    auto actuators = m.def_submodule("actuators", "Built-in actuators (e.g. the kinematic tendon) and their factors.");
    auto interface = m.def_submodule("interface", "Running the estimator: the standalone session, its options, and the snapshot history.");
    auto utilities = m.def_submodule("utilities", "Configuration loading and result export.");

    SpacetimePython::bindTypes(types);
    SpacetimePython::bindPriors(priors);
    SpacetimePython::bindJoints(joints);
    SpacetimePython::bindSensors(sensors);
    SpacetimePython::bindActuators(actuators);
    SpacetimePython::bindEstimator(m);
    SpacetimePython::bindInterface(interface);
    SpacetimePython::bindUtilities(utilities);
#ifdef SPACETIME_PYTHON_ROS2
    auto ros2 = interface.def_submodule("ros2", "Live (or rosbag) estimation on ROS2 nodes.");
    SpacetimePython::bindRos2(ros2);
#endif
#if defined(SPACETIME_PYTHON_VISUALIZER) || defined(SPACETIME_PYTHON_ROS2)
    auto visualizer = m.def_submodule("visualizer", "Visualizing the estimate: the Qt/VTK visualizer and/or rviz2 publishing.");
#endif
#ifdef SPACETIME_PYTHON_VISUALIZER
    SpacetimePython::bindVisualizer(visualizer);
#endif
#ifdef SPACETIME_PYTHON_ROS2
    auto rviz = visualizer.def_submodule("rviz", "Publishing the estimator's snapshots for rviz2.");
    SpacetimePython::bindRviz(rviz);
#endif
}
