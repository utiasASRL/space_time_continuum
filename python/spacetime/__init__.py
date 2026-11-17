"""Continuum robot state estimation in space and time, on GTSAM.

The Python interface to the spacetime C++ library. GTSAM objects (gtsam.Pose3, gtsam.Values,
factors, noise models, ...) are GTSAM's own Python types, passed to and from the library as is.

Submodules:
    types       Base classes, each subclassable in Python: Model, StatefulModel, SensorBase,
                ActuatorBase, State, NodeBase, NodeValues; Robot (a list of models),
                MeasurementBase, KeyManager.
    priors.wnoa The white-noise-on-acceleration prior, its node values, nodes and grid states.
    priors.wnov The white-noise-on-velocity prior, its node values, nodes and grid states.
    priors.wnoa_quasi_static
                The quasi-static prior (WNOA in arclength only, for EstimatorOptions.quasi_static).
    joints      Joints between robots, and end effectors.
    sensors     Built-in sensors and their factors.
    interface   The standalone Session that runs an estimator, its SessionOptions, and the
                SnapshotHistory of its estimates.
    interface.ros2
                Live (or rosbag) estimation on ROS2 nodes (if built with InterfaceROS2): init/shutdown
                rclcpp, the ROS2 Session, pose sensor nodes fed by TF, waitForTransform.
    utilities   Configuration loading and result export.
    visualizer  The Qt/VTK visualizer (if built with it), and visualizer.rviz: publishing the
                estimate for rviz2 (if built with InterfaceROS2).

Conventions (as in the C++ library): poses are T_ib (the body frame in the inertial frame),
and tangent vectors (strain, velocity) are in GTSAM's [rotational, translational] order.
"""

import sys

import gtsam  # noqa: F401  -- registers the GTSAM types the extension takes and returns

from . import _spacetime
from ._spacetime import (  # noqa: F401
    BatchTimings,
    BetweenFactorVector6,
    Estimator,
    EstimatorOptions,
    EstimatorSnapshot,
    LockedFactor,
    PhaseTimings,
    PosteriorCovariance,
    RuntimePhase,
    RuntimeStatistics,
)


def _register(name, module):
    """Makes the extension submodule `module` importable as spacetime.<name>, recursing into its submodules.

    Args:
        name: The dotted name below the package, e.g. "priors.wnoa".
        module: The pybind11 submodule of _spacetime registered under that name.
    """
    sys.modules[f"{__name__}.{name}"] = module
    for attribute in dir(module):
        child = getattr(module, attribute)
        if type(child) is type(module):
            _register(f"{name}.{attribute}", child)


types = _spacetime.types
priors = _spacetime.priors
joints = _spacetime.joints
sensors = _spacetime.sensors
interface = _spacetime.interface
utilities = _spacetime.utilities
_submodules = ["types", "priors", "joints", "sensors", "interface", "utilities"]
if hasattr(_spacetime, "visualizer"):
    visualizer = _spacetime.visualizer
    _submodules.append("visualizer")
for _name in _submodules:
    _register(_name, getattr(_spacetime, _name))
