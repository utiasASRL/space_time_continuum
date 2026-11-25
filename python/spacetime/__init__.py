"""Continuum robot state estimation in space and time, on GTSAM.

The Python interface to the spacetime C++ library. GTSAM objects (gtsam.Pose3, gtsam.Values,
factors, noise models, ...) are GTSAM's own Python types, passed to and from the library as is.

Running the estimator:
    import spacetime
    session = spacetime.Session(spacetime.SessionOptions())  # the default (standalone) interface
    session.start(estimator)
    session.join()

    spacetime.setInterface("ros2")  # "default", "ros2" or "mujoco" (not implemented yet)
    session = spacetime.Session(options, ros_options)       # now the ROS2 session

spacetime.Session always creates a session of the interface selected last with setInterface,
however it was imported (`import spacetime`, `from spacetime import Session` or
`from spacetime import *`). It is a function returning an instance of that interface's session
class (spacetime.interface.Session or spacetime.interface.ros2.Session); use those classes for
isinstance checks.

Submodules:
    types       Base classes, each subclassable in Python: Model, StatefulModel, SensorBase,
                ActuatorBase, State, NodeBase, NodeValuesBase; Robot (a list of models),
                MeasurementBase, KeyManager.
    priors.wnoa The white-noise-on-acceleration prior, its node values, nodes and grid states.
    priors.wnov The white-noise-on-velocity prior, its node values, nodes and grid states.
    priors.wnoa_quasi_static
                The quasi-static prior (WNOA in arclength only, for EstimatorOptions.quasi_static).
    joints      Joints between robots, and end effectors.
    sensors     Built-in sensors (pose, FBG, gyroscope, accelerometer, IMU), their options,
                bias models and factors.
    actuators   Built-in actuators (KinematicTendon: a tendon whose measured length is compared
                with the path through its routing disks) and their factors.
    interface   The sessions behind spacetime.Session, SessionOptions, and the SnapshotHistory of
                the estimates.
    interface.ros2
                ROS2-specific helpers (if built with InterfaceROS2): init/shutdown rclcpp,
                SessionRosOptions, sensor nodes (pose from TF, gyroscope / accelerometer / IMU
                from sensor_msgs/Imu, tendon length from sensor_msgs/JointState), waitForTransform.
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
actuators = _spacetime.actuators
interface = _spacetime.interface
utilities = _spacetime.utilities
_submodules = ["types", "priors", "joints", "sensors", "actuators", "interface", "utilities"]
if hasattr(_spacetime, "visualizer"):
    visualizer = _spacetime.visualizer
    _submodules.append("visualizer")
for _name in _submodules:
    _register(_name, getattr(_spacetime, _name))

SessionOptions = interface.SessionOptions

_INTERFACES = ("default", "ros2", "mujoco")
_interface = "default"


def _sessionClass(name):
    """The Session class of interface `name`, or raises if it is unknown, unimplemented or not built."""
    if name == "default":
        return interface.Session
    if name == "ros2":
        if not hasattr(interface, "ros2"):
            raise ImportError("spacetime was built without the ROS2 interface (InterfaceROS2=OFF).")
        return interface.ros2.Session
    if name == "mujoco":
        raise NotImplementedError("The MuJoCo interface is not implemented yet.")
    raise ValueError(f"Unknown interface {name!r}: expected one of {', '.join(map(repr, _INTERFACES))}.")


def setInterface(name):
    """Selects the interface spacetime.Session runs the estimator on.

    Args:
        name: "default" (the standalone interface: replays recorded data), "ros2" (live or
            rosbag estimation on ROS2 nodes) or "mujoco" (not implemented yet).

    Raises:
        ValueError: for an unknown name.
        NotImplementedError: for "mujoco".
        ImportError: for an interface spacetime was built without.
    """
    global _interface
    _sessionClass(name)
    _interface = name


def getInterface():
    """The name of the interface spacetime.Session currently runs on (see setInterface)."""
    return _interface


def Session(*args, **kwargs):
    """Creates a session of the interface selected with setInterface ("default" unless changed).

    Takes the arguments of that interface's session class: (options) for the default
    interface (spacetime.interface.Session), and (options, ros_options) for ROS2
    (spacetime.interface.ros2.Session). It looks the interface up on every call, so it follows
    setInterface however it was imported.

    Returns:
        An instance of the selected interface's session class.
    """
    return _sessionClass(_interface)(*args, **kwargs)
