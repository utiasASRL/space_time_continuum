"""Joints between robots and end effectors."""

import gtsam
import numpy as np
import pytest

from spacetime import joints, sensors
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import STRAIGHT, batch_estimator, make_wnoa_robot, run_session

LINK = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0))


def test_joint_factory_dispatches_on_prior_types():
    _, a = make_wnoa_robot()
    _, b = make_wnoa_robot(base=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.2, 0, 0)))
    joint = joints.Joint(a, 0.2, b, 0.0, np.eye(6) * 1e-6, joints.JointOptions())
    assert isinstance(joint, joints.Joint)
    assert joint.priorA() is a and joint.priorB() is b
    assert joint.arclengthA() == pytest.approx(0.2)
    assert joint.arclengthB() == 0.0


def test_joint_invalid_inputs_raise():
    _, a = make_wnoa_robot()
    _, b = make_wnoa_robot()
    with pytest.raises(ValueError):
        joints.Joint(a, 0.2, a, 0.0, np.eye(6) * 1e-6)  # same robot
    with pytest.raises(ValueError):
        joints.Joint(a, 0.2, b, 0.0, np.eye(3) * 1e-6)  # wrong covariance size for a rigid joint
    with pytest.raises(TypeError):
        joints.Joint(a, 0.2, None, 0.0, np.eye(6) * 1e-6)  # no prior B


def test_end_effector_attach_and_estimate():
    length = 0.2
    _, prior_a = make_wnoa_robot(length=length)
    tip = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(length, 0, 0))
    ee = joints.EndEffector(tip)
    assert isinstance(ee.endEffectorPrior(), joints.EndEffectorPrior)
    assert isinstance(ee.state(), joints.EndEffectorState)
    joint = ee.attach(prior_a, length, np.eye(6) * 1e-6, joints.JointOptions())
    assert isinstance(joint, joints.Joint)
    assert len(ee.joints()) == 1

    estimator = batch_estimator()
    estimator.addModel(prior_a)
    estimator.addModel(ee)
    run_session(estimator, end_time=0.3)
    node = ee.query(0.3, 0.0, estimator.getCurrentValues())
    assert isinstance(node, joints.EndEffectorNodeValues)
    assert node.pose().equals(tip, 1e-4)

    # A joint on an end effector built through the generic factory is the same class.
    assert isinstance(joints.Joint(prior_a, length, ee.endEffectorPrior(), 0.0, np.eye(6) * 1e-6), joints.Joint)


def test_end_effector_without_robot_raises_on_estimate():
    ee = joints.EndEffector(gtsam.Pose3())
    estimator = batch_estimator()
    estimator.addModel(ee)
    with pytest.raises(RuntimeError):
        run_session(estimator, end_time=0.2)


def test_end_effector_pose_sensor():
    ee = joints.EndEffector(gtsam.Pose3())
    sensor = sensors.PoseSensor(ee.endEffectorPrior())
    assert isinstance(sensor, sensors.PoseSensorEndEffector)


def make_robot(kind, length=0.2, x=0.0):
    """A robot of `kind` ("wnoa", "wnov" or "quasi_static") with 3 nodes, its base at x along world x and locked there."""
    base = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(x, 0, 0))
    if kind == "wnoa":
        return make_wnoa_robot(length=length, base=base)[1]
    if kind == "wnov":
        state = wnov.ContinuumRobotState(length, 3)
        prior = wnov.Prior(state, base, STRAIGHT)
    else:
        state = wnoa_quasi_static.ContinuumRobotState(length, 3)
        prior = wnoa_quasi_static.Prior(state, wnoa_quasi_static.NodeValues(base, STRAIGHT))
    state.lock("pose", 0.0, base)
    return prior


ROBOT_KINDS = ["wnoa", "wnov", "quasi_static"]
PRIOR_CLASSES = {"wnoa": wnoa.Prior, "wnov": wnov.Prior, "quasi_static": wnoa_quasi_static.Prior}


@pytest.mark.parametrize("kind_b", ROBOT_KINDS)
@pytest.mark.parametrize("kind_a", ROBOT_KINDS)
def test_one_joint_class_for_every_prior_pair(kind_a, kind_b):
    a, b = make_robot(kind_a), make_robot(kind_b, x=0.3)
    joint = joints.Joint(a, 0.2, b, 0.0, np.eye(6) * 1e-6)
    assert type(joint) is joints.Joint
    assert type(joint.priorA()) is PRIOR_CLASSES[kind_a] and joint.priorA() is a
    assert type(joint.priorB()) is PRIOR_CLASSES[kind_b] and joint.priorB() is b
    assert joint.joins_velocity == (kind_a == "wnoa" and kind_b == "wnoa")


@pytest.mark.parametrize("kind", ROBOT_KINDS)
def test_end_effector_attaches_every_robot_prior(kind):
    ee = joints.EndEffector(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.2, 0, 0)))
    robot = make_robot(kind)
    joint = ee.attach(robot, 0.2, np.eye(6) * 1e-6)
    assert type(joint) is joints.Joint
    assert joint.priorA() is robot and joint.priorB() is ee.endEffectorPrior()
    assert type(ee.joints()[0]) is joints.Joint
    # A joint built on an end effector's prior directly is the same class.
    assert type(joints.Joint(robot, 0.2, ee.endEffectorPrior(), 0.0, np.eye(6) * 1e-6)) is joints.Joint


def test_unsupported_priors_raise_type_error():
    ee = joints.EndEffector(gtsam.Pose3())
    robot = make_robot("wnoa")
    with pytest.raises(TypeError):
        joints.Joint(ee.endEffectorPrior(), 0.0, robot, 0.0, np.eye(6) * 1e-6)  # an end effector is only ever side B
    with pytest.raises(TypeError):
        ee.attach(ee.endEffectorPrior(), 0.0, np.eye(6) * 1e-6)
    with pytest.raises(TypeError):
        joints.Joint(None, 0.0, robot, 0.0, np.eye(6) * 1e-6)


def test_end_effector_on_a_quasi_static_robot_estimates():
    length = 0.2
    robot = make_robot("quasi_static", length=length)
    tip = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(length, 0, 0))
    ee = joints.EndEffector(tip.retract(np.full(6, 0.01)))  # a perturbed guess: the robot positions it
    joint = ee.attach(robot, length, np.eye(6) * 1e-6)
    assert not joint.joins_velocity
    estimator = batch_estimator()
    estimator.addModel(robot)
    estimator.addModel(ee)
    run_session(estimator, end_time=0.3)
    assert ee.query(0.3, 0.0, estimator.getCurrentValues()).pose().equals(tip, 1e-3)


def test_subclass_extends_the_builtin_joint():
    length = 0.2
    a = make_robot("wnoa", length=length)
    b = make_robot("quasi_static", length=length, x=length)
    calls = []

    class CountingJoint(joints.Joint):
        def getNewFactors(self, time, new_values):
            factors = super().getNewFactors(time, new_values)  # the built-in joint's
            calls.append((time, len(factors)))
            return factors

    joint = CountingJoint(a, length, b, 0.0, np.eye(6) * 1e-6)
    assert isinstance(joint, joints.Joint)
    # Unoverridden methods forward to the built-in joint.
    assert joint.priorA() is a and joint.priorB() is b and joint.arclengthA() == pytest.approx(length)
    assert not joint.joins_velocity and not joint.hasOffset()
    estimator = batch_estimator()
    for model in (a, b, joint):
        estimator.addModel(model)
    run_session(estimator, end_time=0.2)
    assert calls and calls[0] == (pytest.approx(0.0), 1)  # one pose factor per row, from the C++ joint


def test_subclass_overrides_reach_cpp():
    from spacetime import visualizer
    a, b, other = make_robot("wnoa"), make_robot("wnoa", x=0.3), make_robot("wnoa", x=0.6)

    class Rerouted(joints.Joint):
        def stateB(self):
            return other.state()  # a robot the visualizer was not given

    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(a)
    vis.addPrior(b)
    vis.addJoint(joints.Joint(a, 0.2, b, 0.0, np.eye(6) * 1e-6))
    with pytest.raises(ValueError):  # the C++ check sees the Python stateB
        vis.addJoint(Rerouted(a, 0.2, b, 0.0, np.eye(6) * 1e-6))


def test_joint_defined_wholly_in_python():
    """Holds A's tip node and B's base node together with a gtsam.BetweenFactorPose3, B's base free."""
    length = 0.2
    a = make_robot("wnoa", length=length)
    state_b = wnoa.ContinuumRobotState(length, 3)
    reference = wnoa.NodeValues()
    reference.pose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.05, 0)))  # initial guess away from A's tip
    reference.epsilon(STRAIGHT)
    b = wnoa.Prior(state_b, reference)
    noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)

    class NodeJoint(joints.Joint):
        def __init__(self):
            super().__init__()
            self.joined = set()

        def priorA(self):
            return a

        def priorB(self):
            return b

        def arclengthA(self):
            return length

        def arclengthB(self):
            return 0.0

        def offset(self):
            return gtsam.Pose3()

        def options(self):
            return joints.JointOptions()

        def offsetCovariance(self):
            return np.eye(6) * 1e-8

        def getNewFactors(self, time, new_values):
            if time in self.joined:
                return []
            self.joined.add(time)
            node_a = a.state().getClosestShiftedNode(length, time, 0, 0)
            node_b = b.state().getClosestShiftedNode(0.0, time, 0, 0)
            return [gtsam.BetweenFactorPose3(node_a.poseKey(), node_b.poseKey(), gtsam.Pose3(), noise)]

    joint = NodeJoint()
    # The C++ defaults: states from the priors, no offset translation, no velocity coupling.
    assert joint.stateA() is a.state() and joint.stateB() is b.state()
    assert not joint.hasOffset() and not joint.joins_velocity

    estimator = batch_estimator()
    for model in (a, b, joint):
        estimator.addModel(model)
    run_session(estimator, end_time=0.2)
    values = estimator.getCurrentValues()
    tip_a = a.query(0.2, length, values).pose()
    assert b.query(0.2, 0.0, values).pose().equals(tip_a, 1e-3)


def test_python_joint_without_its_accessors_raises():
    class Empty(joints.Joint):
        def __init__(self):
            super().__init__()

    with pytest.raises(RuntimeError, match="pure virtual"):
        Empty().arclengthA()
    with pytest.raises(TypeError):
        joints.Joint()  # only a subclass may skip the arguments
