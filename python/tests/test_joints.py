"""Joints between robots and end effectors."""

import gtsam
import numpy as np
import pytest

from spacetime import joints, sensors
from spacetime.priors import wnoa

from conftest import batch_estimator, make_wnoa_robot, run_session

LINK = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0))


def test_joint_factory_dispatches_on_prior_types():
    _, a = make_wnoa_robot()
    _, b = make_wnoa_robot(base=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.2, 0, 0)))
    joint = joints.Joint(a, 0.2, b, 0.0, np.eye(6) * 1e-6, joints.JointOptions())
    assert isinstance(joint, joints.RobotJoint)
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
    assert isinstance(joint, joints.EndEffectorJoint)
    assert len(ee.joints()) == 1

    estimator = batch_estimator()
    estimator.addModel(prior_a)
    estimator.addModel(ee)
    run_session(estimator, end_time=0.3)
    node = ee.query(0.3, 0.0, estimator.getCurrentValues())
    assert isinstance(node, joints.EndEffectorValues)
    assert node.pose().equals(tip, 1e-4)

    # A joint on an end effector built through the generic factory is the same class.
    assert isinstance(joints.Joint(prior_a, length, ee.endEffectorPrior(), 0.0, np.eye(6) * 1e-6), joints.EndEffectorJoint)


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
