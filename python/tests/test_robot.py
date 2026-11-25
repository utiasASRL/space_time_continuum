"""Grid states (construction, locks, node access) and robots (lists of models)."""

import gtsam
import numpy as np
import pytest

from spacetime import sensors, types
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import POSE_COVARIANCE, STRAIGHT, batch_estimator, make_wnoa_robot


def test_construction_by_length_and_by_arclengths():
    state = wnoa.ContinuumRobotState(0.3, 4)
    np.testing.assert_allclose(state.arclengths, [0.0, 0.1, 0.2, 0.3])
    assert state.length == pytest.approx(0.3)
    assert state.spatial_dim == 4
    state = wnoa.ContinuumRobotState([0.0, 0.05, 0.3])
    np.testing.assert_allclose(state.arclengths, [0.0, 0.05, 0.3])
    assert isinstance(state, types.State)


def test_lock_each_variable_type():
    state = wnoa.ContinuumRobotState(0.2, 3)
    state.lock("pose", 0.0, gtsam.Pose3())
    state.lock("strain", -1.0, STRAIGHT)
    state.lock("velocity", 0.1, np.zeros(6), 0.0)


def test_lock_wrong_type_raises():
    state = wnoa.ContinuumRobotState(0.2, 3)
    with pytest.raises(ValueError):
        state.lock("pose", 0.0, STRAIGHT)
    with pytest.raises(ValueError):
        state.lock("strain", 0.0, gtsam.Pose3())


def test_lock_unknown_variable_or_arclength_raises():
    state = wnoa.ContinuumRobotState(0.2, 3)
    with pytest.raises(ValueError):
        state.lock("acceleration", 0.0, np.zeros(6))
    with pytest.raises(ValueError):
        state.lock("pose", 0.05, gtsam.Pose3())


def test_lock_without_a_value_holds_the_initial_guess():
    state, prior = make_wnoa_robot(length=1.0, nodes=5, lock_base=False)
    state.lock("pose", 0.0)
    state.lock("strain", 1.0)
    sensor = sensors.PoseSensor(prior)
    tip = gtsam.Pose3(gtsam.Rot3.Rz(0.3), gtsam.Point3(0.9, 0.3, 0.0))
    sensor.addMeasurement(types.MeasurementBase(arclength=1.0, time=0.0, value=tip.matrix(), covariance=POSE_COVARIANCE))
    estimator = batch_estimator()
    estimator.addPrior(prior)
    estimator.addSensor(sensor)
    estimator.estimate(0.0)
    estimator.finalize()
    nodes = state.snapshotEstimationNodes()
    values = estimator.getCurrentValues()
    assert nodes[0].getValues(values).pose().equals(gtsam.Pose3(), 1e-12)
    np.testing.assert_allclose(nodes[-1].getValues(values).epsilon(), STRAIGHT, atol=1e-12)
    assert not estimator.getLinearizationPoint().exists(nodes[-1].strainKey())


def test_lock_without_a_value_rejects_invalid_locks():
    state = wnoa.ContinuumRobotState(0.2, 3)
    with pytest.raises(ValueError):
        state.lock("strain", 0.05)
    with pytest.raises(ValueError):
        state.lock("acceleration", 0.0, t=0.0)


def test_wnov_state_locks_pose_only():
    state = wnov.ContinuumRobotState(0.2, 3)
    state.lock("pose", 0.0, gtsam.Pose3())
    with pytest.raises(ValueError):
        state.lock("strain", 0.0, STRAIGHT)


def test_quasi_static_state_locks_pose_and_strain_only():
    state = wnoa_quasi_static.ContinuumRobotState(0.2, 3)
    state.lock("pose", 0.0, gtsam.Pose3())
    state.lock("strain", 0.0, STRAIGHT)
    with pytest.raises(ValueError):
        state.lock("velocity", 0.0, np.zeros(6))


def test_prior_extends_its_state():
    state, prior = make_wnoa_robot()
    assert prior.state() is state
    assert isinstance(prior, types.StatefulModel)


def test_robot_lists_its_models_for_the_estimator():
    state, prior = make_wnoa_robot()
    sensor = sensors.PoseSensor(prior)
    robot = types.Robot()
    robot.addModel(prior)
    robot.addModel(sensor)
    assert list(robot.models()) == [prior, sensor]
    estimator = batch_estimator()
    estimator.addRobot(robot)
    assert len(estimator.getMeasurementQueues()) == 1
    with pytest.raises((TypeError, ValueError)):
        robot.addModel(None)


def test_nodes_after_extension():
    state, prior = make_wnoa_robot(nodes=3)
    km = types.KeyManager()
    new_values = gtsam.Values()
    prior.extendState(km, 0.0, gtsam.Values(), new_values)
    nodes = state.snapshotEstimationNodes()
    assert len(nodes) == 3
    assert isinstance(nodes[0], wnoa.Node)
    assert new_values.exists(nodes[1].poseKey())
    assert list(state.times()) == [0.0]
    nearest = state.getNearestNode(0.09, 0.0)
    assert nearest.arclength == pytest.approx(0.1)
    assert state.getNextSpatialNode(nodes[0]).arclength == pytest.approx(0.1)
