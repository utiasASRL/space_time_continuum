"""WNOA, WNOV and quasi-static priors: values, options, construction and queries."""

import gtsam
import numpy as np
import pytest

from spacetime import sensors
import spacetime as st
from spacetime import types
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import (POSE_COVARIANCE, STRAIGHT, batch_estimator, bent_tip, make_wnoa_robot, pose_measurement,
                      run_session)


def test_wnoa_values_getters_and_setters():
    values = wnoa.Values()
    pose = gtsam.Pose3(gtsam.Rot3.Rx(0.2), gtsam.Point3(0.1, 0.0, 0.3))
    values.pose(pose)
    values.epsilon(STRAIGHT)
    values.varpi(np.arange(6.0))
    assert values.pose().equals(pose, 1e-12)
    np.testing.assert_array_equal(values.epsilon(), STRAIGHT)
    np.testing.assert_array_equal(values.varpi(), np.arange(6.0))
    assert wnoa.Values.dim == 18


def test_wnoa_values_wrong_size_raises():
    with pytest.raises(TypeError):
        wnoa.Values().epsilon(np.zeros(5))


def test_wnov_values():
    pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
    values = wnov.Values(pose)
    assert values.pose().equals(pose, 1e-12)
    assert wnov.Values.dim == 6


@pytest.mark.parametrize("factor_type", [wnoa.FactorType.WNOA, wnoa.FactorType.Legacy, wnoa.FactorType.Experimental])
def test_each_factor_type_recovers_measured_tip(factor_type):
    length, curvature = 0.2, 2.0
    state = wnoa.ContinuumRobotState(length, 3)
    reference = wnoa.Values()
    reference.epsilon(STRAIGHT)
    options = wnoa.PriorOptions()
    options.factor_type = factor_type
    prior = wnoa.Prior(state, reference, options=options)
    state.lock("pose", 0.0, gtsam.Pose3())
    assert prior.options().factor_type == factor_type
    estimator = batch_estimator()
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    tip = bent_tip(length, curvature)
    for k in range(6):
        sensor.addMeasurement(pose_measurement(length, 0.1 * k, tip, POSE_COVARIANCE))
    estimator.addSensor(sensor)
    run_session(estimator, end_time=0.5)
    assert prior.query(0.5, length, estimator.getCurrentValues()).pose().equals(tip, 1e-2)


def test_wnoa_prior_with_all_arguments():
    state = wnoa.ContinuumRobotState(0.2, 3)
    reference = wnoa.Values()
    reference.epsilon(STRAIGHT)
    options = wnoa.PriorOptions()
    options.remove_unary = True
    prior = wnoa.Prior(state, reference, np.eye(18), np.eye(6), np.eye(6) * 2, np.eye(6) * 3, np.eye(6) * 4, options)
    assert prior.options().remove_unary is True
    assert prior.state() is state


def test_wnoa_prior_wrong_q_shape_raises():
    state = wnoa.ContinuumRobotState(0.2, 3)
    with pytest.raises(TypeError):
        wnoa.Prior(state, wnoa.Values(), np.eye(6))


def test_prior_on_wrong_state_type_raises():
    state = wnov.ContinuumRobotState(0.2, 3)
    with pytest.raises(ValueError):
        wnoa.Prior(state, wnoa.Values())


def test_wnov_prior_construction():
    state = wnov.ContinuumRobotState(0.2, 3)
    options = wnov.PriorOptions()
    options.remove_unary = True
    prior = wnov.Prior(state, gtsam.Pose3(), STRAIGHT, np.eye(6), np.eye(6), np.eye(6), options)
    assert prior.state() is state
    assert prior.options().remove_unary is True


def test_query_on_solved_problem_recovers_measured_tip():
    length, curvature = 0.2, 2.0
    state, prior = make_wnoa_robot(length=length)
    estimator = batch_estimator(compute_covariance=True)
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    tip = bent_tip(length, curvature)
    for k in range(6):
        sensor.addMeasurement(pose_measurement(length, 0.1 * k, tip, POSE_COVARIANCE))
    estimator.addSensor(sensor)
    run_session(estimator, end_time=0.5)

    values = estimator.getCurrentValues()
    node = prior.query(0.5, length, values)
    assert isinstance(node, wnoa.Values)
    assert node.pose().equals(tip, 1e-2)

    node, conditional, posterior = prior.queryCovariance(0.25, 0.1, values, estimator.getCurrentCovariance())
    assert isinstance(node, wnoa.Values)
    assert conditional.shape == (18, 18)
    assert posterior.shape == (18, 18)


def test_query_outside_extent_is_none():
    state, prior = make_wnoa_robot()
    assert prior.query(0.0, 0.1, gtsam.Values()) is None


def quasi_static_prior(length=0.2, nodes=3):
    state = wnoa_quasi_static.ContinuumRobotState(length, nodes)
    state.lock("pose", 0.0, gtsam.Pose3())
    return state, wnoa_quasi_static.Prior(state, wnoa_quasi_static.Values(gtsam.Pose3(), STRAIGHT), np.eye(12), np.eye(6))


def test_quasi_static_values_and_options():
    pose = gtsam.Pose3(gtsam.Rot3.Ry(0.1), gtsam.Point3(0.0, 0.0, 0.1))
    values = wnoa_quasi_static.Values(pose, STRAIGHT)
    assert values.pose().equals(pose, 1e-12)
    np.testing.assert_array_equal(values.epsilon(), STRAIGHT)
    assert wnoa_quasi_static.Values.dim == 12
    options = wnoa_quasi_static.PriorOptions()
    assert options.fixed_noise_model is False
    assert options.cache_noise_model is True
    assert list(wnoa_quasi_static.InterpolatedKeys.keys()) == [gtsam.Symbol("q", 0).key(), gtsam.Symbol("q", 1).key()]


def test_quasi_static_prior_rows_are_independent_shapes():
    state, prior = quasi_static_prior()
    km, values = types.KeyManager(), gtsam.Values()
    first = prior.extendState(km, 0.0, values, values)
    second = prior.extendState(km, 0.1, values, values)
    assert len(first) == len(second) == 4  # pose and strain unary, two spatial factors, no time factor
    nodes = state.snapshotEstimationNodes()
    assert isinstance(nodes[0], wnoa_quasi_static.Node)
    first_row = {key for node in nodes[:3] for key in node.keys()}
    assert all(key not in first_row for factor in second for key in factor.keys())


def test_quasi_static_estimation_fits_each_measurement_time():
    length, curvature = 0.2, 2.0
    state, prior = quasi_static_prior(length)
    options = st.EstimatorOptions()
    options.smootherLag = st.Estimator.BATCH_LAG
    options.quasi_static = True
    estimator = st.Estimator(options)
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    assert isinstance(sensor, sensors.PoseSensorQuasiStatic)
    tips = {0.013: bent_tip(length, curvature), 0.07: bent_tip(length, -curvature)}
    for time, tip in tips.items():
        sensor.addMeasurement(pose_measurement(length, time, tip, POSE_COVARIANCE))
    estimator.addSensor(sensor)
    estimator.estimate(0.1)
    estimator.finalize()
    assert list(state.times()) == pytest.approx(list(tips))
    values = estimator.getCurrentValues()
    for time, tip in tips.items():
        node = prior.query(time, length, values)
        assert isinstance(node, wnoa_quasi_static.Values)
        assert node.pose().equals(tip, 1e-2)
    # Between rows, a query takes the nearer row.
    assert prior.query(0.03, length, values).pose().equals(prior.query(0.013, length, values).pose(), 1e-12)
