"""Bindings tools/FindCovariances.py builds its hypergradient from: Prior.queryJacobians, and the
graph's factor types (InterpolationFactor, LockedFactor, BetweenFactorVector6) with their noise models."""

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime import sensors
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import POSE_COVARIANCE, STRAIGHT, batch_estimator, bent_tip, make_wnoa_robot, pose_measurement, run_session

LENGTH, NODES, PERIOD = 0.2, 3, 0.1


@pytest.fixture(scope="module")
def solved():
    """A solved WNOA batch problem, base locked, with tip pose measurements between the node rows."""
    state, prior = make_wnoa_robot(length=LENGTH, nodes=NODES)
    estimator = batch_estimator()
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(POSE_COVARIANCE, prior)
    tip = bent_tip(LENGTH, 2.0)
    for k in range(5):
        sensor.addMeasurement(pose_measurement(LENGTH, PERIOD * k + 0.05, tip, POSE_COVARIANCE))
    estimator.addSensor(sensor)
    run_session(estimator, end_time=0.5, period=PERIOD)
    return state, prior, estimator, estimator.getFactors(), estimator.getCurrentValues()


def factors_of_type(graph, cls):
    return [graph.at(i) for i in range(graph.size()) if isinstance(graph.at(i), cls)]


def test_query_jacobians_returns_the_query_mean_and_a_block_per_corner_key(solved):
    state, prior, _, _, values = solved
    node, jacobians = prior.queryJacobians(0.25, 0.05, values)  # interpolated in time and arclength
    assert isinstance(node, wnoa.Values)
    assert node.pose().equals(prior.query(0.25, 0.05, values).pose(), 1e-12)
    assert len(jacobians) == 12  # four corners x (pose, strain, velocity)
    for key, block in jacobians.items():
        assert isinstance(key, int)
        assert block.shape == (18, 6)


def test_query_jacobians_on_a_node_is_the_identity(solved):
    _, prior, _, _, values = solved
    node, jacobians = prior.queryJacobians(0.2, LENGTH / 2, values)
    assert node is not None
    stacked = np.hstack([jacobians[key] for key in sorted(jacobians, key=lambda k: gtsam.Symbol(k).chr())])
    assert stacked.shape == (18, 18)
    np.testing.assert_allclose(stacked @ stacked.T, np.eye(18), atol=1e-12)


def test_query_jacobians_outside_the_extent_is_none_and_empty():
    _, prior = make_wnoa_robot()
    node, jacobians = prior.queryJacobians(0.0, 0.1, gtsam.Values())
    assert node is None
    assert jacobians == {}


def test_every_grid_prior_has_query_jacobians():
    state = wnov.ContinuumRobotState(LENGTH, NODES)
    assert wnov.Prior(state, gtsam.Pose3(), STRAIGHT).queryJacobians(0.0, 0.0, gtsam.Values()) == (None, {})
    state = wnoa_quasi_static.ContinuumRobotState(LENGTH, NODES)
    prior = wnoa_quasi_static.Prior(state, wnoa_quasi_static.Values(gtsam.Pose3(), STRAIGHT), np.eye(12), np.eye(6))
    assert prior.queryJacobians(0.0, 0.0, gtsam.Values()) == (None, {})


def test_no_graph_factor_is_opaque(solved):
    *_, graph, _ = solved
    for i in range(graph.size()):
        assert type(graph.at(i)) is not gtsam.NonlinearFactor, f"factor {i} has no Python type"
    assert factors_of_type(graph, wnoa.InterpolationFactor)
    assert factors_of_type(graph, st.LockedFactor)
    assert factors_of_type(graph, st.BetweenFactorVector6)


def test_interpolation_factor_accessors(solved):
    *_, graph, values = solved
    factors = factors_of_type(graph, wnoa.InterpolationFactor)
    assert len(factors) == 5
    assert sorted(round(f.time(), 9) for f in factors) == [0.05, 0.15, 0.25, 0.35, 0.45]
    for factor in factors:
        assert isinstance(factor, gtsam.NoiseModelFactor)
        assert factor.arclength() == pytest.approx(LENGTH)
        assert factor.fixedNoiseModel() is False
        assert isinstance(factor.innerFactor(), sensors.PoseFactor)
        np.testing.assert_allclose(factor.noiseModel().covariance(), POSE_COVARIANCE, rtol=1e-12)
        assert factor.innerJacobian(values).shape == (6, 18)


def test_inner_jacobian_and_query_covariance_rebuild_the_whitening(solved):
    """R + G Pcheck G^T, Pcheck from queryCovariance, is the covariance linearize() whitens with."""
    _, prior, _, graph, values = solved
    for factor in factors_of_type(graph, wnoa.InterpolationFactor):
        _, pcheck, _ = prior.queryCovariance(factor.time(), factor.arclength(), values)
        G = factor.innerJacobian(values)
        sigma = POSE_COVARIANCE + G @ pcheck @ G.T
        assert np.linalg.norm(G @ pcheck @ G.T) > 0  # off the node rows: the interpolation adds covariance
        e = factor.unwhitenedError(values)
        _, b = factor.linearize(values).jacobian()
        assert b @ b == pytest.approx(e @ np.linalg.solve(sigma, e), rel=1e-8)


def test_locked_factor_exposes_its_inner_factor_and_constants(solved):
    *_, graph, values = solved
    locked = factors_of_type(graph, st.LockedFactor)
    for factor in locked:
        inner = factor.inner()
        assert isinstance(inner, gtsam.NonlinearFactor)
        constants = factor.constants()
        assert isinstance(constants, gtsam.Values) and constants.size() > 0
        assert set(factor.keys()) | set(constants.keys()) == set(inner.keys())
        assert not set(factor.keys()) & set(constants.keys())
    assert any(isinstance(f.inner(), gtsam.NoiseModelFactor) for f in locked)


def test_between_factor_vector6_noise_is_the_scaled_random_walk(solved):
    """Q3 and Q4 are the identity: the time random walk is dt I, the arclength one ds I."""
    *_, graph, _ = solved
    between = factors_of_type(graph, st.BetweenFactorVector6)
    scales = set()
    for factor in between:
        np.testing.assert_array_equal(factor.measured(), np.zeros(6))
        covariance = factor.noiseModel().covariance()
        scale = covariance[0, 0]
        np.testing.assert_allclose(covariance, scale * np.eye(6), atol=1e-15)
        scales.add(round(scale, 9))
    assert scales == {PERIOD, LENGTH / (NODES - 1)}
