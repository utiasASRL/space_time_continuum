"""Estimator options, solving, covariance and snapshots."""

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime import sensors

from conftest import POSE_COVARIANCE, batch_estimator, estimator_options, bent_tip, make_wnoa_robot, pose_measurement, run_session


def test_smoother_options_pass_through():
    options = estimator_options(0.25)
    options.smoother = st.EstimatorOptions.Smoother.Incremental
    isam2 = gtsam.ISAM2Params()
    isam2.relinearizeSkip = 3
    options.isam2_params = isam2
    assert options.isam2_params.relinearizeSkip == 3
    estimator = st.Estimator(options)
    assert estimator.options().smoother == st.EstimatorOptions.Smoother.Incremental
    assert estimator.options().isam2_params.relinearizeSkip == 3
    assert estimator.smootherLag() == pytest.approx(0.25)
    assert isinstance(estimator.smoother(), gtsam.IncrementalFixedLagSmoother)


def test_quasi_static_estimates_only_at_measurement_times():
    state, prior = make_wnoa_robot()
    options = estimator_options(st.Estimator.BATCH_LAG)
    options.quasi_static = True
    estimator = st.Estimator(options)
    estimator.addPrior(prior)
    sensor = sensors.PoseSensor(prior)
    estimator.addSensor(sensor)
    for time in (0.013, 0.05):
        sensor.addMeasurement(pose_measurement(0.2, time, bent_tip(0.2, 0.0), POSE_COVARIANCE))
    estimator.estimate(0.1)
    times = sorted({node.time for node in state.snapshotEstimationNodes()})
    assert times == pytest.approx([0.013, 0.05])


def solve_bent_robot(compute_covariance):
    length, curvature = 0.2, 3.0
    state, prior = make_wnoa_robot(length=length)
    estimator = batch_estimator(compute_covariance)
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    for k in range(6):
        sensor.addMeasurement(pose_measurement(length, 0.1 * k, bent_tip(length, curvature), POSE_COVARIANCE))
    estimator.addSensor(sensor)
    run_session(estimator, end_time=0.5)
    return estimator, state, prior, bent_tip(length, curvature)


def test_batch_solve_reaches_measurement():
    estimator, state, prior, tip = solve_bent_robot(False)
    assert estimator.isBatch()
    assert estimator.getLastEstimateTime() > 0.0  # wall-clock seconds of the last estimate
    assert estimator.getSnapshot().time == pytest.approx(0.5)
    node = prior.query(0.5, 0.2, estimator.getCurrentValues())
    assert node.pose().equals(tip, 1e-2)
    timings = estimator.getBatchTimings()
    assert timings.solve > 0.0
    result = estimator.getBatchResult()
    assert result.iterations >= 1
    assert result.final_error < result.initial_error


def test_smoother_state_after_a_batch_solve():
    estimator, state, prior, _ = solve_bent_robot(False)
    nodes = state.snapshotEstimationNodes()
    timestamps = estimator.timestamps()
    assert len(timestamps) == estimator.getLinearizationPoint().size()  # every variable; the locked bases are constants
    assert len(timestamps) < 3 * len(nodes)
    assert estimator.getCurrentTimestamp() == pytest.approx(0.5)
    assert estimator.smootherLag() == float("inf")
    assert isinstance(estimator.getFactors(), gtsam.NonlinearFactorGraph)
    assert estimator.getFactors().size() > 0
    key = nodes[-1].poseKey()
    assert estimator.getLinearizationPoint().atPose3(key).equals(estimator.getCurrentValues().atPose3(key), 1e-12)


def sliding_window_tip(smoother, lag=0.25, end_time=0.5):
    """The tip pose of a bent robot at end_time, estimated in a `lag` window by `smoother`."""
    length, curvature = 0.2, 3.0
    state, prior = make_wnoa_robot(length=length)
    options = estimator_options(lag)
    options.smoother = smoother
    isam2 = gtsam.ISAM2Params()
    isam2.setRelinearizeThreshold(0.0)
    isam2.relinearizeSkip = 1
    options.isam2_params = isam2
    estimator = st.Estimator(options)
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    for k in range(6):
        sensor.addMeasurement(pose_measurement(length, 0.1 * k, bent_tip(length, curvature), POSE_COVARIANCE))
    estimator.addSensor(sensor)
    run_session(estimator, end_time=end_time)
    return estimator, prior.query(end_time, length, estimator.getCurrentValues()).pose()


def test_incremental_smoother_matches_the_batch_smoother():
    incremental, tip = sliding_window_tip(st.EstimatorOptions.Smoother.Incremental)
    _, reference = sliding_window_tip(st.EstimatorOptions.Smoother.Batch)
    assert not incremental.isBatch()
    assert tip.equals(reference, 1e-4)
    # Only the window is left.
    cutoff = incremental.getCurrentTimestamp() - incremental.smootherLag()
    assert all(time >= cutoff for time in incremental.timestamps().values())


def test_covariance_access():
    estimator, state, prior, _ = solve_bent_robot(True)
    nodes = state.snapshotEstimationNodes()
    key = nodes[-1].poseKey()
    marginal = estimator.covariance(key)
    assert marginal.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(marginal) > 0)
    posterior = estimator.getCurrentCovariance()
    assert posterior.has(key)
    np.testing.assert_allclose(posterior.covariance(key, key), marginal)
    ok, joint = posterior.joint([key, nodes[-2].poseKey()])
    assert ok and joint.shape == (12, 12)


def test_batch_runtime_statistics_and_snapshot_flags():
    # Batch: the graph is built once per estimate() (6 ticks over [0, 0.5]) but solved once by finalize().
    estimator, _, _, _ = solve_bent_robot(True)
    stats = estimator.getRuntimeStatistics()
    assert stats.count(st.RuntimePhase.GraphConstruction) == 6
    assert stats.count(st.RuntimePhase.Solve) == 1
    assert stats.count(st.RuntimePhase.Covariance) == 1
    assert stats.last().solve == pytest.approx(estimator.getBatchTimings().solve)

    snapshot = estimator.getSnapshot()
    assert snapshot.batch is True
    assert snapshot.covariance_enabled is True
    assert snapshot.runtime.count(st.RuntimePhase.Solve) == 1


def test_sliding_window_snapshot_is_not_batch():
    state, prior = make_wnoa_robot()
    estimator = st.Estimator(estimator_options(0.25))
    estimator.addModel(prior)
    run_session(estimator, end_time=0.3)
    snapshot = estimator.getSnapshot()
    assert snapshot.batch is False
    assert snapshot.covariance_enabled is False
    # Sliding window: every tick is solved, and the snapshot carries the statistics at its publication.
    solves = estimator.getRuntimeStatistics().count(st.RuntimePhase.Solve)
    assert solves > 0
    assert snapshot.runtime.count(st.RuntimePhase.Solve) == solves
    assert snapshot.runtime.count(st.RuntimePhase.GraphConstruction) == solves


def test_add_model_none_raises():
    with pytest.raises((TypeError, ValueError)):
        batch_estimator().addModel(None)
