"""Shared fixtures for the spacetime Python binding tests."""

import pathlib

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime import interface
from spacetime.priors import wnoa

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STRAIGHT = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # [rot, trans] strain of a straight backbone


def make_wnoa_robot(length=0.2, nodes=3, base=None, strain=STRAIGHT, lock_base=True):
    """A WNOA robot of `length` with `nodes` nodes: its grid state, (optionally) locked at the base, and its prior."""
    base = gtsam.Pose3() if base is None else base
    state = wnoa.ContinuumRobotState(length, nodes)
    reference = wnoa.Values()
    reference.pose(base)
    reference.epsilon(strain)
    prior = wnoa.Prior(state, reference)
    if lock_base:
        state.lock("pose", 0.0, base)
    return state, prior


def batch_estimator(compute_covariance=False):
    options = st.EstimatorOptions()
    options.smootherLag = st.Estimator.BATCH_LAG
    options.compute_covariance = compute_covariance
    return st.Estimator(options)


def run_session(estimator, end_time, period=0.1, start_time=0.0):
    """Runs `estimator` offline over [start_time, end_time] through the standalone session."""
    options = interface.SessionOptions()
    options.start_time = start_time
    options.end_time = end_time
    options.period = period
    options.real_time = False
    session = interface.Session(estimator, options)
    session.addMeasurementQueues(estimator.getMeasurementQueues())
    session.start()
    session.join()
    return session


def pose_measurement(s, time, pose, covariance):
    return st.types.MeasurementBase(s, time, pose.matrix(), covariance, np.ones((6, 1), dtype=np.int32))


POSE_COVARIANCE = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])


def bent_tip(length, curvature):
    """Tip pose of a constant-curvature robot from the origin, bending about body y."""
    return gtsam.Pose3.Expmap(length * np.array([0.0, curvature, 0.0, 1.0, 0.0, 0.0]))
