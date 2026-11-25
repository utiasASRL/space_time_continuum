"""Shared fixtures for the spacetime Python binding tests."""

import pathlib

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime.priors import wnoa

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STRAIGHT = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # [rot, trans] strain of a straight backbone


IDENTITY_COVARIANCES = (np.eye(18), np.eye(6), np.eye(6), np.eye(6), np.eye(6))  # the WNOA weights the tests are written for


def make_wnoa_robot(length=0.2, nodes=3, base=None, strain=STRAIGHT, lock_base=True, options=None, covariances=IDENTITY_COVARIANCES):
    """A WNOA robot of `length` with `nodes` nodes: its grid state, (optionally) locked at the base, and its prior.

    `covariances` are the prior's (Q0, Q1, Q2, Q3, Q4): identity by default, or None for wnoa.DefaultCovariances().
    """
    base = gtsam.Pose3() if base is None else base
    state = wnoa.ContinuumRobotState(length, nodes)
    reference = wnoa.NodeValues()
    reference.pose(base)
    reference.epsilon(strain)
    options = wnoa.PriorOptions() if options is None else options
    prior = wnoa.Prior(state, reference, options) if covariances is None else wnoa.Prior(state, reference, *covariances, options)
    if lock_base:
        state.lock("pose", 0.0, base)
    return state, prior


def metis_lm_params():
    """GTSAM's Levenberg-Marquardt parameters, ordered with METIS."""
    params = gtsam.LevenbergMarquardtParams()
    params.setOrderingType("METIS")
    return params


def estimator_options(smoother_lag, lm_params=None):
    """Estimator options with every field set explicitly, so tests do not depend on EstimatorOptions' defaults.

    `smoother_lag` is the fixed-lag window [s], or st.Estimator.BATCH_LAG for batch mode. Mirrors
    tests/TestEstimatorOptions.hpp: when a field is added to EstimatorOptions, set it in both.
    """
    options = st.EstimatorOptions()
    options.smootherLag = smoother_lag
    options.smoother = st.EstimatorOptions.Smoother.Batch
    options.lm_params = metis_lm_params() if lm_params is None else lm_params
    options.isam2_params = gtsam.ISAM2Params()
    options.compute_covariance = False
    options.quasi_static = False
    return options


def batch_estimator(compute_covariance=False):
    options = estimator_options(st.Estimator.BATCH_LAG)
    options.compute_covariance = compute_covariance
    return st.Estimator(options)


def run_session(estimator, end_time, period=0.1, start_time=0.0):
    """Runs `estimator` offline over [start_time, end_time] through the default (standalone) session."""
    options = st.SessionOptions()
    options.start_time = start_time
    options.end_time = end_time
    options.period = period
    options.real_time = False
    session = st.Session(options)
    session.start(estimator)
    session.join()
    return session


def pose_measurement(s, time, pose, covariance):
    return st.types.MeasurementBase(s, time, pose.matrix(), covariance, np.ones((6, 1), dtype=np.int32))


POSE_COVARIANCE = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])


def bent_tip(length, curvature):
    """Tip pose of a constant-curvature robot from the origin, bending about body y."""
    return gtsam.Pose3.Expmap(length * np.array([0.0, curvature, 0.0, 1.0, 0.0, 0.0]))
