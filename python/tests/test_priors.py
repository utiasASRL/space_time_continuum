"""WNOA, WNOV and quasi-static priors: values, options, construction and queries."""

import gtsam
import numpy as np
import pytest

from spacetime import sensors
import spacetime as st
from spacetime import types
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import (POSE_COVARIANCE, STRAIGHT, batch_estimator, bent_tip, estimator_options, make_wnoa_robot, pose_measurement,
                      run_session)


def test_wnoa_values_getters_and_setters():
    values = wnoa.NodeValues()
    pose = gtsam.Pose3(gtsam.Rot3.Rx(0.2), gtsam.Point3(0.1, 0.0, 0.3))
    values.pose(pose)
    values.epsilon(STRAIGHT)
    values.varpi(np.arange(6.0))
    assert values.pose().equals(pose, 1e-12)
    np.testing.assert_array_equal(values.epsilon(), STRAIGHT)
    np.testing.assert_array_equal(values.varpi(), np.arange(6.0))
    assert wnoa.NodeValues.dim == 18


def test_node_values_default_to_a_straight_strain():
    np.testing.assert_array_equal(wnoa.NodeValues().epsilon(), STRAIGHT)
    np.testing.assert_array_equal(wnoa.NodeValues().varpi(), np.zeros(6))
    np.testing.assert_array_equal(wnoa_quasi_static.NodeValues().epsilon(), STRAIGHT)


def test_default_covariances():
    Q0, Q1, Q2, Q3, Q4 = wnoa.DefaultCovariances()
    assert Q0.shape == (18, 18) and all(Q.shape == (6, 6) for Q in (Q1, Q2, Q3, Q4))
    # The paper's experiment tuning, times its prior_factor 0.5, with torsion (about x) tight.
    np.testing.assert_allclose(np.diag(Q0)[6:9], 0.5 * np.array([1e-2, 1e2, 1e2]))
    np.testing.assert_allclose(np.diag(Q1), 0.5 * np.array([3e2] * 3 + [3e0] * 3))
    for (a, b) in zip(wnoa.Prior.DefaultCovariances(), (Q0, Q1, Q2, Q3, Q4)):
        np.testing.assert_array_equal(a, b)
    qs0, qs2 = wnoa_quasi_static.DefaultCovariances()
    np.testing.assert_array_equal(qs0, Q0[:12, :12])
    np.testing.assert_array_equal(qs2, Q2)
    for Q in wnov.DefaultCovariances():
        np.testing.assert_array_equal(Q, np.eye(6))

    state = wnoa.ContinuumRobotState(1.0, 3)
    wnoa.Prior(state, wnoa.NodeValues(), *wnoa.DefaultCovariances())


def test_priors_warn_when_the_tangent_axis_is_not_x(capfd):
    z_tangent = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    values = wnoa.NodeValues()
    values.epsilon(z_tangent)
    wnoa.Prior(wnoa.ContinuumRobotState(1.0, 3), values)
    assert "tangent axis" in capfd.readouterr().out
    wnoa.Prior(wnoa.ContinuumRobotState(1.0, 3), wnoa.NodeValues())
    assert "tangent axis" not in capfd.readouterr().out
    wnov.Prior(wnov.ContinuumRobotState(1.0, 3), gtsam.Pose3(), z_tangent)
    assert "tangent axis" in capfd.readouterr().out


def test_wnoa_values_wrong_size_raises():
    with pytest.raises(TypeError):
        wnoa.NodeValues().epsilon(np.zeros(5))


def test_wnov_values():
    pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
    values = wnov.NodeValues(pose)
    assert values.pose().equals(pose, 1e-12)
    assert wnov.NodeValues.dim == 6


@pytest.mark.parametrize("factor_type", [wnoa.FactorType.WNOA, wnoa.FactorType.Legacy, wnoa.FactorType.Experimental])
def test_each_factor_type_recovers_measured_tip(factor_type):
    length, curvature = 0.2, 2.0
    state = wnoa.ContinuumRobotState(length, 3)
    reference = wnoa.NodeValues()
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
    reference = wnoa.NodeValues()
    reference.epsilon(STRAIGHT)
    options = wnoa.PriorOptions()
    options.remove_unary = True
    prior = wnoa.Prior(state, reference, np.eye(18), np.eye(6), np.eye(6) * 2, np.eye(6) * 3, np.eye(6) * 4, options)
    assert prior.options().remove_unary is True
    assert prior.state() is state


def test_priors_without_covariances_use_the_defaults():
    state = wnoa.ContinuumRobotState(1.0, 3)
    prior = wnoa.Prior(state, wnoa.NodeValues())
    explicit = wnoa.Prior(wnoa.ContinuumRobotState(1.0, 3), wnoa.NodeValues(), *wnoa.DefaultCovariances())
    options = wnoa.PriorOptions()
    options.remove_unary = True
    assert wnoa.Prior(wnoa.ContinuumRobotState(1.0, 3), wnoa.NodeValues(), options).options().remove_unary
    wnov.Prior(wnov.ContinuumRobotState(1.0, 3), gtsam.Pose3(), STRAIGHT)
    wnoa_quasi_static.Prior(wnoa_quasi_static.ContinuumRobotState(1.0, 3), wnoa_quasi_static.NodeValues())
    # The same prior factors as with the explicit defaults.
    a, b = batch_estimator(), batch_estimator()
    a.addPrior(prior)
    b.addPrior(explicit)
    for estimator in (a, b):
        estimator.estimate(0.0)
        estimator.estimate(0.1)
        estimator.finalize()
    assert a.getFactors().size() == b.getFactors().size() > 0
    values = b.getCurrentValues()
    assert a.getFactors().error(values) == pytest.approx(b.getFactors().error(values))


def test_partial_covariances_raise():
    state = wnoa.ContinuumRobotState(0.2, 3)
    with pytest.raises(TypeError):
        wnoa.Prior(state, wnoa.NodeValues(), np.eye(18), np.eye(6))  # Q2..Q4 missing


def test_wnoa_prior_wrong_q_shape_raises():
    state = wnoa.ContinuumRobotState(0.2, 3)
    with pytest.raises(TypeError):
        wnoa.Prior(state, wnoa.NodeValues(), np.eye(6))


def test_prior_on_wrong_state_type_raises():
    state = wnov.ContinuumRobotState(0.2, 3)
    with pytest.raises(ValueError):
        wnoa.Prior(state, wnoa.NodeValues())


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
    assert isinstance(node, wnoa.NodeValues)
    assert node.pose().equals(tip, 1e-2)

    node, conditional, posterior = prior.queryCovariance(0.25, 0.1, values, estimator.getCurrentCovariance())
    assert isinstance(node, wnoa.NodeValues)
    assert conditional.shape == (18, 18)
    assert posterior.shape == (18, 18)


def test_query_outside_extent_is_none():
    state, prior = make_wnoa_robot()
    assert prior.query(0.0, 0.1, gtsam.Values()) is None


def quasi_static_prior(length=0.2, nodes=3):
    state = wnoa_quasi_static.ContinuumRobotState(length, nodes)
    state.lock("pose", 0.0, gtsam.Pose3())
    return state, wnoa_quasi_static.Prior(state, wnoa_quasi_static.NodeValues(gtsam.Pose3(), STRAIGHT), np.eye(12), np.eye(6))


def test_quasi_static_values_and_options():
    pose = gtsam.Pose3(gtsam.Rot3.Ry(0.1), gtsam.Point3(0.0, 0.0, 0.1))
    values = wnoa_quasi_static.NodeValues(pose, STRAIGHT)
    assert values.pose().equals(pose, 1e-12)
    np.testing.assert_array_equal(values.epsilon(), STRAIGHT)
    assert wnoa_quasi_static.NodeValues.dim == 12
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
    options = estimator_options(st.Estimator.BATCH_LAG)
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
        assert isinstance(node, wnoa_quasi_static.NodeValues)
        assert node.pose().equals(tip, 1e-2)
    # Between rows, a query takes the nearer row.
    assert prior.query(0.03, length, values).pose().equals(prior.query(0.013, length, values).pose(), 1e-12)


# ---- Per-arclength covariance overrides --------------------------------------------------------

def test_wnoa_covariance_overrides_resolve_per_node_and_segment():
    Q0, Q1, Q2, Q3, Q4 = wnoa.DefaultCovariances()
    overrides = wnoa.CovarianceOverrides()
    # Chaining, a profile function, and the per-Q builders.
    overrides.overrideQ1(0.0, 10 * Q1).overrideQ2(lambda s, nominal: (1 + s) * nominal)
    overrides.Q4.scale(-1, 0.5)
    overrides.Q3.set(0.1, 2 * Q3).scale(0.1, 3.0)
    assert not overrides.empty() and wnoa.CovarianceOverrides().empty()

    prior = wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues(), Q0, Q1, Q2, Q3, Q4, overrides=overrides)
    profiles = prior.covarianceProfiles()
    assert profiles.Q1.axis() == st.priors.CovarianceAxis.Time and profiles.Q2.axis() == st.priors.CovarianceAxis.Space
    assert profiles.Q1.size() == 3 and profiles.Q2.size() == 2
    np.testing.assert_array_equal(profiles.Q1.at(0), 10 * Q1)
    np.testing.assert_array_equal(profiles.Q1.at(1), Q1)
    np.testing.assert_array_equal(profiles.Q2.at(1), 1.1 * Q2)  # the segment [0.1, 0.2], keyed by 0.1
    np.testing.assert_array_equal(profiles.Q3.at(1), 6 * Q3)
    np.testing.assert_array_equal(profiles.Q4.at(1), 0.5 * Q4)  # -1: the last segment
    np.testing.assert_array_equal(profiles.Q2.atArclength(0.15), profiles.Q2.at(1))
    np.testing.assert_array_equal(profiles.Q2.atArclength(None), Q2)
    assert prior.covarianceProfiles().Q3.isConstant() is False


def test_covariance_overrides_default_to_constant_and_reach_every_constructor():
    prior = wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues())
    assert all(getattr(prior.covarianceProfiles(), q).isConstant() for q in ("Q1", "Q2", "Q3", "Q4"))

    Q1 = np.diag([1.0, 2, 3, 4, 5, 6])
    defaulted = wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues(), overrides=wnoa.CovarianceOverrides().overrideQ1(0.2, Q1))
    np.testing.assert_array_equal(defaulted.covarianceProfiles().Q1.at(2), Q1)

    wnov_prior = wnov.Prior(wnov.ContinuumRobotState(0.2, 3), gtsam.Pose3(), STRAIGHT,
                            overrides=wnov.CovarianceOverrides().overrideQ1(0.0, Q1).overrideQ2(-1, 2 * Q1))
    np.testing.assert_array_equal(wnov_prior.covarianceProfiles().Q1.at(0), Q1)
    np.testing.assert_array_equal(wnov_prior.covarianceProfiles().Q2.at(2), 2 * Q1)

    qs_prior = wnoa_quasi_static.Prior(wnoa_quasi_static.ContinuumRobotState(0.2, 3), wnoa_quasi_static.NodeValues(),
                                       overrides=wnoa_quasi_static.CovarianceOverrides().overrideQ2(0.1, Q1))
    np.testing.assert_array_equal(qs_prior.covarianceProfiles().Q2.at(1), Q1)


def test_covariance_override_on_a_non_node_arclength_raises():
    with pytest.raises(ValueError):
        wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues(), overrides=wnoa.CovarianceOverrides().overrideQ1(0.05, np.eye(6)))
    with pytest.raises(ValueError):  # the tip starts no segment
        wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues(), overrides=wnoa.CovarianceOverrides().overrideQ2(0.2, np.eye(6)))


def test_covariance_overrides_change_the_prior_factors():
    Q0, Q1, Q2, Q3, Q4 = wnoa.DefaultCovariances()

    def solved(overrides):
        estimator = batch_estimator()
        estimator.addPrior(wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues(), Q0, Q1, Q2, Q3, Q4, overrides=overrides))
        estimator.estimate(0.0)
        estimator.estimate(0.1)
        estimator.finalize()
        return estimator

    plain = solved(wnoa.CovarianceOverrides())
    same = solved(wnoa.CovarianceOverrides().overrideQ2(lambda s, nominal: nominal))
    scaled = solved(wnoa.CovarianceOverrides().overrideQ2(0.0, 0.01 * Q2))
    assert plain.getFactors().size() == scaled.getFactors().size() > 0
    values = plain.getCurrentValues()
    for k in values.keys():  # move off the zero-error initial guess
        if gtsam.Symbol(k).chr() == ord('p'):
            values.update(k, values.atPose3(k).retract(np.full(6, 0.01)))
    assert same.getFactors().error(values) == plain.getFactors().error(values)
    assert scaled.getFactors().error(values) > plain.getFactors().error(values)


def test_config_loader_reads_covariance_overrides(tmp_path):
    from spacetime import utilities
    path = tmp_path / "config.json"
    path.write_text('{"prior": {"wnoa": {"weights": {"prior_factor": 0.5}, "overrides": {'
                    '"Q2": [{"arclength": 0.0, "diagonal": [2, 2, 2, 4, 4, 4]}, {"arclength": 0.1, "scale": 3}]}}}}')
    loader = utilities.ConfigLoader()
    assert loader.loadConfig(str(path))
    overrides = loader.loadWNOACovarianceOverrides()
    prior = wnoa.Prior(wnoa.ContinuumRobotState(0.2, 3), wnoa.NodeValues(), overrides=overrides)
    np.testing.assert_array_equal(prior.covarianceProfiles().Q2.at(0), np.diag([1.0, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(prior.covarianceProfiles().Q2.at(1), 3 * wnoa.DefaultCovariances()[2])
    assert loader.loadWNOAQuasiStaticCovarianceOverrides().Q2.empty() is False
    assert loader.loadWNOVCovarianceOverrides().empty()
