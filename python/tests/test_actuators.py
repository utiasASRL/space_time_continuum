"""Built-in actuators: the kinematic tendon, from construction to an estimate."""

import importlib.util

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime import actuators, joints, types
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import REPO_ROOT, STRAIGHT, batch_estimator, make_wnoa_robot, run_session


def test_kinematic_tendon_routing():
    _, prior = make_wnoa_robot(length=1.0, nodes=5)
    tendon = actuators.KinematicTendon(prior, radius=0.01, angle=0.0, termination=1.0, discretization=0.2)
    assert isinstance(tendon, actuators.KinematicTendonWNOA)
    assert isinstance(tendon, types.ActuatorBase)
    np.testing.assert_allclose(tendon.arclengths(), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert tendon.inputDim() == 1
    assert tendon.radius == 0.01 and tendon.angle == 0.0
    assert tendon.termination == 1.0 and tendon.discretization == pytest.approx(0.2)
    np.testing.assert_allclose(tendon.offset, [0.0, 0.01, 0.0])
    tendon.lengthSigma = 1e-3
    assert tendon.lengthSigma == 1e-3

    uneven = actuators.KinematicTendon(prior, radius=0.01, angle=np.pi / 2, termination=1.0, discretization=0.3)
    np.testing.assert_allclose(uneven.arclengths(), [0.0, 0.3, 0.6, 0.9, 1.0])
    np.testing.assert_allclose(uneven.offset, [0.0, 0.0, 0.01], atol=1e-15)


def test_kinematic_tendon_with_explicit_disks():
    _, prior = make_wnoa_robot(length=1.0, nodes=5)
    tendon = actuators.KinematicTendon(prior, radius=0.01, angle=0.0, arclengths=[0.1, 0.25, 0.7, 0.95])
    assert isinstance(tendon, actuators.KinematicTendonWNOA)
    np.testing.assert_allclose(tendon.arclengths(), [0.1, 0.25, 0.7, 0.95])
    assert tendon.termination == pytest.approx(0.95)
    assert tendon.discretization is None
    same = actuators.KinematicTendonWNOA(prior, 0.01, 0.0, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(same.arclengths(), [0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        actuators.KinematicTendon(prior, radius=0.01, angle=0.0, arclengths=[0.5])
    with pytest.raises(ValueError):
        actuators.KinematicTendon(prior, radius=0.01, angle=0.0, arclengths=[0.6, 0.3])
    with pytest.raises(ValueError):
        actuators.KinematicTendon(prior, radius=0.01, angle=0.0, arclengths=[0.0, 1.5])


def test_kinematic_tendon_dispatches_on_the_prior():
    state = wnov.ContinuumRobotState(1.0, 3)
    tendon = actuators.KinematicTendon(wnov.Prior(state, gtsam.Pose3(), STRAIGHT), 0.01, 0.0, 1.0, 0.5)
    assert isinstance(tendon, actuators.KinematicTendonWNOV)
    state = wnoa_quasi_static.ContinuumRobotState(1.0, 3)
    tendon = actuators.KinematicTendon(wnoa_quasi_static.Prior(state, wnoa_quasi_static.NodeValues()), 0.01, 0.0, 1.0, 0.5)
    assert isinstance(tendon, actuators.KinematicTendonQuasiStatic)
    end_effector_state = joints.EndEffectorState([0.0])
    end_effector = joints.EndEffectorPrior(end_effector_state, gtsam.Pose3(), joints.EndEffectorPrior.DefaultCovariances())
    with pytest.raises(TypeError):
        actuators.KinematicTendon(end_effector, 0.01, 0.0, 1.0, 0.5)


def test_kinematic_tendon_invalid_arguments_raise():
    _, prior = make_wnoa_robot(length=1.0, nodes=5)
    with pytest.raises(ValueError):
        actuators.KinematicTendon(prior, radius=-0.01, angle=0.0, termination=1.0, discretization=0.2)
    with pytest.raises(ValueError):
        actuators.KinematicTendon(prior, radius=0.01, angle=0.0, termination=1.5, discretization=0.2)
    with pytest.raises(ValueError):
        actuators.KinematicTendon(prior, radius=0.01, angle=0.0, termination=1.0, discretization=0.0)


def test_predicted_length_of_a_straight_robot_is_the_termination():
    _, prior = make_wnoa_robot(length=1.0, nodes=5)
    tendon = actuators.KinematicTendon(prior, radius=0.02, angle=0.3, termination=0.8, discretization=0.2)
    poses = [gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(s, 0.0, 0.0)) for s in tendon.arclengths()]
    assert tendon.predictedLength(poses) == pytest.approx(0.8)
    assert actuators.TendonLengthFactor.pathLength(poses, tendon.offset) == pytest.approx(0.8)


def test_tendon_length_factor_error():
    keys = [gtsam.symbol("a", i) for i in range(3)]
    u = gtsam.symbol("u", 0)
    factor = actuators.TendonLengthFactor(keys, u, np.array([0.0, 0.01, 0.0]), gtsam.noiseModel.Isotropic.Sigma(1, 1e-3))
    values = gtsam.Values()
    for i, key in enumerate(keys):
        values.insert(key, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.5 * i, 0.0, 0.0)))
    values.insert(u, np.array([0.9]))
    np.testing.assert_allclose(factor.unwhitenedError(values), [0.1])


def test_estimator_fits_the_robot_to_the_tendon_length():
    state, prior = make_wnoa_robot(length=1.0, nodes=5)
    tendon = actuators.KinematicTendon(prior, radius=0.01, angle=0.0, termination=1.0, discretization=0.2)
    tendon.addMeasurement(types.MeasurementBase(time=0.0, value=0.97, covariance=1e-8))
    estimator = batch_estimator()
    estimator.addPrior(prior)
    estimator.addActuator(tendon)
    run_session(estimator, end_time=0.0)

    values = estimator.getCurrentValues()
    (_, key), = tendon.inputKeys()
    assert values.atVector(key)[0] == pytest.approx(0.97, abs=1e-4)
    poses = [prior.query(0.0, s, values).pose() for s in tendon.arclengths()]
    assert tendon.predictedLength(poses) == pytest.approx(0.97, abs=1e-4)


def test_guiding_example_runs():
    spec = importlib.util.spec_from_file_location("kinematic_tendon", REPO_ROOT / "python" / "examples" / "kinematic_tendon.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    estimator, prior, actuator = example.main()
    values = estimator.getCurrentValues()
    (_, key), = actuator.inputKeys()
    poses = [prior.query(0.0, s, values).pose() for s in actuator.arclengths()]
    # The tendon length variable agrees with the path the estimate traces.
    assert actuator.predictedLength(poses) == pytest.approx(values.atVector(key)[0], abs=1e-5)
