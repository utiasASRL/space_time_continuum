"""Models, sensors, actuators, stateful models (priors), states and nodes defined in Python."""

import gc

import gtsam
import numpy as np
import pytest

from spacetime import sensors, types
from spacetime.priors import wnoa

from conftest import POSE_COVARIANCE, batch_estimator, bent_tip, make_wnoa_robot, pose_measurement, run_session

LENGTH, CURVATURE, END_TIME = 0.2, 3.0, 0.5


class PythonPoseSensor(types.SensorBase):
    """The built-in PoseSensor, written in Python: builds the same C++ factor."""

    def getFactor(self, measurement):
        noise = gtsam.noiseModel.Gaussian.Covariance(measurement.covariance)
        return sensors.PoseFactor(noise, gtsam.Pose3(measurement.value), measurement.mask)


class GtsamPriorSensor(types.SensorBase):
    """A pose sensor returning GTSAM's own PriorFactorPose3 on the interpolated pose."""

    def getFactor(self, measurement):
        noise = gtsam.noiseModel.Gaussian.Covariance(measurement.covariance)
        return gtsam.PriorFactorPose3(wnoa.InterpolatedKeys.Pose(), gtsam.Pose3(measurement.value), noise)


def solve_with_tip_sensor(sensor_factory):
    _, prior = make_wnoa_robot(length=LENGTH)
    estimator = batch_estimator()
    estimator.addModel(prior)
    sensor = sensor_factory(prior)
    for k in range(6):
        sensor.addMeasurement(pose_measurement(LENGTH, 0.1 * k, bent_tip(LENGTH, CURVATURE), POSE_COVARIANCE))
    estimator.addSensor(sensor)
    run_session(estimator, end_time=END_TIME)
    return estimator.getCurrentValues(), prior


def assert_values_equal(a, b, tol):
    assert a.size() == b.size()
    assert a.equals(b, tol)


def test_python_sensor_matches_builtin_sensor():
    builtin, _ = solve_with_tip_sensor(lambda prior: sensors.PoseSensor(prior))
    python, _ = solve_with_tip_sensor(PythonPoseSensor)
    assert_values_equal(builtin, python, 1e-12)


def test_python_sensor_returning_none_drops_measurement():
    class Silent(types.SensorBase):
        def getFactor(self, measurement):
            return None

    values, prior = solve_with_tip_sensor(Silent)
    straight_tip = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(LENGTH, 0, 0))
    assert prior.query(END_TIME, LENGTH, values).pose().equals(straight_tip, 1e-6)


def test_python_sensor_survives_without_python_references():
    state, prior = make_wnoa_robot(length=LENGTH)
    estimator = batch_estimator()
    estimator.addModel(prior)

    def add_sensor():
        sensor = PythonPoseSensor(prior)
        for k in range(6):
            sensor.addMeasurement(pose_measurement(LENGTH, 0.1 * k, bent_tip(LENGTH, CURVATURE), POSE_COVARIANCE))
        estimator.addSensor(sensor)

    add_sensor()
    gc.collect()
    run_session(estimator, end_time=END_TIME)
    assert prior.query(END_TIME, LENGTH, estimator.getCurrentValues()).pose().equals(bent_tip(LENGTH, CURVATURE), 1e-2)


def test_exception_in_python_override_propagates_out_of_the_session():
    class Broken(types.SensorBase):
        def getFactor(self, measurement):
            raise ValueError("broken sensor")

    with pytest.raises(ValueError, match="broken sensor"):
        solve_with_tip_sensor(Broken)


def test_python_sensor_must_implement_get_factor():
    class Incomplete(types.SensorBase):
        pass

    with pytest.raises(RuntimeError):
        solve_with_tip_sensor(Incomplete)


def test_python_model_adds_factors():
    """A Model pulling every new tip node to a target pose, with its own (non-sensor) factors."""
    state, prior = make_wnoa_robot(length=LENGTH)
    target = bent_tip(LENGTH, CURVATURE)
    calls = []

    class TipTarget(types.Model):
        def extendState(self, km, time, current_values, new_values):
            calls.append(("extend", time))
            return []

        def getNewFactors(self, time, new_values):
            calls.append(("factors", time))
            tip = state.snapshotEstimationNodes()[-1]
            noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            return [gtsam.PriorFactorPose3(tip.poseKey(), target, noise)]

    estimator = batch_estimator()
    estimator.addModel(prior)
    estimator.addModel(TipTarget())
    run_session(estimator, end_time=END_TIME)
    assert calls[0] == ("extend", 0.0) and calls[1] == ("factors", 0.0)
    assert prior.query(END_TIME, LENGTH, estimator.getCurrentValues()).pose().equals(target, 1e-2)


def test_python_model_defaults():
    model = types.Model()
    assert model.extendState(types.KeyManager(), 0.0, gtsam.Values(), gtsam.Values()) == []
    assert model.getNewFactors(0.0, gtsam.Values()) == []
    assert model.lockedValues().size() == 0
    assert model.covarianceKeyGroups(gtsam.Values()) == []
    model.onMarginalization(0.0, gtsam.Values())


def test_python_actuator_sets_strain():
    """An actuator commanding the full strain at the tip: its factor ties the interpolated strain to
    the input u_k (error = strain - u), as tests/ActuatorBaseTests.cpp's StrainInputFactor does."""
    commanded = np.array([0.0, CURVATURE, 0.0, 1.0, 0.0, 0.0])
    strain_key = wnoa.InterpolatedKeys.Strain()

    def strain_minus_input(factor, values, jacobians):
        strain, u = values.atVector6(factor.keys()[0]), values.atVector(factor.keys()[1])  # a fixed-size strain, a dynamic input
        if jacobians is not None:
            jacobians[0] = np.eye(6)
            jacobians[1] = -np.eye(6)
        return strain - u

    class StrainActuator(types.ActuatorBase):
        def getFactor(self, measurement, input_key, arclength):
            assert arclength == LENGTH
            noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            return gtsam.CustomFactor(noise, [strain_key, input_key], strain_minus_input)

    state, prior = make_wnoa_robot(length=LENGTH)
    actuator = StrainActuator(prior, [LENGTH], 6)
    assert actuator.inputDim() == 6 and list(actuator.arclengths()) == [LENGTH]
    for k in range(6):
        actuator.addMeasurement(types.MeasurementBase(0.0, 0.1 * k, commanded.reshape(6, 1), np.eye(6) * 1e-6))
    estimator = batch_estimator()
    estimator.addModel(prior)
    estimator.addModel(actuator)
    run_session(estimator, end_time=END_TIME)
    np.testing.assert_allclose(prior.query(END_TIME, LENGTH, estimator.getCurrentValues()).epsilon(), commanded,
                               atol=1e-2)
    assert len(actuator.inputKeys()) == 6


def test_python_actuator_invalid_arguments_raise():
    class Noop(types.ActuatorBase):
        def getFactor(self, measurement, input_key, arclength):
            return None

    _, prior = make_wnoa_robot()
    with pytest.raises(ValueError):
        Noop(prior, [], 6)
    with pytest.raises(ValueError):
        Noop(prior, [0.1], 0)


# A rigid body written entirely in Python: its node, node values, state, prior and robot.

class BodyValues(types.NodeValues):
    def __init__(self, pose):
        super().__init__()
        self.values = [pose.matrix()]

    def pose(self):
        return gtsam.Pose3(self.values[0])


class BodyNode(types.NodeBase):
    def __init__(self, km, time):
        super().__init__(km, 0.0, time)
        self.pose_key = self.registerVariable(km, "pose", "x", gtsam.Pose3)

    def interpolatedKeys(self):
        return [(gtsam.Symbol("q", 0).key(), self.pose_key)]

    def getValuesBase(self, values):
        return BodyValues(values.atPose3(self.pose_key))


class BodyState(types.State):
    def __init__(self):
        super().__init__()
        self.nodes = []

    def addStateAtTime(self, km, time):
        self.nodes.append(BodyNode(km, time))

    def getNearestNode(self, arclength, time):
        if not self.nodes:
            return None
        return min(self.nodes, key=lambda node: abs(node.time - time))


class RandomWalkPrior(types.StatefulModel):
    def __init__(self, state, start):
        super().__init__(state)
        self.body = state
        self.start = start

    def extendState(self, km, time, current_values, new_values):
        previous = self.body.nodes[-1] if self.body.nodes else None
        self.body.addStateAtTime(km, time)
        node = self.body.nodes[-1]
        node.insert("pose", self.start, new_values)
        if previous is None:
            return [gtsam.PriorFactorPose3(node.pose_key, self.start, gtsam.noiseModel.Isotropic.Sigma(6, 1.0))]
        return [gtsam.BetweenFactorPose3(previous.pose_key, node.pose_key, gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 0.1))]


def test_python_prior_state_and_node():
    state = BodyState()
    prior = RandomWalkPrior(state, gtsam.Pose3())
    assert prior.state() is state

    target = gtsam.Pose3(gtsam.Rot3.Rz(0.3), gtsam.Point3(0.1, 0.2, 0.3))
    sensor = GtsamPriorSensor(prior)
    for k in range(6):
        sensor.addMeasurement(pose_measurement(0.0, 0.1 * k, target, POSE_COVARIANCE))
    assert sensor.stateful() is prior

    # The robot: its prior and sensor, added together.
    body = types.Robot()
    body.addModel(prior)
    body.addModel(sensor)
    estimator = batch_estimator()
    estimator.addRobot(body)
    run_session(estimator, end_time=END_TIME)

    assert len(state.nodes) == 6
    node = prior.query(END_TIME, 0.0, estimator.getCurrentValues())
    assert isinstance(node, BodyValues)
    assert node.pose().equals(target, 1e-2)


def test_node_insert_type_mismatch_raises():
    km = types.KeyManager()
    node = BodyNode(km, 0.0)
    with pytest.raises(RuntimeError):
        node.insert("pose", np.zeros(6), gtsam.Values())
    assert list(node.names()) == ["pose"]
    assert node.pose_key == node.key("pose")
    assert list(node.keys()) == [node.pose_key]


def test_register_variable_unsupported_type_raises():
    class Bad(types.NodeBase):
        def __init__(self, km):
            super().__init__(km, 0.0, 0.0)
            self.registerVariable(km, "thing", "t", dict)

    with pytest.raises(TypeError):
        Bad(types.KeyManager())
