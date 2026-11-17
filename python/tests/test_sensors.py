"""Built-in sensors and their factories."""

import gtsam
import numpy as np
import pytest

from spacetime import sensors, types
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import STRAIGHT, make_wnoa_robot


def wnov_prior():
    state = wnov.ContinuumRobotState(0.2, 3)
    return state, wnov.Prior(state, gtsam.Pose3(), STRAIGHT)


def quasi_static_prior():
    state = wnoa_quasi_static.ContinuumRobotState(0.2, 3)
    return state, wnoa_quasi_static.Prior(state, wnoa_quasi_static.Values(gtsam.Pose3(), STRAIGHT))


def test_factories_pick_the_prior_values_type():
    _, prior = make_wnoa_robot()
    assert isinstance(sensors.PoseSensor(prior), sensors.PoseSensorWNOA)
    assert isinstance(sensors.FBGSensor(prior), sensors.FBGSensorWNOA)
    assert isinstance(sensors.GyroSensor(prior), sensors.GyroSensorWNOA)
    _, prior_v = wnov_prior()
    assert isinstance(sensors.PoseSensor(prior_v), sensors.PoseSensorWNOV)
    _, prior_q = quasi_static_prior()
    assert isinstance(sensors.PoseSensor(prior_q), sensors.PoseSensorQuasiStatic)
    assert isinstance(sensors.FBGSensor(prior_q), sensors.FBGSensorQuasiStatic)
    with pytest.raises(TypeError):
        sensors.GyroSensor(prior_q)  # no velocity in a quasi-static node


def test_sensors_are_models_and_queues():
    _, prior = make_wnoa_robot()
    sensor = sensors.PoseSensor(prior)
    assert isinstance(sensor, types.SensorBase)
    assert isinstance(sensor, types.Model)
    assert isinstance(sensor, types.MeasurementQueue)
    assert sensor.stateful() is prior


def test_null_prior_raises():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        sensors.PoseSensor(None)


def test_unsupported_prior_raises():
    class OwnPrior(types.StatefulModel):
        def extendState(self, km, time, current_values, new_values):
            return []

    prior = OwnPrior(types.State())
    with pytest.raises(TypeError):
        sensors.PoseSensor(prior)


def test_pose_sensor_get_factor():
    _, prior = make_wnoa_robot()
    sensor = sensors.PoseSensor(prior)
    measurement = types.MeasurementBase(0.1, 0.0, gtsam.Pose3().matrix(), np.eye(6), np.ones((6, 1), dtype=np.int32))
    factor = sensor.getFactor(measurement)
    assert isinstance(factor, gtsam.NoiseModelFactor)
    assert list(factor.keys()) == [wnoa.InterpolatedKeys.Pose()]


def test_pose_factor_is_constructible_directly():
    noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6))
    mask = np.ones((6, 1), dtype=np.int32)
    factor = sensors.PoseFactor(noise, gtsam.Pose3(), mask)
    values = gtsam.Values()
    values.insert(wnoa.InterpolatedKeys.Pose(), gtsam.Pose3())
    assert factor.error(values) == pytest.approx(0.0)
