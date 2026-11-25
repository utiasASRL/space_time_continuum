"""Built-in sensors and their factories."""

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime import sensors, types
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import STRAIGHT, estimator_options, make_wnoa_robot


def wnov_prior():
    state = wnov.ContinuumRobotState(0.2, 3)
    return state, wnov.Prior(state, gtsam.Pose3(), STRAIGHT)


def quasi_static_prior():
    state = wnoa_quasi_static.ContinuumRobotState(0.2, 3)
    return state, wnoa_quasi_static.Prior(state, wnoa_quasi_static.NodeValues(gtsam.Pose3(), STRAIGHT))


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


def test_gyro_options_default_to_no_bias():
    _, prior = make_wnoa_robot()
    gyro = sensors.GyroSensor(prior)
    assert not gyro.options().estimate_bias
    assert gyro.biasModel() is None

    options = sensors.GyroOptions()
    options.estimate_bias = True
    options.offset = gtsam.Pose3(gtsam.Rot3.Rz(0.3), np.zeros(3))
    biased = sensors.GyroSensor(prior, options)
    assert isinstance(biased, sensors.GyroSensorWNOA)
    assert biased.options().estimate_bias
    assert isinstance(biased.biasModel(), sensors.BiasModel3)
    np.testing.assert_allclose(biased.biasModel().options().initial_covariance, sensors.GyroOptions.defaultBias().initial_covariance)


def test_gyro_biased_factor_is_constructible_directly():
    b0, b1 = gtsam.symbol("b", 0), gtsam.symbol("b", 1)
    factor = sensors.GyroBiasedFactor(gtsam.noiseModel.Unit.Create(3), np.array([0.1, 0.2, 0.3]), np.ones((3, 1), dtype=np.int32),
                                      [(b0, 0.5), (b1, 0.5)])
    assert list(factor.keys())[1:] == [b0, b1]


def test_accelerometer_and_imu_factories():
    _, prior = make_wnoa_robot()
    accelerometer = sensors.AccelerometerSensor(prior, 0.1)
    assert isinstance(accelerometer, sensors.AccelerometerSensorWNOA)
    assert accelerometer.arclength() == pytest.approx(0.1)
    assert accelerometer.biasModel() is not None

    imu = sensors.IMUSensor(prior, 0.2)
    assert isinstance(imu, sensors.IMUSensorWNOA)
    assert isinstance(imu, types.SensorBase)
    # Both biases are estimated by default.
    assert imu.options().gyro.estimate_bias and imu.options().accelerometer.estimate_bias
    assert imu.gyro().biasModel() is not None and imu.accelerometer().biasModel() is not None

    with pytest.raises(ValueError):
        sensors.IMUSensor(prior, 1.0)  # past the robot
    _, prior_q = quasi_static_prior()
    with pytest.raises(TypeError):
        sensors.IMUSensor(prior_q, 0.1)
    with pytest.raises(TypeError):
        sensors.AccelerometerSensor(prior_q, 0.1)


def test_imu_estimates_a_robot_at_rest():
    _, prior = make_wnoa_robot()
    imu = sensors.IMUSensor(prior, 0.2)
    estimator = st.Estimator(estimator_options(0.5))
    estimator.addModel(prior)
    estimator.addSensor(imu)
    assert len(estimator.getMeasurementQueues()) == 1

    # The base is locked: at rest, the gyro reads zero and the accelerometer -g (along the robot's z).
    covariance = np.diag([1e-4] * 3 + [1e-2] * 3)
    for i in range(51):
        value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81]).reshape(6, 1)
        imu.addMeasurement(types.MeasurementBase(0.2, 0.01 * i, value, covariance, np.ones((6, 1), dtype=np.int32)))
    for k in range(6):
        estimator.estimate(0.1 * k)
    bias = imu.gyro().biasModel().query(0.5, 0.0, estimator.getCurrentValues())
    assert isinstance(bias, sensors.BiasNodeValues3)
    assert np.linalg.norm(bias.bias()) < 1e-2
