"""Defining your own sensor and model in Python.

A single straight robot, its base locked, is bent by measurements of its tip pose from a sensor
written in Python (TipSensor) instead of the built-in sensors.PoseSensor; a Python Model
(Logger) watches the estimation. Python sensors and models build their factors from existing
C++ factors -- GTSAM's own or the library's (sensors.PoseFactor, StrainFactor, GyroFactor) -- on
the prior's interpolated keys, so the optimization itself stays in C++.

Usage: python custom_sensor.py
"""

import math

import gtsam
import numpy as np

import spacetime as st
from spacetime import interface, types
from spacetime.priors import wnoa

LENGTH, NODES, PERIOD, END_TIME = 0.2, 5, 0.1, 1.0


class TipSensor(types.SensorBase):
    """A pose sensor: each measurement becomes GTSAM's PriorFactorPose3 on the interpolated pose."""

    def getFactor(self, measurement):
        noise = gtsam.noiseModel.Gaussian.Covariance(measurement.covariance)
        return gtsam.PriorFactorPose3(wnoa.InterpolatedKeys.Pose(), gtsam.Pose3(measurement.value), noise)


class Logger(types.Model):
    """A model adding no factors: it reports every estimation time."""

    def getNewFactors(self, time, new_values):
        print(f"t = {time:.1f}: {new_values.size()} new variables")
        return []


def main():
    state = wnoa.ContinuumRobotState(LENGTH, NODES)
    reference = wnoa.Values()
    reference.epsilon(np.array([0.0, 0, 0, 1, 0, 0]))  # straight, [rot, trans]
    prior = wnoa.Prior(state, reference)
    state.lock("pose", 0.0, gtsam.Pose3())

    options = st.EstimatorOptions()
    options.smootherLag = st.Estimator.BATCH_LAG
    estimator = st.Estimator(options)
    estimator.addModel(prior)
    estimator.addModel(Logger())

    # The tip, bending up to a quarter turn over the run.
    sensor = TipSensor(prior)
    covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    for k in range(int(round(END_TIME / PERIOD)) + 1):
        t = k * PERIOD
        curvature = (math.pi / 2) / LENGTH * t / END_TIME
        tip = gtsam.Pose3.Expmap(LENGTH * np.array([0.0, curvature, 0, 1, 0, 0]))
        sensor.addMeasurement(types.MeasurementBase(LENGTH, t, tip.matrix(), covariance))
    estimator.addSensor(sensor)

    session_options = interface.SessionOptions()
    session_options.end_time = END_TIME
    session_options.period = PERIOD
    session = interface.Session(estimator, session_options)
    session.addMeasurementQueues(estimator.getMeasurementQueues())
    session.start()
    session.join()

    tip = prior.query(session.endTime(), LENGTH, estimator.getCurrentValues()).pose()
    print(f"Estimated tip at t = {session.endTime():g}: {tip.translation()}, "
          f"rotated by {np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(tip.rotation()))):.1f} deg")


if __name__ == "__main__":
    main()
