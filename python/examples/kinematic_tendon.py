"""A robot with a pose sensor and a kinematic tendon.

A 1 m WNOA robot, its base pose and tip strain locked at their initial guesses, with the default
tuned prior covariances (wnoa.DefaultCovariances()). A pose sensor measures the tip, and a tendon
routed 1 cm from the backbone through disks every 0.2 m (actuators.KinematicTendon) measures its
own length: each tendon length becomes a variable compared with the straight-line path through
the disks traced by the estimate. The estimate is solved once, at t = 0, and printed.

Usage: python kinematic_tendon.py
"""

import numpy as np

from spacetime import *  # noqa: F403  -- Estimator, Session, priors, sensors, actuators, types


def main():
    estimator = Estimator()  # batch

    # Model construction goes here
    Model = priors.wnoa

    ### State construction
    state = Model.ContinuumRobotState(length=1.0, spatial_dim=5)
    state.lock("pose", 0.0)  # the base pose, at its initial guess (the boundary pose)
    state.lock("strain", 1.0)  # the tip strain, at its initial guess (straight)

    ### Prior construction
    boundaryEstimate = Model.NodeValues() # Defaults to identity
    prior = Model.Prior(state, boundaryEstimate) # Default Q values
    estimator.addPrior(prior)

    ### Sensor construction
    sensor = sensors.PoseSensor(prior)
    estimator.addSensor(sensor)

    ### Sensor data filling (possibly asynchronous)
    data = np.eye(4)
    data[1, 3] = 0.2
    sensor.addMeasurement(types.MeasurementBase(arclength=1.0, time=0.0, value=data, covariance=np.eye(6)))

    ### Actuator construction
    actuator = actuators.KinematicTendon(prior, radius=0.01, angle=0.0, termination=1.0, discretization=0.2)
    estimator.addActuator(actuator)

    ### Actuator data filling (possibly asynchronous)
    data = 1.1  # Tendon length
    actuator.addMeasurement(types.MeasurementBase(time=0.0, value=data, covariance=1.0))

    session = Session()
    # Visualizer construction goes here
    session.start(estimator)  # Begins estimation
    session.join()  # Ends estimation

    estimator.print()
    return estimator, prior, actuator


if __name__ == "__main__":
    main()
