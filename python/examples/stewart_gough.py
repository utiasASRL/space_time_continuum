"""Six continuum robots as the legs of a Stewart-Gough platform: the Python port of
examples/stewart_gough.cpp.

Each leg's tip is rigidly attached to the platform: an end effector whose frame sits at the
platform's centre, equidistant from every tip. The bases are locked; each leg is actuated by
elongation, measured at its base by a strain sensor that sees only the elongation component.
Nothing on the platform is measured: its motion comes from the six leg elongations through the
joints, and the end effector has no prior of its own in space.

Geometry (world z up): base anchors at 0/120/240 deg +- 12 deg on a 15 cm circle, platform
anchors on an 8 cm circle rotated by 60 deg (the classic 6-6 layout). The platform height makes
every straight leg exactly its length, so the initial guess satisfies the joints.

Usage: python stewart_gough.py [--headless] [--output DIR]
"""

import math
import sys

import gtsam
import numpy as np

import spacetime as st
from spacetime import joints, sensors, types

from joint_demo import StrainPrior, make_robot, parse_arguments, rotation_with_x_axis, run

PERIOD = 0.1
END_TIME = 8.0
LEG_LENGTH = 0.3
NODES = 3
BASE_RADIUS, PLATFORM_RADIUS = 0.15, 0.08
TRANSLATION, TILT, TWIST = 0.05, 50.0, 15.0  # platform motion amplitudes [m], [deg]
DEG = math.pi / 180.0
ELONGATION_MEASUREMENT_VARIANCE = 1e-8

BASE_ANGLES = [-12, 12, 108, 132, 228, 252]
PLATFORM_ANGLES = [-48, 48, 72, 168, 192, 288]


def on_circle(radius, degrees, z):
    return np.array([radius * math.cos(degrees * DEG), radius * math.sin(degrees * DEG), z])


def main():
    args = parse_arguments(__doc__.splitlines()[0])

    # Every leg spans the same horizontal distance; pick the height that makes it straight.
    span = np.linalg.norm(on_circle(PLATFORM_RADIUS, PLATFORM_ANGLES[0], 0) - on_circle(BASE_RADIUS, BASE_ANGLES[0], 0))
    height = math.sqrt(LEG_LENGTH * LEG_LENGTH - span * span)

    # Locked bases. Each leg is actuated by elongation: uniform along its backbone (stiff in the
    # arclength prior), free at its base (loose in the unary factor), and set there by a tight
    # elongation measurement.
    elongation_prior = StrainPrior()
    elongation_prior.unary = 1e2
    elongation_prior.arclength = 1e-6
    robots, anchors, nominal_tips = [], [], []
    for i in range(6):
        base, tip = on_circle(BASE_RADIUS, BASE_ANGLES[i], 0), on_circle(PLATFORM_RADIUS, PLATFORM_ANGLES[i], height)
        base_pose = gtsam.Pose3(rotation_with_x_axis(tip - base), gtsam.Point3(base))
        robots.append(make_robot(LEG_LENGTH, NODES, base_pose, np.array([0.0, 0, 0, 1, 0, 0]), True, elongation_prior))
        anchors.append(base)
        nominal_tips.append(base_pose.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(LEG_LENGTH, 0, 0))))

    estimator_options = st.EstimatorOptions()
    estimator_options.smootherLag = st.Estimator.BATCH_LAG
    estimator_options.compute_covariance = True
    estimator = st.Estimator(estimator_options)
    for robot in robots:
        estimator.addPrior(robot.prior)

    # The platform: an end effector framed at its centre, every tip rigidly attached to it at
    # its nominal pose, held to 1 mm / 1 mrad.
    platform0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, height))
    platform_ee = joints.EndEffector(platform0)
    for robot, nominal_tip in zip(robots, nominal_tips):
        options = joints.JointOptions()
        options.offset = nominal_tip.between(platform0)
        platform_ee.attach(robot.prior, LEG_LENGTH, np.eye(6) * 1e-6, options)
    estimator.addModel(platform_ee)

    # The target platform motion, relative to its nominal pose; it starts there at t = 0.
    def platform(t):
        phase = 2.0 * math.pi * t / END_TIME
        rotation = gtsam.Rot3.RzRyRx(TILT * DEG * math.sin(phase), TILT * DEG * math.sin(2.0 * phase), TWIST * DEG * math.sin(3.0 * phase))
        translation = gtsam.Point3(TRANSLATION * math.sin(2.0 * phase), TRANSLATION * (1.0 - math.cos(phase)),
                                   height + TRANSLATION * math.sin(phase))
        return gtsam.Pose3(rotation, translation)

    # Each leg's commanded elongation (stretch, 1 = rest length): the distance from its anchor to
    # its target tip over its rest length, as a prismatic leg would extend.
    def elongation(leg, t):
        tip = platform(t).compose(platform0.between(nominal_tips[leg]))
        return np.linalg.norm(tip.translation() - anchors[leg]) / LEG_LENGTH

    # The actuators: each base's elongation strain alone (the other components masked out), measured tightly.
    covariance = np.diag([1, 1, 1, ELONGATION_MEASUREMENT_VARIANCE, 1, 1])
    elongation_only = np.zeros((6, 1), dtype=np.int32)
    elongation_only[3, 0] = 1
    for i, robot in enumerate(robots):
        sensor = sensors.FBGSensor(robot.prior)
        k = 0
        while k * PERIOD <= END_TIME + 1e-9:
            t = k * PERIOD
            measured = np.array([0.0, 0, 0, elongation(i, t), 0, 0]).reshape(6, 1)
            sensor.addMeasurement(types.MeasurementBase(0.0, t, measured, covariance, elongation_only))
            k += 1
        estimator.addSensor(sensor)

    return run(estimator, robots, [], [platform_ee], 0.0, END_TIME, PERIOD, args)


if __name__ == "__main__":
    sys.exit(main())
