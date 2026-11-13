"""A two-leg parallel continuum robot: the Python port of examples/robot_joining_parallel.cpp.

Two equal-length robots with fixed (locked) bases, their tips rigidly attached to a shared end
effector -- a platform with a width, so the tips never touch. Only the end effector is measured:
its pose, as it is moved around. Nothing on the robots is measured; their shapes come from the
joints and their priors. Both priors are stiff in elongation and shear, so the legs bend rather
than stretch or shear to follow.

Geometry (world z up): the bases stand upright 20 cm apart along x and each leg bows inward in
the x-z plane (constant curvature); the platform is level, centred between the two tips. This
initial guess satisfies the joints exactly. Everything is hard-coded; no configuration file.

Motion: the platform swings 27 cm out of the legs' plane (along y), dipping and rolling with the
swing as a pair of bending legs would carry it, and yaws -- motions the legs make by bending alone.

Usage: python robot_joining_parallel.py [--headless] [--output DIR]
"""

import math
import sys

import gtsam
import numpy as np

import spacetime as st
from spacetime import joints, sensors, types

from joint_demo import StrainPrior, make_robot, parse_arguments, run

PERIOD = 0.1
END_TIME = 5.0
LENGTH = 0.35
NODES = 5
BASE_HALF_SPACING = 0.1  # [m] each base's distance from the centre
BOW_CURVATURE = 1.0      # [rad/m] of each leg's initial inward bow
STRETCH_VARIANCE = 1e-2  # each leg's elongation and shear priors, relative to its torsion (identity)
BENDING_VARIANCE = 1e3   # each leg's bending arclength PSD: compliant, so its large bends are cheap
SWING = 0.27             # platform swing amplitude [m]
YAW = 10.0               # [deg]
DEG = math.pi / 180.0


def strain(curvature):
    """Bending in the world x-z plane (about body y)."""
    return np.array([0.0, curvature, 0.0, 1.0, 0.0, 0.0])


def main():
    args = parse_arguments(__doc__.splitlines()[0])
    upright = gtsam.Pose3(gtsam.Rot3.Ry(-math.pi / 2), gtsam.Point3(0, 0, 0))  # body x along world z

    # The legs, mirror images in x: the right one (+x) bows towards -x and vice versa.
    bending = StrainPrior()
    bending.arclength = BENDING_VARIANCE
    robots, tips = [], []
    for side in (-1.0, 1.0):
        base = gtsam.Pose3(upright.rotation(), gtsam.Point3(side * BASE_HALF_SPACING, 0, 0))
        bow = strain(-side * BOW_CURVATURE)
        robots.append(make_robot(LENGTH, NODES, base, bow, True, StrainPrior.uniform(STRETCH_VARIANCE),
                                 StrainPrior.uniform(STRETCH_VARIANCE), bending))
        tips.append(base.compose(gtsam.Pose3.Expmap(LENGTH * bow)))

    estimator_options = st.EstimatorOptions()
    estimator_options.smootherLag = st.Estimator.BATCH_LAG
    estimator_options.compute_covariance = True
    estimator = st.Estimator(estimator_options)
    for robot in robots:
        estimator.addPrior(robot.prior)

    # The platform: level, centred between the tips, each tip rigidly attached at its nominal
    # pose and held to 1 mm / 1 mrad.
    platform0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.5 * (tips[0].translation() + tips[1].translation())))
    platform_ee = joints.EndEffector(platform0)
    for robot, tip in zip(robots, tips):
        options = joints.JointOptions()
        options.offset = tip.between(platform0)
        platform_ee.attach(robot.prior, LENGTH, np.eye(6) * 1e-6, options)
    estimator.addModel(platform_ee)

    # The platform's motion, relative to its nominal pose; it starts and ends there. The dip dz(y)
    # and roll(y) are fits of the least-bending platform pose for each swing y (see the C++ demo).
    def platform(t):
        phase = 2.0 * math.pi * t / END_TIME
        y = SWING * math.sin(phase)
        y2 = y * y
        dz = y2 * (-1.787 + y2 * (-2.295 - 81.98 * y2))
        roll = y * (-4.273 + y2 * (-13.18 + 37.31 * y2))
        rotation = gtsam.Rot3.RzRyRx(roll, 0.0, YAW * DEG * math.sin(2.0 * phase))
        return gtsam.Pose3(rotation, platform0.translation() + gtsam.Point3(0.0, y, dz))

    # The end effector's pose, measured directly (e.g. a tracker on the platform).
    covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    sensor = sensors.PoseSensor(covariance, platform_ee.endEffectorPrior())
    k = 0
    while k * PERIOD <= END_TIME + 1e-9:
        t = k * PERIOD
        sensor.addMeasurement(types.MeasurementBase(0.0, t, platform(t).matrix(), covariance, np.ones((6, 1), dtype=np.int32)))
        k += 1
    estimator.addSensor(sensor)

    return run(estimator, robots, [], [platform_ee], 0.0, END_TIME, PERIOD, args)


if __name__ == "__main__":
    sys.exit(main())
