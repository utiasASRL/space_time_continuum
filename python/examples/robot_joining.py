"""Two continuum robots with offset bases, rigidly joined from the tip of robot A to the mid-point
of robot B through a short (3 cm) link: the Python port of examples/robot_joining.cpp.

Robot B, whose tip is free, is measured at its mid-point and its tip as it is driven into a
growing "S" shape, one way and then the other, sideways (orthogonal to robot A's bending plane);
the joint drags robot A along. Robot A's prior is stiff in elongation, so it bends rather than
stretches to follow.

Geometry (world z up): robot A stands at the origin and bends over by a quarter turn, so its tip
points along +x. Robot B stands upright further along +x, its mid-point 3 cm past A's tip along
A's tip tangent. Everything is hard-coded; no configuration file.

Usage: python robot_joining.py [--headless] [--output DIR]
"""

import math
import sys

import gtsam
import numpy as np

import spacetime as st
from spacetime import joints, sensors

from joint_demo import StrainPrior, make_robot, measure, parse_arguments, run

PERIOD = 0.1
END_TIME = 8.0
LENGTH_A = 0.4
NODES = 5
LINK = 0.03
MAX_S_CURVATURE = 3.0       # [rad/m] of each half of robot B's S
ELONGATION_VARIANCE = 1e-4  # robot A's elongation prior, relative to its other strains (identity)


def strain(curvature):
    """Bending in the world x-z plane (about body y)."""
    return np.array([0.0, curvature, 0.0, 1.0, 0.0, 0.0])


def sideways_strain(curvature):
    """Bending in the world y-z plane (about body z)."""
    return np.array([0.0, 0.0, curvature, 1.0, 0.0, 0.0])


def main():
    args = parse_arguments(__doc__.splitlines()[0])
    upright = gtsam.Pose3(gtsam.Rot3.Ry(-math.pi / 2), gtsam.Point3(0, 0, 0))  # body x along world z

    # Robot A: a quarter circle in the x-z plane, from upright to pointing along +x.
    curvature_a = (math.pi / 2) / LENGTH_A
    tip_a = upright.compose(gtsam.Pose3.Expmap(LENGTH_A * strain(curvature_a)))

    # Robot B: upright and straight, its mid-point LINK past A's tip along A's tip tangent.
    joint_point = tip_a.transformFrom(gtsam.Point3(LINK, 0, 0))
    length_b, mid_b = 2.0 * joint_point[2], joint_point[2]
    base_b = gtsam.Pose3(upright.rotation(), gtsam.Point3(joint_point[0], joint_point[1], 0.0))

    # Robot B's S shape, in the world y-z plane: curvature +k over B's lower half and -k over
    # its upper half, so its tip ends up parallel to its base. k sweeps 0 -> +max -> -max -> 0.
    def s_curvature(t):
        return MAX_S_CURVATURE * math.sin(2.0 * math.pi * t / END_TIME)

    def pose_b(s, t):
        k = s_curvature(t)
        if s <= mid_b:
            return base_b.compose(gtsam.Pose3.Expmap(s * sideways_strain(k)))
        return base_b.compose(gtsam.Pose3.Expmap(mid_b * sideways_strain(k))).compose(
            gtsam.Pose3.Expmap((s - mid_b) * sideways_strain(-k)))

    robots = [make_robot(LENGTH_A, NODES, upright, strain(curvature_a), True, StrainPrior.uniform(ELONGATION_VARIANCE)),
              make_robot(length_b, NODES, base_b, strain(0.0))]

    estimator_options = st.EstimatorOptions()
    estimator_options.smootherLag = st.Estimator.BATCH_LAG
    estimator_options.compute_covariance = True
    estimator = st.Estimator(estimator_options)
    for robot in robots:
        estimator.addPrior(robot.prior)

    # The rigid link from A's tip to B's mid-point, at their nominal relative pose, held to 1 mm / 1 mrad.
    options = joints.JointOptions()
    options.offset = tip_a.between(pose_b(mid_b, 0.0))
    robot_joints = [joints.Joint(robots[0].prior, LENGTH_A, robots[1].prior, mid_b, np.eye(6) * 1e-6, options)]
    for joint in robot_joints:
        estimator.addModel(joint)

    # Robot B's mid-point and tip, measured as it goes through the S.
    covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    sensor = sensors.PoseSensor(covariance, robots[1].prior)
    k = 0
    while k * PERIOD <= END_TIME + 1e-9:
        t = k * PERIOD
        for s in (mid_b, length_b):
            measure(sensor, s, t, pose_b(s, t), covariance)
        k += 1
    estimator.addSensor(sensor)

    return run(estimator, robots, robot_joints, [], 0.0, END_TIME, PERIOD, args)


if __name__ == "__main__":
    sys.exit(main())
