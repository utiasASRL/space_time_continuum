"""Shared plumbing of the multi-robot joint demos (robot_joining.py, robot_joining_parallel.py,
stewart_gough.py), a port of examples/JointDemo.hpp: building upright WNOA robots with locked
bases, running the estimator through the standalone interface, drawing every robot and joint,
and reporting how well each joint holds.
"""

import argparse
import os
import time
from dataclasses import dataclass

import gtsam
import numpy as np

import spacetime as st
from spacetime import joints, types
from spacetime.priors import wnoa

try:
    from spacetime import visualizer
except ImportError:  # built without the visualizer
    visualizer = None


@dataclass
class DemoRobot:
    """A WNOA robot's grid state, its prior, and its initial base pose."""
    state: wnoa.ContinuumRobotState
    prior: wnoa.Prior
    base: gtsam.Pose3


@dataclass
class StrainPrior:
    """How a robot's prior treats a group of its strains -- elongation (translational x), shear
    (translational y and z) or bending (rotational y and z), in GTSAM [rot, trans] order -- as
    scales of the corresponding identity variances."""
    unary: float = 1.0      # in the base unary factor (Q0): large lets the base strain be set by measurements
    arclength: float = 1.0  # in the arclength PSD (Q2): small keeps the strain uniform along the backbone
    time: float = 1.0       # in the strain's time PSD (Q3)

    @staticmethod
    def uniform(variance):
        """The same scale everywhere."""
        return StrainPrior(variance, variance, variance)


def make_robot(length, nodes, base, strain, lock_base=True, elongation=None, shear=None, bending=None,
               unary_pose=1.0, unary_velocity=1.0):
    """A robot of `length` with `nodes` evenly spaced nodes, its base at `base` (T_ib, the backbone
    along the body x axis) -- locked there unless `lock_base` is false -- and whose prior mean has
    the constant strain `strain` ([rot, trans]; (0, 0, 0, 1, 0, 0) is straight). `unary_pose` and
    `unary_velocity` scale the base's unary pose and velocity variances (Q0): large lets a free
    base move."""
    Q0, Q2, Q3 = np.eye(18), np.eye(6), np.eye(6)
    Q0[:6, :6] *= unary_pose
    Q0[12:, 12:] *= unary_velocity
    # Rotational y and z (bending), translational x (elongation), y and z (shear), in GTSAM [rot, trans] order.
    scaled = [(1, bending), (2, bending), (3, elongation), (4, shear), (5, shear)]
    for i, scale in scaled:
        scale = scale or StrainPrior()
        Q0[6 + i, 6 + i] = scale.unary  # [pose, strain, velocity] blocks
        Q2[i, i] = scale.arclength
        Q3[i, i] = scale.time

    state = wnoa.ContinuumRobotState(length, nodes)
    reference = wnoa.NodeValues()
    reference.pose(base)
    reference.epsilon(strain)
    prior = wnoa.Prior(state, reference, Q0, np.eye(6), Q2, Q3, np.eye(6))
    if lock_base:
        state.lock("pose", 0.0, base)
    return DemoRobot(state, prior, base)


def rotation_with_x_axis(x):
    """Rotation whose x axis is `x` (normalized); y is horizontal where possible."""
    ex = np.asarray(x, dtype=float) / np.linalg.norm(x)
    ey = np.cross([0.0, 0.0, 1.0], ex)
    if np.linalg.norm(ey) < 1e-9:
        ey = np.array([0.0, 1.0, 0.0])
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return gtsam.Rot3(np.column_stack([ex, ey, ez]))


def measure(sensor, s, t, pose, covariance):
    """Adds a 6-DoF pose measurement at arclength `s` and time `t`, with covariance `covariance` ([rot, trans])."""
    sensor.addMeasurement(types.MeasurementBase(s, t, pose.matrix(), covariance, np.ones((6, 1), dtype=np.int32)))


def pose_of(prior, s, t, values):
    """The estimated pose at arclength `s` and time `t` of the robot (WNOA) or end effector of `prior`."""
    return prior.query(t, s, values).pose()


def joint_residual(joint, t, values):
    """How far a joint is from holding at `t`: |Log(T_off^-1 T_A^-1 T_B)| ([rad, m] mixed)."""
    T_a = pose_of(joint.priorA(), joint.arclengthA(), t, values)
    T_b = pose_of(joint.priorB(), joint.arclengthB(), t, values)
    return np.linalg.norm(gtsam.Pose3.Logmap(joint.offset().between(T_a.between(T_b))))


def travel(prior, s, start_time, end_time, period, values):
    """Furthest the point at arclength `s` of the robot of `prior` moves from its start_time position, sampled every `period`."""
    start = pose_of(prior, s, start_time, values).translation()
    furthest = 0.0
    t = start_time
    while t <= end_time + 1e-9:
        furthest = max(furthest, np.linalg.norm(pose_of(prior, s, t, values).translation() - start))
        t += period
    return furthest


def parse_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true", help="estimate without opening the visualizer")
    parser.add_argument("--output", metavar="DIR", help="write the estimate to DIR for regression testing")
    return parser.parse_args()


def _format(value):
    """`value` as C++'s std::setprecision(17) writes it, so outputs compare as text too."""
    return f"{value:.17g}"


def write_output(directory, values, reports):
    """Writes estimates.csv (every entry of every estimated variable, a pose as its row-major 4x4
    matrix) and joints.csv (each joint's residual and travels) to `directory`, as JointDemo.hpp does."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "estimates.csv"), "w") as estimates:
        estimates.write("key,value\n")
        for key in values.keys():
            symbol = gtsam.Symbol(key)
            # The robots' and end effectors' variables: poses ('p') and 6-vectors (strains, velocities).
            entries = values.atPose3(key).matrix().ravel() if chr(symbol.chr()) == "p" else values.atVector6(key)
            for i, entry in enumerate(entries):
                estimates.write(f"{symbol.string()}[{i}],{_format(entry)}\n")
    with open(os.path.join(directory, "joints.csv"), "w") as joints_file:
        joints_file.write("joint,residual,travel_a,travel_b\n")
        for joint_id, residual, travel_a, travel_b in reports:
            joints_file.write(f"{joint_id},{_format(residual)},{_format(travel_a)},{_format(travel_b)}\n")


def ground_plane_under(robots):
    """The floor the robots stand on: a horizontal plane (normal +z) at z = 0, centred under the
    robots' bases and sized to their horizontal spread, with a margin (as JointDemo::groundPlaneUnder)."""
    plane = visualizer.GroundPlaneOptions()
    if not robots:
        return plane
    bases = np.array([robot.base.translation()[:2] for robot in robots])
    centre = bases.mean(axis=0)
    spread = float(np.max(np.linalg.norm(bases - centre, axis=1)))
    plane.position = np.array([centre[0], centre[1], 0.0])
    plane.size = max(0.3, 3.0 * spread)
    return plane


def run(estimator, robots, robot_joints, end_effectors, start_time, end_time, period, args, summary_only=False):
    """Estimates [start_time, end_time] with a batch `estimator` already holding every robot,
    joint, end effector and sensor, then (unless headless) replays it in the visualizer, and
    prints each joint's residual at the end time -- or, if `summary_only`, just their largest and
    mean residuals and travels."""
    options = st.SessionOptions()
    options.start_time = start_time
    options.end_time = end_time
    options.period = period
    options.real_time = False
    session = st.Session(options)

    initial_wall_time = time.perf_counter()
    window = None
    if not args.headless and visualizer is not None:
        vis_options = visualizer.VisualizerOptions()
        vis_options.online = False
        vis_options.estimator_window = st.Estimator.BATCH_LAG if estimator.isBatch() else estimator.smootherLag()
        vis_options.estimation_period = period  # arrow keys and V step through the estimation ticks
        window = visualizer.Visualizer(vis_options)
        for demo_robot in robots:
            window.addPrior(demo_robot.prior)
        for joint in robot_joints:
            window.addJoint(joint)
        for end_effector in end_effectors:
            window.addEndEffector(end_effector)
        window.addGroundPlane(ground_plane_under(robots))  # toggle P

    # Offline: estimate everything first (reported below), then replay it in the window, if
    # any (session.join()). Closing the window stops the session.
    session.start(estimator, window)
    session.joinEstimation()
    print(f"Estimation complete in {time.perf_counter() - initial_wall_time} s.")

    last_time = session.endTime()
    values = estimator.getCurrentValues()
    reports = []

    def report(name, joint_id, joint):
        residual = joint_residual(joint, last_time, values)
        travel_a = travel(joint.priorA(), joint.arclengthA(), start_time, last_time, period, values)
        travel_b = travel(joint.priorB(), joint.arclengthB(), start_time, last_time, period, values)
        if not summary_only:
            print(f"{name}: residual at t = {last_time:g} is {residual:g}; its point on robot A travels up to "
                  f"{travel_a:g} m, and on side B up to {travel_b:g} m.")
        reports.append((joint_id, residual, travel_a, travel_b))

    for i, joint in enumerate(robot_joints):
        report(f"Joint {i}", str(i), joint)
    for e, end_effector in enumerate(end_effectors):
        for i, joint in enumerate(end_effector.joints()):
            report(f"End effector {e}, joint {i}", f"ee{e}.{i}", joint)
    if summary_only and reports:
        residuals = np.array([r[1] for r in reports])
        travels = np.array([r[2:] for r in reports])
        print(f"{len(reports)} joints at t = {last_time:g}: residual max {residuals.max():g}, mean {residuals.mean():g}; "
              f"their points travel {travels.min():g} to {travels.max():g} m.")
    if args.output:
        write_output(args.output, values, reports)

    session.join()  # shows the window, if any, until it is closed
    return session.viewerExitCode()
