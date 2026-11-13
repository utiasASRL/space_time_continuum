"""A bucky ball pressed from the top: a stress test of joining many robots through end effectors.

The ball is a truncated icosahedron (12 pentagons, 20 hexagons): each of its 90 edges is a
straight continuum robot with 2 nodes (its ends), and each of its 60 vertices an end effector rigidly joined,
with no translation between them, to the ends of its 3 edges. The joint offsets are the rotations
between each edge's end frame and its vertex's frame, so the priors' zero-error solution is the
perfect ball. No edge's base is locked; its unary pose and velocity priors are loose, while its
unary strain prior pulls it back to straight.

The ball rests on the ground (z = 0) with its bottom hexagon's 6 end effectors locked there. One
vertex of its top hexagon is measured, by position only: it is pushed straight down PUSH_DEPTH over
PUSH_RAMP, held there for PUSH_HOLD, then released (no more measurements), and the ball springs
back. Everything is hard-coded; no configuration file.

It is one large batch problem (every vertex couples three edges at every time step), so its size is
kept down: 2 nodes per edge, a 0.1 s period, and a loose LM convergence tolerance.

Usage: python bucky_ball.py [--headless] [--output DIR]
"""

import itertools
import sys
from dataclasses import dataclass, field

import gtsam
import numpy as np

import spacetime as st
from spacetime import joints, sensors, types

from joint_demo import StrainPrior, make_robot, parse_arguments, rotation_with_x_axis, run

PERIOD = 0.1
EDGE_LENGTH = 0.1          # [m] every edge (ball diameter ~0.5 m)
NODES = 2                  # per edge robot
PUSH_DEPTH = 0.1           # [m] how far the top vertex is pushed down
PUSH_RAMP = 1.0            # [s] to push it down
PUSH_HOLD = 1.0            # [s] held down, then released
RECOVERY = 1.0             # [s] after the release
END_TIME = PUSH_RAMP + PUSH_HOLD + RECOVERY
STRETCH_VARIANCE = 1e-2    # each edge's elongation and shear priors, relative to its torsion and bending (identity)
FREE_BASE_VARIANCE = 1e4   # each edge's unary base pose and velocity variances: its base is free to move
JOINT_COVARIANCE = np.eye(6) * 1e-6
POSITION_COVARIANCE = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
POSITION_MASK = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.int32)  # [rot, trans]: position only
STRAIGHT = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
PHI = (1.0 + np.sqrt(5.0)) / 2.0


def truncated_icosahedron(edge_length):
    """The ball's vertices (60 x 3), edges ((i, j) pairs), and bottom and top hexagons (6 vertex
    indices each), oriented with the bottom hexagon level on the ground (z = 0)."""
    # The even (cyclic) permutations of these, with every sign, have edge length 2.
    generators = [(0.0, 1.0, 3.0 * PHI), (1.0, 2.0 + PHI, 2.0 * PHI), (PHI, 2.0, 2.0 * PHI + 1.0)]
    points = set()
    for generator in generators:
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            signed = np.multiply(generator, signs)
            for shift in range(3):
                points.add(tuple(np.roll(signed, shift)))
    vertices = np.array(sorted(points)) * (edge_length / 2.0)

    # A hexagon is centred on (1, 1, 1): turn it to face down (-z).
    down = np.array([0.0, 0.0, -1.0])
    centre = np.ones(3) / np.sqrt(3.0)
    axis = np.cross(centre, down)
    angle = np.arccos(np.dot(centre, down))
    rotation = gtsam.Rot3.AxisAngle(axis / np.linalg.norm(axis), angle).matrix()
    vertices = vertices @ rotation.T
    vertices[:, 2] -= vertices[:, 2].min()

    distances = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
    edges = [(i, j) for i, j in itertools.combinations(range(len(vertices)), 2)
             if abs(distances[i, j] - edge_length) < 1e-6 * edge_length]
    tolerance = 1e-6 * edge_length
    bottom = [int(i) for i in np.flatnonzero(vertices[:, 2] < tolerance)]
    top = [int(i) for i in np.flatnonzero(vertices[:, 2] > vertices[:, 2].max() - tolerance)]
    return vertices, edges, bottom, top


def edge_frames(start, end):
    """The base and tip poses of the straight edge robot from vertex `start` to vertex `end`,
    its backbone along body x."""
    base = gtsam.Pose3(rotation_with_x_axis(end - start), gtsam.Point3(start))
    return base, base.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(np.linalg.norm(end - start), 0.0, 0.0)))


def vertex_pose(vertex):
    """The frame of the end effector at `vertex`: world-aligned."""
    return gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(vertex))


def push_profile(depth, ramp, hold, period):
    """The pushed vertex's measured drop (<= 0) at every `period`, as (time, drop) pairs: pushed
    down `depth` at a constant rate over `ramp`, held for `hold`, then released (no more pairs)."""
    profile = []
    k = 0
    while k * period <= ramp + hold + 1e-9:
        t = k * period
        profile.append((t, -depth * min(t / ramp, 1.0)))
        k += 1
    return profile


@dataclass
class Demo:
    estimator: st.Estimator
    pushed: int = -1  # the vertex pushed down
    robots: list = field(default_factory=list)
    end_effectors: list = field(default_factory=list)


def build(edge_length, nodes, period, depth, ramp, hold):
    """The ball's robots and end effectors, its bottom hexagon locked to the ground and a top
    vertex's position measured as it is pushed, in a batch estimator."""
    vertices, edges, bottom, top = truncated_icosahedron(edge_length)

    estimator_options = st.EstimatorOptions()
    estimator_options.smootherLag = st.Estimator.BATCH_LAG
    estimator_options.compute_covariance = False
    lm_params = estimator_options.lm_params
    lm_params.setRelativeErrorTol(1e-3)  # later iterations change the error by < 1e-3
    estimator_options.lm_params = lm_params
    demo = Demo(st.Estimator(estimator_options), top[0])

    # The edges: straight, stiff in elongation and shear so the ball keeps its size; free bases.
    stretch = StrainPrior.uniform(STRETCH_VARIANCE)
    ends = [[] for _ in vertices]  # per vertex: (robot prior, arclength, end pose)
    for i, j in edges:
        base, tip = edge_frames(vertices[i], vertices[j])
        robot = make_robot(edge_length, nodes, base, STRAIGHT, False, stretch, stretch,
                           unary_pose=FREE_BASE_VARIANCE, unary_velocity=FREE_BASE_VARIANCE)
        demo.robots.append(robot)
        demo.estimator.addPrior(robot.prior)
        ends[i].append((robot.prior, 0.0, base))
        ends[j].append((robot.prior, edge_length, tip))

    # The vertices: each end rigidly joined to its vertex with no translation, only the rotation
    # between the edge's end frame and the vertex's frame. The bottom hexagon is fixed to the ground.
    for v, (vertex, vertex_ends) in enumerate(zip(vertices, ends)):
        pose = vertex_pose(vertex)
        end_effector = joints.EndEffector(pose)
        for prior, s, end_pose in vertex_ends:
            options = joints.JointOptions()
            options.offset = end_pose.between(pose)
            end_effector.attach(prior, s, JOINT_COVARIANCE, options)
        if v in bottom:
            end_effector.state().lock("pose", 0.0, pose)
        demo.estimator.addModel(end_effector)
        demo.end_effectors.append(end_effector)

    # The pushed vertex's position (e.g. a tracked probe pressing on it).
    pushed = demo.end_effectors[demo.pushed]
    sensor = sensors.PoseSensor(POSITION_COVARIANCE, pushed.endEffectorPrior())
    for t, drop in push_profile(depth, ramp, hold, period):
        measured = vertex_pose(vertices[demo.pushed] + np.array([0.0, 0.0, drop]))
        sensor.addMeasurement(types.MeasurementBase(0.0, t, measured.matrix(), POSITION_COVARIANCE, POSITION_MASK))
    demo.estimator.addSensor(sensor)
    return demo


def main():
    args = parse_arguments(__doc__.splitlines()[0])
    demo = build(EDGE_LENGTH, NODES, PERIOD, PUSH_DEPTH, PUSH_RAMP, PUSH_HOLD)
    return run(demo.estimator, demo.robots, [], demo.end_effectors, 0.0, END_TIME, PERIOD, args, summary_only=True)


if __name__ == "__main__":
    sys.exit(main())
