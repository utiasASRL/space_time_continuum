"""The bucky ball stress-test demo (python/examples/bucky_ball.py): its geometry, its zero-error
reference shape, its push profile, and an estimate of the push and release."""

import sys

import gtsam
import numpy as np
import pytest

from conftest import REPO_ROOT, STRAIGHT, run_session

sys.path.insert(0, str(REPO_ROOT / "python" / "examples"))
import bucky_ball  # noqa: E402
from joint_demo import joint_residual, pose_of  # noqa: E402

EDGE = 0.1


@pytest.fixture(scope="module")
def ball():
    return bucky_ball.truncated_icosahedron(EDGE)


def test_truncated_icosahedron_has_60_vertices_and_90_equal_edges(ball):
    vertices, edges, _, _ = ball
    assert vertices.shape == (60, 3)
    assert len(edges) == 90
    assert len(set(map(frozenset, edges))) == 90
    for i, j in edges:
        assert np.linalg.norm(vertices[j] - vertices[i]) == pytest.approx(EDGE, rel=1e-9)
    degree = np.bincount(np.array(edges).ravel(), minlength=60)
    assert np.all(degree == 3)


def test_no_non_edge_pair_is_as_close_as_an_edge(ball):
    vertices, edges, _, _ = ball
    edge_set = set(map(frozenset, edges))
    for i in range(60):
        for j in range(i + 1, 60):
            if frozenset((i, j)) not in edge_set:
                assert np.linalg.norm(vertices[j] - vertices[i]) > 1.1 * EDGE


def assert_hexagonal_face(edges, face):
    assert len(face) == 6 and len(set(face)) == 6
    inner = [(i, j) for i, j in edges if i in face and j in face]
    assert len(inner) == 6  # a single 6-cycle in the edge graph
    assert np.all(np.bincount(np.array(inner).ravel(), minlength=60)[face] == 2)


def test_bottom_hexagon_rests_level_on_the_ground(ball):
    vertices, edges, bottom, _ = ball
    z = vertices[:, 2]
    assert z.min() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(z[bottom], 0.0, atol=1e-12)
    assert np.delete(z, bottom).min() > 1e-3
    assert_hexagonal_face(edges, bottom)


def test_top_hexagon_is_level_at_the_top(ball):
    vertices, edges, bottom, top = ball
    z = vertices[:, 2]
    np.testing.assert_allclose(z[top], z.max(), atol=1e-12)
    assert np.delete(z, top).max() < z.max() - 1e-3
    assert not set(top) & set(bottom)
    assert_hexagonal_face(edges, top)


def test_edge_frames_and_offsets_satisfy_every_joint_exactly(ball):
    vertices, edges, _, _ = ball
    for i, j in edges:
        base, tip = bucky_ball.edge_frames(vertices[i], vertices[j])
        # The tip is where the straight reference strain carries the base.
        assert tip.equals(base.compose(gtsam.Pose3.Expmap(EDGE * STRAIGHT)), 1e-12)
        for end, v in ((base, vertices[i]), (tip, vertices[j])):
            ee = bucky_ball.vertex_pose(v)
            offset = end.between(ee)
            np.testing.assert_allclose(offset.translation(), 0.0, atol=1e-12)
            assert end.compose(offset).equals(ee, 1e-12)


def test_push_ramps_down_holds_then_releases():
    depth, ramp, hold, period = 0.1, 1.0, 1.0, 0.05
    push = bucky_ball.push_profile(depth, ramp, hold, period)
    times = np.array([t for t, _ in push])
    dz = np.array([d for _, d in push])
    np.testing.assert_allclose(np.diff(times), period)
    assert times[0] == 0.0 and dz[0] == 0.0
    # No measurements after the hold: the push is released.
    assert times[-1] == pytest.approx(ramp + hold)
    pressing = times <= ramp + 1e-9
    np.testing.assert_allclose(dz[pressing], -depth * times[pressing] / ramp, atol=1e-12)
    np.testing.assert_allclose(dz[~pressing], -depth, atol=1e-12)


def test_build_joins_every_vertex_to_its_three_edges(ball):
    demo = bucky_ball.build(EDGE, bucky_ball.NODES, 0.1, 0.1, 1.0, 1.0)
    assert len(demo.robots) == 90
    assert len(demo.end_effectors) == 60
    for end_effector in demo.end_effectors:
        assert len(end_effector.joints()) == 3
        for joint in end_effector.joints():
            np.testing.assert_allclose(joint.offset().translation(), 0.0, atol=1e-12)
    assert demo.pushed in ball[3]


def test_push_dents_the_ball_then_it_springs_back(ball):
    vertices, _, bottom, _ = ball
    depth, ramp, hold, period, recovery = 0.1, 1.0, 1.0, 0.2, 2.0
    end_time = ramp + hold + recovery
    demo = bucky_ball.build(EDGE, bucky_ball.NODES, period, depth, ramp, hold)
    run_session(demo.estimator, end_time, period)
    values = demo.estimator.getCurrentValues()

    def z_of(v, t):
        return pose_of(demo.end_effectors[v], 0.0, t, values).translation()[2]

    # The bottom hexagon stays exactly where it was locked.
    for v in bottom:
        for t in np.arange(0.0, end_time + 1e-9, period):
            assert pose_of(demo.end_effectors[v], 0.0, t, values).equals(bucky_ball.vertex_pose(vertices[v]), 1e-9)
    # The pushed vertex follows its measurements, but for the prior rounding the push's corners
    # (at the start, the end of the ramp and the release).
    top_z = vertices[demo.pushed, 2]
    assert z_of(demo.pushed, 0.0) == pytest.approx(top_z, abs=5e-3)
    assert z_of(demo.pushed, ramp + 0.5 * hold) == pytest.approx(top_z - depth, abs=2e-3)
    # Released, it recovers at least half of its dent.
    assert z_of(demo.pushed, end_time) > top_z - 0.5 * depth
    # The joints hold throughout.
    for end_effector in demo.end_effectors:
        for joint in end_effector.joints():
            for t in (ramp, ramp + hold, end_time):
                assert joint_residual(joint, t, values) < 1e-2
