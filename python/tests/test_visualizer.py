"""Visualizer options and registration (the window itself is not opened here)."""

import numpy as np
import pytest

visualizer = pytest.importorskip("spacetime.visualizer", reason="built without the visualizer")

from conftest import make_wnoa_robot  # noqa: E402


def test_cameras_are_editable():
    options = visualizer.VisualizerOptions()
    camera = visualizer.CameraOptions()
    camera.position = np.array([1.0, 2.0, 3.0])
    options.cameras = [camera, visualizer.CameraOptions()]
    assert len(options.cameras) == 2
    np.testing.assert_array_equal(options.cameras[0].position, [1, 2, 3])


def test_robot_options_are_editable_in_place():
    options = visualizer.VisualizerOptions()
    options.robot.radius = 0.025
    assert options.robot.radius == pytest.approx(0.025)
    robot = visualizer.RobotOptions()
    robot.radius = 0.03
    options.robot = robot
    assert options.robot.radius == pytest.approx(0.03)
    options.estimation_period = 1.0 / 30.0
    assert options.estimation_period == pytest.approx(1.0 / 30.0)


def test_registration_before_start():
    from spacetime import joints
    _, prior = make_wnoa_robot()
    _, other = make_wnoa_robot()
    joint = joints.Joint(prior, 0.2, other, 0.0, np.eye(6) * 1e-6)
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(prior)
    vis.addPrior(other)
    vis.addJoint(joint)
    assert vis.robotCount() == 2


def test_joint_needs_its_robots_registered_first():
    from spacetime import joints
    _, prior = make_wnoa_robot()
    _, other = make_wnoa_robot()
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(prior)
    with pytest.raises(ValueError):
        vis.addJoint(joints.Joint(prior, 0.2, other, 0.0, np.eye(6) * 1e-6))
