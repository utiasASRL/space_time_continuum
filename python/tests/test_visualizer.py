"""Visualizer options and registration, and a session running the window (offscreen)."""

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


def test_ground_plane_options():
    plane = visualizer.GroundPlaneOptions()
    np.testing.assert_array_equal(plane.normal, [0, 0, 1])
    np.testing.assert_array_equal(plane.position, [0, 0, 0])
    assert plane.size == 1.0
    plane = visualizer.GroundPlaneOptions(normal=[0, 1, 0], position=[1, 2, 3], size=0.5)
    np.testing.assert_array_equal(plane.normal, [0, 1, 0])
    np.testing.assert_array_equal(plane.position, [1, 2, 3])
    assert plane.size == 0.5
    assert not hasattr(visualizer.VisualizerOptions(), "show_floor")  # the floor is an object one adds


def test_ground_plane_registration_is_checked():
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addGroundPlane()
    vis.addGroundPlane(visualizer.GroundPlaneOptions(normal=[1, 0, 0], position=[0, 0, 0.1], size=0.3))
    for bad in (visualizer.GroundPlaneOptions(normal=[0, 0, 0]), visualizer.GroundPlaneOptions(size=0.0),
                visualizer.GroundPlaneOptions(size=-1.0)):
        with pytest.raises(ValueError):
            vis.addGroundPlane(bad)


def test_robot_options_are_editable_in_place():
    options = visualizer.VisualizerOptions()
    options.robot.radius = 0.025
    assert options.robot.radius == pytest.approx(0.025)
    robot = visualizer.RobotOptions()
    robot.radius = 0.03
    options.robot = robot
    assert options.robot.radius == pytest.approx(0.03)
    assert options.robot.spatial_interpolation_points == 0
    options.robot.spatial_interpolation_points = 4
    assert options.robot.spatial_interpolation_points == 4
    options.estimation_period = 1.0 / 30.0
    assert options.estimation_period == pytest.approx(1.0 / 30.0)


def test_too_many_spatial_interpolation_points_are_rejected():
    options = visualizer.VisualizerOptions()
    options.robot.spatial_interpolation_points = 11
    with pytest.raises(ValueError):
        visualizer.Visualizer(options)


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


BOX_STL = """solid box
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid box
"""


def test_stl_registration_before_start(tmp_path):
    import gtsam
    mesh = tmp_path / "box.stl"
    mesh.write_text(BOX_STL)
    _, prior = make_wnoa_robot(length=0.2)
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(prior)
    vis.addSTL(prior, str(mesh), 0.1)
    vis.addSTL(prior, str(mesh), 0.0, scale=0.001, offset=gtsam.Pose3(gtsam.Rot3(), np.array([0.0, 0.0, 0.01])), color=[1.0, 0.0, 0.0])
    vis.addSTL(prior, str(mesh), 0.2)


def test_stl_arguments_are_checked(tmp_path):
    mesh = tmp_path / "box.stl"
    mesh.write_text(BOX_STL)
    _, prior = make_wnoa_robot(length=0.2)
    _, other = make_wnoa_robot()
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(prior)
    with pytest.raises(ValueError):
        vis.addSTL(other, str(mesh), 0.1)  # robot not added
    for arclength in (-0.01, 0.21, float("nan")):
        with pytest.raises(ValueError):
            vis.addSTL(prior, str(mesh), arclength)
    for scale in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            vis.addSTL(prior, str(mesh), 0.1, scale=scale)
    with pytest.raises(ValueError):
        vis.addSTL(prior, str(tmp_path / "missing.stl"), 0.1)
    garbage = tmp_path / "garbage.stl"
    garbage.write_text("not an stl")
    with pytest.raises(ValueError):
        vis.addSTL(prior, str(garbage), 0.1)


def test_quasi_static_robot_and_stl_registration(tmp_path):
    import gtsam
    from spacetime.priors import wnoa_quasi_static
    mesh = tmp_path / "box.stl"
    mesh.write_text(BOX_STL)
    state = wnoa_quasi_static.ContinuumRobotState(0.2, 3)
    prior = wnoa_quasi_static.Prior(state, wnoa_quasi_static.NodeValues(gtsam.Pose3(), np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])), np.eye(12), np.eye(6))
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(prior)
    assert vis.robotCount() == 1
    vis.addSTL(prior, str(mesh), 0.2, color=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        vis.addSTL(prior, str(mesh), 0.3)


def test_joints_and_end_effectors_of_any_robot_prior_register():
    import gtsam
    from spacetime import joints
    from spacetime.priors import wnoa_quasi_static
    straight = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    state = wnoa_quasi_static.ContinuumRobotState(0.2, 3)
    quasi_static = wnoa_quasi_static.Prior(state, wnoa_quasi_static.NodeValues(gtsam.Pose3(), straight))
    _, wnoa_prior = make_wnoa_robot()
    ee = joints.EndEffector(gtsam.Pose3())
    ee.attach(quasi_static, 0.2, np.eye(6) * 1e-6)
    vis = visualizer.Visualizer(visualizer.VisualizerOptions())
    vis.addPrior(quasi_static)
    vis.addPrior(wnoa_prior)
    vis.addJoint(joints.Joint(quasi_static, 0.2, wnoa_prior, 0.0, np.eye(6) * 1e-6))
    vis.addEndEffector(ee)
    with pytest.raises(ValueError):  # an end effector's joint is drawn by addEndEffector
        vis.addJoint(ee.joints()[0])
    with pytest.raises(TypeError):
        vis.addPrior(ee.endEffectorPrior())


WINDOW_SCRIPT = """
import sys, threading
sys.path.insert(0, sys.argv[2])
import spacetime as st
from spacetime import visualizer
from conftest import batch_estimator, make_wnoa_robot

_, prior = make_wnoa_robot()
estimator = batch_estimator()
estimator.addModel(prior)
options = st.SessionOptions()
options.end_time = 0.5
options.period = 0.25
session = st.Session(options)
window = visualizer.Visualizer(visualizer.VisualizerOptions(), host_mode=getattr(visualizer.HostMode, sys.argv[1]))
window.addPrior(prior)
assert isinstance(window, st.interface.Viewer)
session.start(estimator, window)
try:
    window.addPrior(prior)
    raise AssertionError("adding after Session.start must raise")
except RuntimeError:
    pass
session.joinEstimation()
assert session.finished()
closer = threading.Timer(0.5, window.requestClose)  # as the user closing the window, from another thread
closer.start()
session.join()
closer.cancel()
assert session.viewerExitCode() == 0
print("WINDOW-OK", flush=True)
"""


@pytest.mark.parametrize("host_mode", ["WorkerThread", "CallingThread"])
def test_session_runs_the_window_until_it_is_closed(host_mode, tmp_path):
    """In its own process: one QApplication per process, on the GUI thread the host mode picks (the main thread in CallingThread mode)."""
    import os
    import subprocess
    import sys

    if sys.platform == "darwin" and host_mode == "WorkerThread":
        pytest.skip("macOS only allows a GUI on the main thread")
    script = tmp_path / "window.py"
    script.write_text(WINDOW_SCRIPT)
    tests = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run([sys.executable, str(script), host_mode, tests], capture_output=True, text=True, timeout=120,
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert "WINDOW-OK" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]
    if result.returncode != 0 and host_mode == "WorkerThread":
        # Known, and older than HostMode: Qt destroys its platform plugins (QObjects created on the GUI
        # thread) at process exit, after that worker thread has ended, and reads its freed thread data.
        pytest.xfail(f"Qt's exit-time plugin cleanup after a worker-thread GUI (exit code {result.returncode})")
    assert result.returncode == 0, result.stderr[-2000:]
