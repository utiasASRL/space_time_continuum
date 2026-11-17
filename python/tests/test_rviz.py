"""The rviz snapshot publisher (markers, backbone poses, covariance ellipsoids, joints)."""

import os
import time

import gtsam
import numpy as np
import pytest

rviz = pytest.importorskip("spacetime.visualizer.rviz", reason="built without the ROS2 interface")
ros2 = pytest.importorskip("spacetime.interface.ros2", reason="built without the ROS2 interface")
rclpy = pytest.importorskip("rclpy", reason="rclpy is not available (source ROS2)")

import spacetime as st  # noqa: E402
from spacetime import joints  # noqa: E402

from conftest import make_wnoa_robot  # noqa: E402

NODES = 4


def unique(prefix):
    unique.counter = getattr(unique, "counter", 0) + 1
    return f"{prefix}_{os.getpid()}_{unique.counter}"


@pytest.fixture(scope="module", autouse=True)
def contexts():
    ros2.init([])
    rclpy.init()
    yield
    rclpy.shutdown()
    ros2.shutdown()


@pytest.fixture
def listener():
    node = rclpy.create_node(unique("rviz_listener"))
    yield node
    node.destroy_node()


def solved_session(compute_covariance=False, extra_robot=False):
    """An estimator solved live for a short while, its session (for the clock) and robot priors."""
    options = st.EstimatorOptions()
    options.smootherLag = 1.0
    options.compute_covariance = compute_covariance
    estimator = st.Estimator(options)
    _, prior = make_wnoa_robot(length=1.0, nodes=NODES, lock_base=False)
    estimator.addModel(prior)
    priors = [prior]
    if extra_robot:
        _, other = make_wnoa_robot(length=0.5, nodes=NODES, base=gtsam.Pose3(gtsam.Rot3(), np.array([0.0, 0.2, 0.0])), lock_base=False)
        estimator.addModel(other)
        priors.append(other)
    session_options = st.interface.SessionOptions()
    session_options.end_time = 0.1
    session_options.period = 0.05
    session_options.real_time = True
    ros_options = ros2.SessionRosOptions()
    ros_options.node_prefix = unique("spacetime")
    session = ros2.Session(estimator, session_options, ros_options)
    session.start()
    session.join()
    return estimator, session, priors


def publisher_options():
    options = rviz.SnapshotPublisherOptions()
    options.rate = 0.0  # published by hand
    options.topic_prefix = unique("spacetime")
    options.tf_topic = "/" + unique("tf")
    return options


def receive(listener, publisher, options, robot="a"):
    """publishLatest(), then the MarkerArray and PoseArray it sent."""
    from geometry_msgs.msg import PoseArray
    from visualization_msgs.msg import MarkerArray

    received = {}
    listener.create_subscription(MarkerArray, options.topic_prefix + "/markers", lambda m: received.setdefault("markers", m), 10)
    listener.create_subscription(PoseArray, f"{options.topic_prefix}/{robot}/poses", lambda m: received.setdefault("poses", m), 10)
    deadline = time.monotonic() + 3.0
    while listener.count_publishers(options.topic_prefix + "/markers") == 0 and time.monotonic() < deadline:
        rclpy.spin_once(listener, timeout_sec=0.01)
    time.sleep(0.2)  # let the publisher see the subscription too
    assert publisher.publishLatest()
    while len(received) < 2 and time.monotonic() < deadline:
        rclpy.spin_once(listener, timeout_sec=0.01)
    assert len(received) == 2
    return received["markers"], received["poses"]


def test_nothing_is_published_twice():
    estimator, session, (prior,) = solved_session()
    publisher = rviz.SnapshotPublisher(estimator, session.clock(), publisher_options())
    publisher.addRobot("a", prior)
    assert publisher.publishLatest()
    assert not publisher.publishLatest()  # the same snapshot


def test_publishes_the_robot(listener):
    estimator, session, (prior,) = solved_session()
    options = publisher_options()
    publisher = rviz.SnapshotPublisher(estimator, session.clock(), options)
    display = rviz.RobotDisplayOptions()
    display.frame_id = "aurora_field"
    display.extra_points = 2
    publisher.addRobot("a", prior, display)

    markers, poses = receive(listener, publisher, options)

    points = NODES + (NODES - 1) * display.extra_points
    assert poses.header.frame_id == "aurora_field"
    assert len(poses.poses) == points
    tube = [m for m in markers.markers if m.ns == "a"]
    assert len(tube) == points - 1
    assert sum(m.scale.z for m in tube) == pytest.approx(1.0, abs=1e-3)


def test_draws_joints(listener):
    estimator, session, (prior_a, prior_b) = solved_session(extra_robot=True)
    options = publisher_options()
    publisher = rviz.SnapshotPublisher(estimator, session.clock(), options)
    publisher.addRobot("a", prior_a)
    publisher.addRobot("b", prior_b)
    joint = joints.RobotJoint(prior_a, 0.0, prior_b, 0.0, np.eye(6) * 1e-6)
    publisher.addJoint("ball", joint, 0.01)

    markers, _ = receive(listener, publisher, options)

    ball = [m for m in markers.markers if m.ns == "ball"]
    assert len(ball) == 1
    assert ball[0].scale.x == pytest.approx(0.02)


def test_invalid_robots_and_joints_raise():
    estimator, session, (prior_a, prior_b) = solved_session(extra_robot=True)
    publisher = rviz.SnapshotPublisher(estimator, session.clock(), publisher_options())
    publisher.addRobot("a", prior_a)
    with pytest.raises(ValueError):
        publisher.addRobot("a", prior_b)  # name taken
    with pytest.raises(ValueError):
        publisher.addRobot("bad name!", prior_b)
    joint = joints.RobotJoint(prior_a, 0.0, prior_b, 0.0, np.eye(6) * 1e-6)
    with pytest.raises(ValueError):
        publisher.addJoint("joint", joint)  # robot b not added


def test_runs_on_a_session():
    estimator = st.Estimator(st.EstimatorOptions())
    _, prior = make_wnoa_robot(lock_base=False)
    estimator.addModel(prior)
    session_options = st.interface.SessionOptions()
    session_options.end_time = 0.2
    session_options.period = 0.02
    session_options.real_time = True
    ros_options = ros2.SessionRosOptions()
    ros_options.node_prefix = unique("spacetime")
    session = ros2.Session(estimator, session_options, ros_options)
    options = publisher_options()
    options.rate = 50.0
    publisher = rviz.SnapshotPublisher(estimator, session.clock(), options)
    publisher.addRobot("a", prior)
    session.addNode(publisher)

    session.start()
    session.join()

    assert session.finished()
    assert estimator.getSnapshot() is not None
