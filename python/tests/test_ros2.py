"""The ROS2 interface: rclcpp init, the ROS2 session, pose sensor nodes and the ROS clock."""

import os
import threading
import time

import gtsam
import numpy as np
import pytest

ros2 = pytest.importorskip("spacetime.interface.ros2", reason="built without the ROS2 interface")

import spacetime as st  # noqa: E402
from spacetime import sensors  # noqa: E402

from conftest import make_wnoa_robot  # noqa: E402

LENGTH = 0.2


@pytest.fixture(scope="module", autouse=True)
def rclcpp_context():
    ros2.init([])
    yield
    ros2.shutdown()


def unique(prefix):
    unique.counter = getattr(unique, "counter", 0) + 1
    return f"{prefix}_{os.getpid()}_{unique.counter}"


def sliding_estimator(smoother_lag=0.3):
    options = st.EstimatorOptions()
    options.smootherLag = smoother_lag
    return st.Estimator(options)


def ros_options():
    options = ros2.SessionRosOptions()
    options.node_prefix = unique("spacetime")
    return options


def session_options(end_time, period=0.02):
    options = st.interface.SessionOptions()
    options.end_time = end_time
    options.period = period
    options.real_time = True
    return options


def test_ok_after_init():
    assert ros2.ok()


def test_pose_sensor_options_are_editable():
    options = ros2.PoseSensorOptions()
    options.offset = gtsam.Pose3(gtsam.Rot3.Rz(0.5), np.array([0.0, 0.0, 0.01]))
    options.covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    options.mask = np.array([1, 1, 0, 1, 1, 1])
    assert options.offset.z() == pytest.approx(0.01)
    assert options.covariance[5, 5] == pytest.approx(1e-6)
    np.testing.assert_array_equal(options.mask, [1, 1, 0, 1, 1, 1])


def test_body_pose_applies_the_offset():
    T_is = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, 0.3), np.array([0.1, 0.2, 0.3]))
    T_bs = gtsam.Pose3(gtsam.Rot3.Ry(0.4), np.array([0.0, 0.01, 0.02]))
    T_ib = ros2.PoseSensorNode.bodyPose(T_is, T_bs)
    assert T_ib.compose(T_bs).equals(T_is, 1e-12)


def test_batch_estimator_cannot_start():
    _, prior = make_wnoa_robot(lock_base=False)
    options = st.EstimatorOptions()
    options.smootherLag = st.Estimator.BATCH_LAG
    estimator = st.Estimator(options)
    estimator.addModel(prior)
    session = ros2.Session(estimator, session_options(-1.0), ros_options())
    with pytest.raises(ValueError):
        session.start()


def test_session_estimates_every_period_until_the_end_time():
    _, prior = make_wnoa_robot(lock_base=False)
    estimator = sliding_estimator()
    estimator.addModel(prior)
    session = ros2.Session(estimator, session_options(0.2), ros_options())
    times = []
    session.setTickCallback(lambda t: times.append(t))

    session.start()
    session.join()

    assert session.finished()
    assert len(times) >= 3
    # A period apart (never a near-duplicate row, e.g. one clamped to the end time), up to the end time.
    assert all(b - a > 0.5 * 0.02 for a, b in zip(times, times[1:]))
    assert 0.2 - 2 * 0.02 < times[-1] <= 0.2
    assert estimator.getSnapshot().time == pytest.approx(times[-1])
    assert session.clock().started()


def test_request_stop_ends_a_live_run():
    _, prior = make_wnoa_robot(lock_base=False)
    estimator = sliding_estimator()
    estimator.addModel(prior)
    session = ros2.Session(estimator, session_options(-1.0), ros_options())
    ticked = threading.Event()
    session.setTickCallback(lambda t: ticked.set())

    session.start()
    assert ticked.wait(2.0)
    assert session.now() > 0.0
    session.requestStop()
    session.join()
    assert session.finished()


def test_join_raises_an_error_from_the_tick_callback():
    _, prior = make_wnoa_robot(lock_base=False)
    estimator = sliding_estimator()
    estimator.addModel(prior)
    session = ros2.Session(estimator, session_options(-1.0), ros_options())

    def fail(_):
        raise RuntimeError("tick failed")

    session.setTickCallback(fail)
    session.start()
    with pytest.raises(RuntimeError, match="tick failed"):
        session.join()


def test_unfed_queues_are_reported():
    _, prior = make_wnoa_robot(lock_base=False)
    estimator = sliding_estimator()
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    estimator.addSensor(sensor)
    session = ros2.Session(estimator, session_options(-1.0), ros_options())
    session.addMeasurementQueues(estimator.getMeasurementQueues())
    assert len(session.unfedQueues()) == 1

    options = ros2.PoseSensorOptions()
    options.parent_frame = unique("field")
    options.child_frame = unique("sensor")
    options.topic = "/" + unique("tf")
    node = session.addPoseSensor(sensor, options)
    assert session.unfedQueues() == []
    assert node.measurementsAdded() == 0
    assert node.options().child_frame == options.child_frame


def test_invalid_pose_sensor_options_raise():
    _, prior = make_wnoa_robot(lock_base=False)
    estimator = sliding_estimator()
    estimator.addModel(prior)
    sensor = sensors.PoseSensor(prior)
    session = ros2.Session(estimator, session_options(-1.0), ros_options())
    with pytest.raises(ValueError):
        session.addPoseSensor(sensor, ros2.PoseSensorOptions())  # no frames


def test_wait_for_transform_times_out():
    start = time.monotonic()
    assert ros2.waitForTransform(unique("field"), unique("sensor"), 0.2) is None
    assert time.monotonic() - start >= 0.2


# The rest drive /tf from Python with rclpy, side by side with the library's rclcpp nodes.


@pytest.fixture(scope="module")
def rclpy_node():
    rclpy = pytest.importorskip("rclpy", reason="rclpy is not available (source ROS2)")
    rclpy.init()
    node = rclpy.create_node(unique("spacetime_test"))
    yield node
    node.destroy_node()
    rclpy.shutdown()


def transform(parent, child, pose, stamp):
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = parent
    t.header.stamp = stamp
    t.child_frame_id = child
    q = pose.rotation().toQuaternion()
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pose.translation()
    t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = q.w(), q.x(), q.y(), q.z()
    return t


def test_wait_for_transform_returns_a_published_transform(rclpy_node):
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

    parent, child = unique("field"), unique("sensor")
    pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, 0.3), np.array([0.4, 0.5, 0.6]))
    broadcaster = StaticTransformBroadcaster(rclpy_node)
    broadcaster.sendTransform(transform(parent, child, pose, rclpy_node.get_clock().now().to_msg()))

    found = ros2.waitForTransform(parent, child, 3.0)

    assert found is not None
    assert found.equals(pose, 1e-9)


def test_estimates_a_robot_from_transforms(rclpy_node):
    from tf2_msgs.msg import TFMessage

    _, prior = make_wnoa_robot(length=LENGTH, lock_base=False)
    estimator = sliding_estimator()
    estimator.addModel(prior)
    covariance = np.eye(6) * 1e-6
    base_sensor = sensors.PoseSensor(prior)
    tip_sensor = sensors.PoseSensor(prior)
    estimator.addSensor(base_sensor)
    estimator.addSensor(tip_sensor)

    session = ros2.Session(estimator, session_options(1.0), ros_options())
    session.addMeasurementQueues(estimator.getMeasurementQueues())
    topic, field = "/" + unique("tf"), unique("field")
    base_options = ros2.PoseSensorOptions()
    base_options.parent_frame = field
    base_options.child_frame = unique("base")
    base_options.topic = topic
    base_options.covariance = covariance
    base_node = session.addPoseSensor(base_sensor, base_options)
    tip_options = ros2.PoseSensorOptions()
    tip_options.parent_frame = field
    tip_options.child_frame = unique("tip")
    tip_options.topic = topic
    tip_options.covariance = covariance
    tip_options.arclength = LENGTH
    session.addPoseSensor(tip_sensor, tip_options)

    # The true robot: straight along its x (the conftest strain), based at T_base.
    T_base = gtsam.Pose3(gtsam.Rot3.Rx(0.3), np.array([0.05, -0.02, 0.01]))
    T_tip = T_base.compose(gtsam.Pose3(gtsam.Rot3(), np.array([LENGTH, 0.0, 0.0])))
    publisher = rclpy_node.create_publisher(TFMessage, topic, 100)
    stop = threading.Event()

    def broadcast():
        while not stop.is_set():
            stamp = rclpy_node.get_clock().now().to_msg()
            message = TFMessage()
            message.transforms = [transform(field, base_options.child_frame, T_base, stamp),
                                  transform(field, tip_options.child_frame, T_tip, stamp)]
            publisher.publish(message)
            time.sleep(0.02)

    broadcaster = threading.Thread(target=broadcast)
    broadcaster.start()
    try:
        session.start()
        session.join()
    finally:
        stop.set()
        broadcaster.join()
        rclpy_node.destroy_publisher(publisher)

    assert base_node.measurementsAdded() > 10
    snapshot = estimator.getSnapshot()
    assert snapshot is not None
    base = prior.query(snapshot.time, 0.0, snapshot.values)
    tip = prior.query(snapshot.time, LENGTH, snapshot.values)
    assert np.linalg.norm(base.pose().translation() - T_base.translation()) < 1e-3
    assert np.linalg.norm(tip.pose().translation() - T_tip.translation()) < 1e-3
