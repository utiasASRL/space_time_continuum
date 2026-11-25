"""The extension imports, and GTSAM objects cross into it as GTSAM's own Python types."""

import gtsam
import pytest

import spacetime as st
from spacetime.priors import wnoa

from conftest import estimator_options


def test_gtsam_pose_round_trips_as_a_gtsam_type():
    pose = gtsam.Pose3(gtsam.Rot3.Ry(0.3), gtsam.Point3(1.0, 2.0, 3.0))
    values = wnoa.NodeValues()
    values.pose(pose)
    out = values.pose()
    assert isinstance(out, gtsam.Pose3)
    assert out.equals(pose, 1e-12)


def test_estimator_holds_a_gtsam_fixed_lag_smoother():
    estimator = st.Estimator(estimator_options(0.0))
    assert not isinstance(estimator, gtsam.FixedLagSmoother)
    assert isinstance(estimator.smoother(), gtsam.BatchFixedLagSmoother)
    assert isinstance(estimator.getCurrentValues(), gtsam.Values)


def test_one_libgtsam_is_loaded():
    """GTSAM's module and ours must share one libgtsam, or GTSAM objects would cross library copies."""
    with open("/proc/self/maps") as maps:
        libraries = {line.split()[-1] for line in maps if "libgtsam.so" in line}
    if not libraries:
        pytest.skip("libgtsam is linked statically")
    assert len(libraries) == 1, libraries


@pytest.fixture
def restore_interface():
    yield
    st.setInterface("default")


def test_session_is_the_default_interface_unless_selected():
    assert st.getInterface() == "default"
    assert st._sessionClass(st.getInterface()) is st.interface.Session
    assert st.SessionOptions is st.interface.SessionOptions
    session = st.Session(st.SessionOptions())
    assert isinstance(session, st.interface.Session)


def test_star_import_session_follows_set_interface(restore_interface):
    namespace = {}
    exec("from spacetime import *", namespace)
    assert namespace["Session"] is st.Session
    assert namespace["SessionOptions"] is st.SessionOptions
    for name in ("Estimator", "priors", "sensors", "actuators", "types"):
        assert name in namespace, name
    session = namespace["Session"](namespace["SessionOptions"]())
    assert isinstance(session, st.interface.Session)
    if hasattr(st.interface, "ros2"):
        st.setInterface("ros2")
        assert st._sessionClass(st.getInterface()) is st.interface.ros2.Session


def test_set_interface_selects_the_ros2_session(restore_interface):
    if not hasattr(st.interface, "ros2"):
        with pytest.raises(ImportError):
            st.setInterface("ros2")
        assert st.getInterface() == "default"
        pytest.skip("built without the ROS2 interface")
    st.setInterface("ros2")
    assert st.getInterface() == "ros2"
    assert st._sessionClass(st.getInterface()) is st.interface.ros2.Session
    st.setInterface("default")
    assert st._sessionClass(st.getInterface()) is st.interface.Session


def test_set_interface_rejects_mujoco_and_unknown_names(restore_interface):
    with pytest.raises(NotImplementedError):
        st.setInterface("mujoco")
    with pytest.raises(ValueError):
        st.setInterface("standalone")
    assert st.getInterface() == "default"
