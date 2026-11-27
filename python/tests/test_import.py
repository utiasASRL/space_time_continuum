"""The extension imports, and GTSAM objects cross into it as GTSAM's own Python types."""

import ctypes
import os
import re
import sys

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


def loaded_libraries():
    """The paths of the shared libraries loaded into this process, or None where unsupported."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/maps") as maps:
            return {line.split()[-1] for line in maps if "/" in line}
    if sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        return {dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())}
    if sys.platform == "win32":
        from ctypes import wintypes
        psapi, kernel32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        modules = (wintypes.HMODULE * 4096)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            return None
        paths = set()
        name = ctypes.create_unicode_buffer(32768)
        for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            if psapi.GetModuleFileNameExW(process, wintypes.HMODULE(module), name, len(name)):
                paths.add(name.value)
        return paths
    return None


# libgtsam's file name on each platform, as built or as a wheel repair tool renames it
# (auditwheel: libgtsam-<hash>.so.<version>; delvewheel: gtsam-<hash>.dll).
LIBGTSAM = re.compile(r"^(libgtsam(-[0-9a-f]+)?\.so|libgtsam[\w.-]*\.dylib$|gtsam(-[0-9a-f]+)?\.dll$)", re.IGNORECASE)


def test_one_libgtsam_is_loaded():
    """GTSAM's module and ours must share one libgtsam, or GTSAM objects would cross library copies."""
    libraries = loaded_libraries()
    if libraries is None:
        pytest.skip(f"listing the loaded libraries is not supported on {sys.platform}")
    libgtsam = {path for path in libraries if LIBGTSAM.match(os.path.basename(path))}
    if not libgtsam:
        pytest.skip("libgtsam is linked statically")
    assert len(libgtsam) == 1, libgtsam


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
