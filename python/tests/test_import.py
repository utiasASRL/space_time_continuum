"""The extension imports, and GTSAM objects cross into it as GTSAM's own Python types."""

import gtsam
import pytest

import spacetime as st
from spacetime.priors import wnoa


def test_gtsam_pose_round_trips_as_a_gtsam_type():
    pose = gtsam.Pose3(gtsam.Rot3.Ry(0.3), gtsam.Point3(1.0, 2.0, 3.0))
    values = wnoa.Values()
    values.pose(pose)
    out = values.pose()
    assert isinstance(out, gtsam.Pose3)
    assert out.equals(pose, 1e-12)


def test_estimator_holds_a_gtsam_fixed_lag_smoother():
    estimator = st.Estimator(st.EstimatorOptions())
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
