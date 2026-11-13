"""Configuration loading, window history and ground-truth export."""

import json

import gtsam
import numpy as np
import pytest

import spacetime as st
from spacetime import utilities
from spacetime.priors import wnoa, wnoa_quasi_static, wnov

from conftest import REPO_ROOT, batch_estimator, make_wnoa_robot, run_session

DEFAULT_CONFIG = REPO_ROOT / "assets" / "config" / "default_config.json"


@pytest.fixture
def loader():
    loader = utilities.ConfigLoader()
    assert loader.loadConfig(str(DEFAULT_CONFIG))
    return loader


def test_missing_config_returns_false():
    assert utilities.ConfigLoader().loadConfig("/nonexistent/config.json") is False


def test_estimator_options_from_config(loader):
    options = loader.loadEstimatorOptions()
    assert isinstance(options, st.EstimatorOptions)
    assert options.smootherLag >= 0.0 or options.smootherLag == st.Estimator.BATCH_LAG
    assert options.compute_covariance is True
    assert options.lm_params.getMaxIterations() == 100


def test_prior_loaders(loader):
    assert loader.loadPriorType().lower() in ("wnoa", "wnov")
    reference, period, Q0, Q1, Q2, Q3, Q4 = loader.loadWNOAPrior()
    assert isinstance(reference, wnoa.Values)
    assert period > 0
    assert Q0.shape == (18, 18) and Q4.shape == (6, 6)
    assert isinstance(loader.loadWNOAPriorOptions(), wnoa.PriorOptions)
    T0, epsilon0, period, Q0, Q1, Q2 = loader.loadWNOVPrior()
    assert isinstance(T0, gtsam.Pose3) and epsilon0.shape == (6,)
    assert isinstance(loader.loadWNOVPriorOptions(), wnov.PriorOptions)
    reference, period, Q0, Q2 = loader.loadWNOAQuasiStaticPrior()
    assert isinstance(reference, wnoa_quasi_static.Values)
    assert Q0.shape == (12, 12) and Q2.shape == (6, 6)
    assert isinstance(loader.loadWNOAQuasiStaticPriorOptions(), wnoa_quasi_static.PriorOptions)


@pytest.mark.parametrize("name, factor_type", [("wnoa", wnoa.FactorType.WNOA), ("legacy", wnoa.FactorType.Legacy),
                                                ("paper", wnoa.FactorType.Paper)])
def test_wnoa_factor_type_is_read(tmp_path, name, factor_type):
    path = tmp_path / "factor_type.json"
    path.write_text('{ "prior": { "type": "wnoa", "wnoa": { "factor_type": "%s" } } }' % name)
    loader = utilities.ConfigLoader()
    assert loader.loadConfig(str(path))
    assert loader.loadWNOAPriorOptions().factor_type == factor_type


def test_demo_metadata_and_window(loader):
    length, nodes = loader.loadDemoMetadata()
    assert length > 0 and nodes >= 2
    start, end = loader.loadDemoTimeWindow()
    assert isinstance(start, float) and isinstance(end, float)
    assert loader.loadExtraction() in (utilities.Extraction.Final, utilities.Extraction.Back)


def test_load_locks(loader):
    length, nodes = loader.loadDemoMetadata()
    loader.loadLocks(wnoa.ContinuumRobotState(length, nodes))
    loader.loadLocks(wnoa_quasi_static.ContinuumRobotState(length, nodes))


def test_visualizer_options_from_config(tmp_path):
    visualizer = pytest.importorskip("spacetime.visualizer", reason="built without the visualizer")
    config = tmp_path / "visualizer.json"
    config.write_text(
        json.dumps(
            {
                "visualizer": {
                    "robot": {"radius": 0.02},
                    "cameras": [
                        {"position": [0, 1, 0.233], "look_at": [0, 0, 0.233], "up": [0, 0, -1]},
                        {"position": [1, 2, 3]},
                    ],
                }
            }
        )
    )
    loader = utilities.ConfigLoader()
    assert loader.loadConfig(str(config))
    options = loader.loadVisualizerOptions()
    assert isinstance(options, visualizer.VisualizerOptions)
    assert options.robot.radius == pytest.approx(0.02)
    assert len(options.cameras) == 2
    np.testing.assert_allclose(options.cameras[0].position, [0, 1, 0.233])
    np.testing.assert_allclose(options.cameras[0].focal_point, [0, 0, 0.233])
    np.testing.assert_allclose(options.cameras[0].view_up, [0, 0, -1])
    np.testing.assert_allclose(options.cameras[1].position, [1, 2, 3])
    np.testing.assert_allclose(options.cameras[1].view_up, visualizer.CameraOptions().view_up)


def test_invalid_visualizer_options_raise(tmp_path):
    pytest.importorskip("spacetime.visualizer", reason="built without the visualizer")
    config = tmp_path / "visualizer.json"
    config.write_text(json.dumps({"visualizer": {"robot": {"radius": 0.0}}}))
    loader = utilities.ConfigLoader()
    assert loader.loadConfig(str(config))
    with pytest.raises(ValueError):
        loader.loadVisualizerOptions()


def test_parse_extraction():
    assert utilities.parseExtraction("final") == utilities.Extraction.Final
    assert utilities.parseExtraction("back") == utilities.Extraction.Back
    with pytest.raises(ValueError):
        utilities.parseExtraction("middle")


def test_window_history():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator(compute_covariance=True)
    estimator.addModel(prior)
    run_session(estimator, end_time=0.2)
    history = utilities.WindowHistory()
    assert history.posteriorAt(0.1) is None
    history.record(estimator)
    assert history.posteriorAt(0.2) is not None


def test_export_ground_truth_predictions_writes_logs(tmp_path):
    from spacetime import sensors
    length = 0.2
    _, prior = make_wnoa_robot(length=length)
    estimator = batch_estimator(compute_covariance=True)
    estimator.addModel(prior)
    gt = sensors.PoseSensor(np.eye(6), prior)
    gt.addMeasurement(st.types.MeasurementBase(length, 0.1, gtsam.Pose3().matrix(), np.eye(6)))
    run_session(estimator, end_time=0.2)
    count = utilities.exportGroundTruthPredictions(prior, estimator.getCurrentValues(),
                                                   estimator.getCurrentCovariance(), [gt], str(tmp_path))
    assert count == 1
    assert (tmp_path / "error_pred.txt").exists()
    assert (tmp_path / "error_pred_cov.txt").exists()
