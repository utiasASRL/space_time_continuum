"""The standalone interface session."""

import threading
import time

import pytest

from spacetime import interface

from conftest import batch_estimator, make_wnoa_robot, run_session


def test_session_options_defaults_match_cpp():
    options = interface.SessionOptions()
    assert options.start_time == 0.0
    assert options.end_time == -1.0
    assert options.period == pytest.approx(1.0 / 30.0)
    assert options.real_time is False


def test_offline_session_runs_to_end_time():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator()
    estimator.addModel(prior)
    session = run_session(estimator, end_time=0.5, period=0.25)
    assert session.finished()
    assert session.endTime() == pytest.approx(0.5)
    assert estimator.getSnapshot().time == pytest.approx(0.5)
    assert session.isRealTime() is False


def test_join_releases_the_gil():
    _, prior = make_wnoa_robot(nodes=5)
    estimator = batch_estimator()
    estimator.addModel(prior)
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.001)

    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        run_session(estimator, end_time=3.0)
    finally:
        stop.set()
        thread.join()
    assert len(ticks) > 1


def test_tick_callback_sees_every_estimate():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator()
    estimator.addModel(prior)
    options = interface.SessionOptions()
    options.end_time = 0.5
    options.period = 0.25
    session = interface.Session(estimator, options)
    times = []
    session.setTickCallback(lambda t: times.append(t))
    session.start()
    session.join()
    assert times == pytest.approx([0.0, 0.25, 0.5])


def test_snapshot_history():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator()
    estimator.addModel(prior)
    run_session(estimator, end_time=0.2)
    history = interface.SnapshotHistory()
    assert history.latest() is None
    history.record(estimator.getSnapshot())
    assert history.size() == 1
    assert history.at(0.2).time == pytest.approx(0.2)
