"""The standalone interface session."""

import threading
import time

import pytest

import spacetime as st
from spacetime import interface

from conftest import batch_estimator, make_wnoa_robot, run_session


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
    options = st.SessionOptions()
    options.end_time = 0.5
    options.period = 0.25
    session = st.Session(options)
    times = []
    session.setTickCallback(lambda t: times.append(t))
    session.start(estimator)
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


class FakeViewer(interface.Viewer):
    """A Viewer written in Python, logging what the session asks of it."""

    def __init__(self, online):
        super().__init__()
        self.online = online
        self.log = []
        self.on_closed = None
        self.live = None

    def open(self, estimator, on_closed):
        self.log.append("open")
        self.on_closed = on_closed

    def isOnline(self):
        return self.online

    def beginPlayback(self, start_time, end_time, live_time_source):
        self.log.append(("play", start_time, end_time, live_time_source is not None))
        self.live = live_time_source

    def run(self):
        self.log.append("run")
        return 3

    def requestClose(self):
        self.log.append("close")


def offline_session(end_time=0.5, period=0.25):
    options = st.SessionOptions()
    options.end_time = end_time
    options.period = period
    return st.Session(options)


def test_offline_viewer_replays_the_run_after_estimation():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator()
    estimator.addModel(prior)
    viewer = FakeViewer(online=False)
    session = offline_session()
    session.start(estimator, viewer)
    assert viewer.log == ["open"]
    session.joinEstimation()
    assert session.finished()
    assert viewer.log == ["open"]  # not shown yet
    session.join()
    assert viewer.log == ["open", ("play", 0.0, 0.5, False), "run"]
    assert session.viewerExitCode() == 3


def test_online_viewer_follows_the_session_clock():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator()
    estimator.addModel(prior)
    viewer = FakeViewer(online=True)
    session = offline_session()
    session.start(estimator, viewer)
    assert viewer.log[1] == ("play", 0.0, 0.5, True)
    session.join()
    assert viewer.log == ["open", ("play", 0.0, 0.5, True), "run"]
    assert viewer.live() == pytest.approx(session.now())


def test_closing_the_viewer_stops_the_session():
    _, prior = make_wnoa_robot(nodes=5)
    estimator = batch_estimator()
    estimator.addModel(prior)
    viewer = FakeViewer(online=True)
    options = st.SessionOptions()
    options.end_time = 100.0
    options.period = 0.01
    session = st.Session(options)
    session.start(estimator, viewer)
    viewer.on_closed()  # e.g. the user closed the window
    session.join()
    assert session.finished()
    assert estimator.getSnapshot() is None or estimator.getSnapshot().time < 100.0


def test_dropping_an_unjoined_session_closes_its_viewer():
    _, prior = make_wnoa_robot()
    estimator = batch_estimator()
    estimator.addModel(prior)
    viewer = FakeViewer(online=False)
    session = offline_session()
    session.start(estimator, viewer)
    del session
    assert viewer.log == ["open", "close", "run"]
