"""Measurements, measurement queues and the key manager."""

import gtsam
import numpy as np
import pytest

from spacetime import types


def test_measurement_fields():
    mask = np.array([[1], [0], [1], [1], [1], [1]], dtype=np.int32)
    m = types.MeasurementBase(0.1, 2.0, np.arange(6.0).reshape(6, 1), np.eye(6) * 3, mask)
    assert m.arclength == 0.1
    assert m.time == 2.0
    np.testing.assert_array_equal(m.value.ravel(), np.arange(6.0))
    np.testing.assert_array_equal(m.covariance, np.eye(6) * 3)
    np.testing.assert_array_equal(m.mask, mask)
    m.time = 3.0
    assert m.time == 3.0


def test_measurement_without_mask_and_default():
    m = types.MeasurementBase(0.0, 1.0, np.eye(4), np.eye(6))
    assert m.time == 1.0
    types.MeasurementBase()


def test_queue_add_drain_and_last_time():
    queue = types.MeasurementQueue()
    assert queue.empty()
    assert queue.lastTime() is None
    for t in (0.0, 0.5, 1.0):
        queue.addMeasurement(types.MeasurementBase(0.0, t, np.zeros((6, 1)), np.eye(6)))
    assert queue.size() == 3
    assert queue.lastTime() == 1.0
    assert [m.time for m in queue.peekAll()] == [0.0, 0.5, 1.0]
    drained = queue.drainUpTo(0.5)
    assert [m.time for m in drained] == [0.0, 0.5]
    assert [m.time for m in queue.takeAll()] == [1.0]
    assert queue.empty()


def test_queue_callback_sees_each_measurement():
    queue = types.MeasurementQueue()
    seen = []
    queue.addMeasurementCallback(lambda m: seen.append(m.time))
    queue.addMeasurement(types.MeasurementBase(0.0, 0.25, np.zeros((6, 1)), np.eye(6)))
    assert seen == [0.25]


def test_key_manager_counts_per_character_and_stamps():
    km = types.KeyManager()
    a0 = km.nextKey("a", 0.5)
    a1 = km.nextKey("a", 1.5)
    b0 = km.nextKey("b", 2.0)
    assert gtsam.Symbol(a0).index() == 0 and gtsam.Symbol(a1).index() == 1
    assert chr(gtsam.Symbol(b0).chr()) == "b" and gtsam.Symbol(b0).index() == 0
    assert km.timestampOf(a1) == 1.5
    pending = km.drainPending()
    assert set(pending.keys()) == {a0, a1, b0}
    assert km.drainPending() == {}


def test_key_manager_unknown_key_raises():
    with pytest.raises(IndexError):
        types.KeyManager().timestampOf(gtsam.Symbol("z", 9).key())
