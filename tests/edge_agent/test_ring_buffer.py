from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from edge_agent.video.ring_buffer import RingBuffer


def _frame(w: int = 4, h: int = 3, value: int = 0) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def test_ring_buffer_keeps_only_recent_seconds():
    ring = RingBuffer(seconds=10)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    # Push frames at t0, t0+5, t0+11
    ring.push(t0, _frame(value=1))
    ring.push(t0 + timedelta(seconds=5), _frame(value=2))
    ring.push(t0 + timedelta(seconds=11), _frame(value=3))

    # At t0+11, cutoff is t0+1, so the t0 frame should be evicted.
    assert ring.size() == 2


def test_ring_buffer_latest_returns_newest_frame():
    ring = RingBuffer(seconds=10)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    ring.push(t0, _frame(value=10))
    ring.push(t0 + timedelta(seconds=1), _frame(value=20))

    latest = ring.latest()
    assert latest is not None
    assert int(latest[0, 0]) == 20


def test_ring_buffer_window_returns_only_frames_in_range():
    ring = RingBuffer(seconds=10)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    ring.push(t0, _frame(value=1))
    ring.push(t0 + timedelta(seconds=2), _frame(value=2))
    ring.push(t0 + timedelta(seconds=4), _frame(value=3))

    window = ring.window(t0 + timedelta(seconds=1), t0 + timedelta(seconds=3))
    # should include only the t0+2 frame
    assert len(window) == 1
    assert int(window[0].frame[0, 0]) == 2


def test_ring_buffer_window_empty_when_no_frames_match():
    ring = RingBuffer(seconds=10)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    ring.push(t0, _frame(value=1))
    window = ring.window(t0 + timedelta(seconds=1), t0 + timedelta(seconds=2))
    assert window == []
