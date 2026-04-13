from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from edge_agent.config import EdgeSettings
from edge_agent.triggers.local_motion_trigger import (
    LocalMotionTrigger,
    build_score_mask,
    motion_score,
    to_grayscale,
)


def test_motion_score_zero_when_identical():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    assert motion_score(a, b, pixel_delta=10) == 0.0


def test_motion_score_detects_change_ratio():
    prev = np.zeros((10, 10), dtype=np.uint8)
    curr = np.zeros((10, 10), dtype=np.uint8)

    coords = [(0, i) for i in range(10)]
    for r, c in coords:
        curr[r, c] = 50

    score = motion_score(prev, curr, pixel_delta=25)
    assert abs(score - 0.10) < 1e-6


def test_motion_score_raises_on_shape_mismatch():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((9, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        motion_score(a, b, pixel_delta=10)


def test_to_grayscale_accepts_bgr():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    gray = to_grayscale(frame)

    assert gray.shape == (20, 30)
    assert gray.dtype == np.uint8


def test_to_grayscale_accepts_single_channel_3d():
    frame = np.zeros((20, 30, 1), dtype=np.uint8)
    gray = to_grayscale(frame)

    assert gray.shape == (20, 30)
    assert gray.dtype == np.uint8


def test_motion_score_respects_score_mask():
    prev = np.zeros((4, 4), dtype=np.uint8)
    curr = np.zeros((4, 4), dtype=np.uint8)

    curr[0, 0] = 100
    curr[3, 3] = 100

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255

    score = motion_score(prev, curr, pixel_delta=25, score_mask=mask)
    assert score == 1.0


def test_build_score_mask_with_include_and_exclude():
    mask = build_score_mask(
        (10, 10),
        include_polygons=[[[0.0, 0.0], [0.9, 0.0], [0.9, 0.9], [0.0, 0.9]]],
        exclude_polygons=[[[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]],
    )

    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask[8, 8] == 255
    assert mask[0, 0] == 0


def test_mark_ptz_motion_requires_consecutive_hits():
    cfg = EdgeSettings(ptz_consecutive_frames=2, ptz_suppress_sec=3.0)
    trigger = LocalMotionTrigger(cfg, ring=object(), mgr=object())

    now = datetime.now(timezone.utc)

    trigger._mark_ptz_motion(now, 0.6)
    assert trigger._ptz_hits == 1
    assert trigger._ptz_suppress_until is None

    trigger._mark_ptz_motion(now, 0.6)
    assert trigger._ptz_hits == 2
    assert trigger._ptz_suppress_until is not None


def test_ptz_active_respects_expiry():
    cfg = EdgeSettings()
    trigger = LocalMotionTrigger(cfg, ring=object(), mgr=object())

    now = datetime.now(timezone.utc)
    trigger._ptz_suppress_until = now + timedelta(seconds=1)

    assert trigger._ptz_active(now) is True
    assert trigger._ptz_active(now + timedelta(seconds=2)) is False
