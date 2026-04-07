from __future__ import annotations

import numpy as np
import pytest

from edge_agent.triggers.local_motion_trigger import motion_score, to_grayscale


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
