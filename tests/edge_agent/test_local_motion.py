from __future__ import annotations

import numpy as np
import pytest

from edge_agent.triggers.local_motion_trigger import motion_score


def test_motion_score_zero_when_identical():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    assert motion_score(a, b, pixel_delta=10) == 0.0


def test_motion_score_detects_change_ratio():
    prev = np.zeros((10, 10), dtype=np.uint8)
    curr = np.zeros((10, 10), dtype=np.uint8)

    # Change exactly 10 pixels by +50
    coords = [(0, i) for i in range(10)]
    for r, c in coords:
        curr[r, c] = 50

    score = motion_score(prev, curr, pixel_delta=25)
    # 10 changed pixels out of 100 total => 0.10
    assert abs(score - 0.10) < 1e-6


def test_motion_score_raises_on_shape_mismatch():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((9, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        motion_score(a, b, pixel_delta=10)
