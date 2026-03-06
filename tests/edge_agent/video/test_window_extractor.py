from datetime import datetime, timedelta, timezone

import numpy as np

from edge_agent.video.ring_buffer import FrameItem
from edge_agent.video.window_extractor import select_frames_evenly


def test_select_frames_evenly_deterministic_and_capped():
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    items = []
    # 0..9 seconds, one frame per second
    for i in range(10):
        items.append(
            FrameItem(
                ts=t0 + timedelta(seconds=i), frame=np.zeros((2, 2), dtype=np.uint8)
            )
        )

    start = t0
    end = t0 + timedelta(seconds=9)

    selected = select_frames_evenly(
        items, start=start, end=end, target_fps=2.0, max_frames=5
    )
    assert len(selected) <= 5
    # Should be monotonically increasing timestamps
    assert all(selected[i].ts <= selected[i + 1].ts for i in range(len(selected) - 1))


def test_select_frames_handles_single_point_window():
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    items = [
        FrameItem(ts=t0, frame=np.zeros((2, 2), dtype=np.uint8)),
        FrameItem(ts=t0 + timedelta(seconds=1), frame=np.zeros((2, 2), dtype=np.uint8)),
    ]
    selected = select_frames_evenly(
        items, start=t0, end=t0, target_fps=5.0, max_frames=10
    )
    assert len(selected) == 1
