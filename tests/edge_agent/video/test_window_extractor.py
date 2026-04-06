from datetime import datetime, timedelta, timezone

import numpy as np

from edge_agent.video.ring_buffer import FrameItem
from edge_agent.video.window_extractor import select_frames_evenly


def test_select_frames_evenly_biases_recent_frames_but_stays_capped():
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    items = [
        FrameItem(
            ts=t0 + timedelta(seconds=i), frame=np.zeros((2, 2, 3), dtype=np.uint8)
        )
        for i in range(30)
    ]

    selected = select_frames_evenly(
        items,
        start=t0,
        end=t0 + timedelta(seconds=29),
        target_fps=5.0,
        max_frames=10,
    )

    assert len(selected) <= 10
    assert selected == sorted(selected, key=lambda it: it.ts)
    assert selected[-1].ts >= t0 + timedelta(seconds=24)


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
    selected_again = select_frames_evenly(
        items, start=start, end=end, target_fps=2.0, max_frames=5
    )
    assert len(selected) <= 5
    # Should be monotonically increasing timestamps
    assert all(selected[i].ts <= selected[i + 1].ts for i in range(len(selected) - 1))
    assert [item.ts for item in selected] == [item.ts for item in selected_again]


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


def test_select_frames_handles_empty_or_capped_inputs():
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    items = [
        FrameItem(ts=t0, frame=np.zeros((2, 2), dtype=np.uint8)),
    ]

    assert (
        select_frames_evenly(items, start=t0, end=t0, target_fps=5.0, max_frames=0)
        == []
    )
    assert (
        select_frames_evenly([], start=t0, end=t0, target_fps=5.0, max_frames=10) == []
    )


def test_select_frames_sorts_unsorted_inputs():
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    items = [
        FrameItem(ts=t0 + timedelta(seconds=3), frame=np.zeros((2, 2), dtype=np.uint8)),
        FrameItem(ts=t0 + timedelta(seconds=1), frame=np.zeros((2, 2), dtype=np.uint8)),
        FrameItem(ts=t0 + timedelta(seconds=2), frame=np.zeros((2, 2), dtype=np.uint8)),
    ]

    selected = select_frames_evenly(
        items,
        start=t0,
        end=t0 + timedelta(seconds=3),
        target_fps=10.0,
        max_frames=3,
    )

    assert selected == sorted(selected, key=lambda it: it.ts)
