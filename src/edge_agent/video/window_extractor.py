from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .ring_buffer import FrameItem


class WindowStatus(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    DROPPED = "dropped"


@dataclass(slots=True, frozen=True)
class WindowResult:
    incident_id: str
    camera_id: str

    window_start: datetime
    window_end: datetime

    selected: list[FrameItem]

    status: WindowStatus
    reason: str  # "ok" | "timeout" | "no_frames" | "no_ring"


def _ensure_sorted(items: list[FrameItem]) -> list[FrameItem]:
    return sorted(items, key=lambda it: it.ts)


def select_frames_evenly(
    items: list[FrameItem],
    *,
    start: datetime,
    end: datetime,
    target_fps: float,
    max_frames: int,
) -> list[FrameItem]:
    """
    Deterministic selection:
    - build evenly spaced target timestamps from [start..end]
    - pick the closest available frame to each target
    - dedupe and cap
    """
    if not items or max_frames <= 0:
        return []

    items = _ensure_sorted(items)
    ts_list = [it.ts for it in items]

    duration_s = (end - start).total_seconds()
    if duration_s <= 0:
        # pick the closest frame to start
        idx = bisect_left(ts_list, start)
        if idx <= 0:
            return [items[0]]
        if idx >= len(items):
            return [items[-1]]
        # choose nearer neighbor
        left = items[idx - 1]
        right = items[idx]
        return [left if (start - left.ts) <= (right.ts - start) else right]

    step = 1.0 / max(float(target_fps), 0.1)
    # number of targets: inclusive of both ends-ish, and bounded by max_frames
    n_targets = int(duration_s / step) + 1
    n_targets = max(1, min(n_targets, int(max_frames)))

    # If max_frames is smaller than theoretical targets, increase step accordingly
    if n_targets == max_frames:
        step = max(duration_s / max_frames, step)

    chosen_idx: list[int] = []
    for i in range(n_targets):
        t = start + (end - start) * (i / max(n_targets - 1, 1))
        j = bisect_left(ts_list, t)

        if j <= 0:
            pick = 0
        elif j >= len(items):
            pick = len(items) - 1
        else:
            left = items[j - 1]
            right = items[j]
            pick = (j - 1) if (t - left.ts) <= (right.ts - t) else j

        if not chosen_idx or chosen_idx[-1] != pick:
            chosen_idx.append(pick)

        if len(chosen_idx) >= max_frames:
            break

    return [items[i] for i in chosen_idx]
