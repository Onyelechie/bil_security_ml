from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
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


def _closest_index(ts_list: list[datetime], target: datetime) -> int:
    j = bisect_left(ts_list, target)
    if j <= 0:
        return 0
    if j >= len(ts_list):
        return len(ts_list) - 1

    left = ts_list[j - 1]
    right = ts_list[j]
    return (j - 1) if (target - left) <= (right - target) else j


def _build_targets(start: datetime, end: datetime, n: int) -> list[datetime]:
    if n <= 0:
        return []
    if n == 1 or end <= start:
        return [end]
    return [start + (end - start) * (i / (n - 1)) for i in range(n)]


def select_frames_evenly(
    items: list[FrameItem],
    *,
    start: datetime,
    end: datetime,
    target_fps: float,
    max_frames: int,
) -> list[FrameItem]:
    """
    Smarter deterministic selection:
    - keeps some early context
    - covers the full incident
    - biases more frames toward the end/recent motion

    This is still deterministic and capped, but improves over purely even sampling.
    """
    if not items or max_frames <= 0:
        return []

    items = _ensure_sorted(items)
    ts_list = [it.ts for it in items]

    duration_s = (end - start).total_seconds()
    if duration_s <= 0:
        idx = _closest_index(ts_list, start)
        return [items[idx]]

    step = 1.0 / max(float(target_fps), 0.1)
    theoretical_targets = int(duration_s / step) + 1
    n_total = max(1, min(theoretical_targets, int(max_frames)))

    if n_total <= 3:
        targets = _build_targets(start, end, n_total)
    else:
        recent_n = max(1, int(round(n_total * 0.50)))
        full_n = max(1, int(round(n_total * 0.30)))
        early_n = max(0, n_total - recent_n - full_n)

        recent_span_s = min(max(duration_s * 0.40, 2.0), 6.0, duration_s)
        early_span_s = min(max(duration_s * 0.20, 1.0), 3.0, duration_s)

        recent_start = end - timedelta(seconds=recent_span_s)
        early_end = start + timedelta(seconds=early_span_s)

        targets = []
        targets.extend(_build_targets(start, early_end, early_n))
        targets.extend(_build_targets(start, end, full_n))
        targets.extend(_build_targets(recent_start, end, recent_n))

    chosen_idx: set[int] = set()
    for t in targets:
        chosen_idx.add(_closest_index(ts_list, t))
        if len(chosen_idx) >= max_frames:
            break

    ordered_idx = sorted(chosen_idx)
    if len(ordered_idx) > max_frames:
        ordered_idx = ordered_idx[:max_frames]

    return [items[i] for i in ordered_idx]
