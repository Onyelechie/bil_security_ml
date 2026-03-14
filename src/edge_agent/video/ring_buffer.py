from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class FrameItem:
    ts: datetime
    frame: np.ndarray  # grayscale uint8 HxW


class RingBuffer:
    def __init__(self, *, seconds: int) -> None:
        self._seconds = int(seconds)
        self._items: deque[FrameItem] = deque()
        self._lock = Lock()

    @property
    def seconds(self) -> int:
        return self._seconds

    def push(self, ts: datetime, frame: np.ndarray) -> None:
        cutoff = ts - timedelta(seconds=self._seconds)
        with self._lock:
            self._items.append(FrameItem(ts=ts, frame=frame))
            while self._items and self._items[0].ts < cutoff:
                self._items.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._items[-1].frame if self._items else None

    def latest_item(self) -> FrameItem | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def latest_ts(self) -> datetime | None:
        it = self.latest_item()
        return it.ts if it else None

    def window(self, start: datetime, end: datetime) -> list[FrameItem]:
        with self._lock:
            return [it for it in self._items if start <= it.ts <= end]
