from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .types import MotionEvent


class TriggerManager:
    """
    Accepts triggers (TCP now, local later) and outputs only "accepted" events.

    Per camera_id:
      - Merge window: drop triggers that arrive within merge_window_sec of the last seen trigger.
      - Cooldown: after firing, drop triggers until cooldown_sec passes.
    """

    def __init__(self, *, cooldown_sec: int, merge_window_sec: float) -> None:
        self._cooldown = timedelta(seconds=cooldown_sec)
        self._merge_window = timedelta(seconds=merge_window_sec)

        self._last_seen: dict[str, datetime] = defaultdict(self._min_dt)
        self._last_fired: dict[str, datetime] = defaultdict(self._min_dt)

        self._out: asyncio.Queue[MotionEvent] = asyncio.Queue()

    @staticmethod
    def _min_dt() -> datetime:
        return datetime.min.replace(tzinfo=timezone.utc)

    @staticmethod
    def _key(evt: MotionEvent) -> str:
        return evt.camera_id or "unknown"

    def accept(self, evt: MotionEvent) -> bool:
        cam = self._key(evt)
        now = evt.received_at_utc
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Merge/dedupe
        if now - self._last_seen[cam] <= self._merge_window:
            self._last_seen[cam] = now
            return False
        self._last_seen[cam] = now

        # Cooldown
        if now - self._last_fired[cam] < self._cooldown:
            return False

        self._last_fired[cam] = now
        self._out.put_nowait(evt)
        return True

    async def next_event(self) -> MotionEvent:
        return await self._out.get()
