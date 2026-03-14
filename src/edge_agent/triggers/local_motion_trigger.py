from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from ..config import EdgeSettings
from ..video.ring_buffer import RingBuffer
from .trigger_manager import TriggerManager
from .types import MotionEvent

logger = logging.getLogger(__name__)

OnMotionFn = Callable[[MotionEvent, bool], None]


def motion_score(prev: np.ndarray, curr: np.ndarray, *, pixel_delta: int) -> float:
    """
    Returns fraction (0..1) of pixels that changed by more than pixel_delta.
    Expects grayscale uint8 frames with the same shape.
    """
    if prev.shape != curr.shape:
        raise ValueError("prev and curr must have the same shape")

    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
    changed = (diff > int(pixel_delta)).mean()
    return float(changed)


class LocalMotionTrigger:
    """
    Lightweight motion trigger. Emits accepted MotionEvent through TriggerManager.
    Also can call on_motion(evt, accepted) for BOTH accepted and dropped events.
    """

    def __init__(
        self,
        cfg: EdgeSettings,
        ring: RingBuffer,
        mgr: TriggerManager,
        *,
        queue_max: int = 1000,
        on_motion: OnMotionFn | None = None,
    ) -> None:
        self._cfg = cfg
        self._ring = ring
        self._mgr = mgr
        self._prev: np.ndarray | None = None

        self._on_motion = on_motion

        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[MotionEvent] = asyncio.Queue(maxsize=queue_max)

    @property
    def queue(self) -> asyncio.Queue[MotionEvent]:
        return self._queue

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run(), name="local-motion-trigger")
        logger.info("Local motion trigger started")

    async def stop(self) -> None:
        self._stop.set()

        task = self._task
        self._task = None

        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        logger.info("Local motion trigger stopped")

    async def run(self) -> None:
        period = 1.0 / max(self._cfg.motion_fps, 0.1)

        while not self._stop.is_set():
            curr = self._ring.latest()
            if curr is None:
                await asyncio.sleep(0.2)
                continue

            if self._prev is not None:
                try:
                    score = motion_score(
                        self._prev, curr, pixel_delta=self._cfg.motion_pixel_delta
                    )
                except ValueError:
                    self._prev = curr
                    await asyncio.sleep(period)
                    continue

                if score >= self._cfg.motion_threshold:
                    evt = MotionEvent(
                        received_at_utc=datetime.now(timezone.utc),
                        site_id=self._cfg.site_id,
                        edge_pc_id=self._cfg.edge_pc_id,
                        camera_id=self._cfg.default_camera_id,
                        source="local",
                    )

                    accepted = self._mgr.accept(evt)

                    if self._on_motion:
                        with suppress(Exception):
                            self._on_motion(evt, accepted)

                    if accepted:
                        logger.info(
                            "LOCAL MOTION(accepted): camera_id=%s score=%.4f",
                            evt.camera_id,
                            score,
                        )
                        with suppress(asyncio.QueueFull):
                            self._queue.put_nowait(evt)
                    else:
                        logger.debug(
                            "LOCAL MOTION(dropped): camera_id=%s score=%.4f",
                            evt.camera_id,
                            score,
                        )

            self._prev = curr
            await asyncio.sleep(period)
