from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np

from ..config import EdgeSettings
from ..video.ring_buffer import RingBuffer
from .trigger_manager import TriggerManager
from .types import MotionEvent

logger = logging.getLogger(__name__)


def motion_score(prev: np.ndarray, curr: np.ndarray, *, pixel_delta: int) -> float:
    """
    Returns fraction (0..1) of pixels that changed by more than pixel_delta.
    Expects grayscale uint8 frames with the same shape.
    """
    if prev.shape != curr.shape:
        raise ValueError("prev and curr must have the same shape")

    # Use int16 to avoid uint8 wraparound on subtraction
    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
    changed = (diff > int(pixel_delta)).mean()
    return float(changed)


class LocalMotionTrigger:
    """
    Lightweight motion trigger: compares frames from RingBuffer.latest().
    Emits MotionEvent(source="local") through TriggerManager when motion is detected.
    """

    def __init__(self, cfg: EdgeSettings, ring: RingBuffer, mgr: TriggerManager) -> None:
        self._cfg = cfg
        self._ring = ring
        self._mgr = mgr
        self._prev: np.ndarray | None = None

    async def run(self) -> None:
        period = 1.0 / max(self._cfg.motion_fps, 0.1)

        while True:
            curr = self._ring.latest()
            if curr is None:
                await asyncio.sleep(0.2)
                continue

            if self._prev is not None:
                try:
                    score = motion_score(self._prev, curr, pixel_delta=self._cfg.motion_pixel_delta)
                except ValueError:
                    # frame shape mismatch (shouldn't happen if reader is stable)
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
                    if accepted:
                        logger.info("LOCAL MOTION(accepted): camera_id=%s score=%.4f", evt.camera_id, score)
                    else:
                        logger.debug("LOCAL MOTION(dropped): camera_id=%s score=%.4f", evt.camera_id, score)

            self._prev = curr
            await asyncio.sleep(period)
