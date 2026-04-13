from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np

from ..config import EdgeSettings
from ..video.ring_buffer import RingBuffer
from .trigger_manager import TriggerManager
from .types import MotionEvent

logger = logging.getLogger(__name__)

OnMotionFn = Callable[[MotionEvent, bool], None]


def _polygon_to_pixels(
    polygon: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert a normalized polygon [[x, y], ...] in 0..1 coordinates
    into OpenCV pixel coordinates.
    """
    pts: list[list[int]] = []

    for pt in polygon:
        if len(pt) != 2:
            raise ValueError(f"Invalid polygon point: {pt}")

        x_norm = float(pt[0])
        y_norm = float(pt[1])

        x = int(round(max(0.0, min(1.0, x_norm)) * (width - 1)))
        y = int(round(max(0.0, min(1.0, y_norm)) * (height - 1)))

        pts.append([x, y])

    if len(pts) < 3:
        raise ValueError(f"Polygon must have at least 3 points: {polygon}")

    return np.asarray(pts, dtype=np.int32)


def build_score_mask(
    frame_shape: tuple[int, int],
    *,
    include_polygons: list[list[list[float]]],
    exclude_polygons: list[list[list[float]]],
) -> np.ndarray:
    """
    Build a uint8 mask for motion scoring.
    255 means pixel is active for scoring.
    0 means ignored.
    """
    height, width = frame_shape

    if include_polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in include_polygons:
            pts = _polygon_to_pixels(poly, width, height)
            cv2.fillPoly(mask, [pts], 255)
    else:
        mask = np.full((height, width), 255, dtype=np.uint8)

    for poly in exclude_polygons:
        pts = _polygon_to_pixels(poly, width, height)
        cv2.fillPoly(mask, [pts], 0)

    return mask


def motion_score(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    pixel_delta: int,
    score_mask: np.ndarray | None = None,
) -> float:
    """
    Returns fraction (0..1) of changed pixels.
    If score_mask is provided, only pixels where mask > 0 are considered.
    Expects grayscale uint8 frames with the same shape.
    """
    if prev.shape != curr.shape:
        raise ValueError("prev and curr must have the same shape")

    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))

    if score_mask is None:
        changed = diff > int(pixel_delta)
        return float(changed.mean())

    if score_mask.shape != curr.shape:
        raise ValueError("score_mask must have the same shape as frames")

    active = score_mask > 0
    if not np.any(active):
        return 0.0

    changed = diff[active] > int(pixel_delta)
    return float(changed.mean())


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a frame to 2D grayscale uint8 for cheap motion detection.
    Accepts:
    - HxW grayscale
    - HxWx1 grayscale
    - HxWx3 BGR
    """
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported frame shape for motion detection: {frame.shape}")


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
        self._score_mask: np.ndarray | None = None
        self._score_mask_shape: tuple[int, int] | None = None

    @property
    def queue(self) -> asyncio.Queue[MotionEvent]:
        return self._queue

    def _ensure_score_mask(self, gray_frame: np.ndarray) -> np.ndarray:
        shape = gray_frame.shape

        if self._score_mask is not None and self._score_mask_shape == shape:
            return self._score_mask

        mask = build_score_mask(
            shape,
            include_polygons=self._cfg.motion_include_polygons,
            exclude_polygons=self._cfg.motion_exclude_polygons,
        )

        self._score_mask = mask
        self._score_mask_shape = shape

        active_ratio = float((mask > 0).mean())
        logger.info(
            "Local motion mask ready: active_pixels=%.1f%% frame=%sx%s",
            active_ratio * 100.0,
            shape[1],
            shape[0],
        )

        return mask

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

            try:
                curr_gray = to_grayscale(curr)
                score_mask = self._ensure_score_mask(curr_gray)
            except ValueError:
                await asyncio.sleep(period)
                continue

            if self._prev is not None:
                try:
                    score = motion_score(
                        self._prev,
                        curr_gray,
                        pixel_delta=self._cfg.motion_pixel_delta,
                        score_mask=score_mask,
                    )
                except ValueError:
                    self._prev = curr_gray
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

            self._prev = curr_gray
            await asyncio.sleep(period)
