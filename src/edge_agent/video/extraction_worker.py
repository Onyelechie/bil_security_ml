from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Callable

from ..triggers.incident_manager import ExtractionJob
from .ring_buffer import RingBuffer
from .window_extractor import WindowResult, WindowStatus, select_frames_evenly

logger = logging.getLogger(__name__)


RingProvider = Callable[[str], RingBuffer | None]


class ExtractionWorker:
    """
    Background worker that:
    - waits (bounded) for post-roll frames to exist up to window_end
    - extracts window frames from ring buffer
    - selects frames deterministically (target_fps/max_frames)
    """

    def __init__(
        self,
        *,
        ring_provider: RingProvider,
        target_fps: float,
        max_frames: int,
        wait_grace_sec: float,
        queue_max: int = 100,
        results_max: int = 100,
    ) -> None:
        self._ring_provider = ring_provider
        self._target_fps = float(target_fps)
        self._max_frames = int(max_frames)
        self._wait_grace = float(wait_grace_sec)

        self._queue: asyncio.Queue[ExtractionJob] = asyncio.Queue(maxsize=queue_max)
        self._results: asyncio.Queue[WindowResult] = asyncio.Queue(maxsize=results_max)

        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def results(self) -> asyncio.Queue[WindowResult]:
        return self._results

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run(), name="extraction-worker")
        logger.info("Extraction worker started")

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.info("Extraction worker stopped")

    async def enqueue(self, job: ExtractionJob) -> None:
        """
        Keep newest jobs if overloaded: drop oldest then enqueue newest.
        """
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            with suppress(asyncio.QueueEmpty):
                _ = self._queue.get_nowait()
            with suppress(asyncio.QueueFull):
                self._queue.put_nowait(job)

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            ring = self._ring_provider(job.camera_id)
            if ring is None:
                res = WindowResult(
                    incident_id=job.incident_id,
                    camera_id=job.camera_id,
                    window_start=job.window_start,
                    window_end=job.window_end,
                    selected=[],
                    status=WindowStatus.DROPPED,
                    reason="no_ring",
                )
                self._put_result(res)
                logger.warning(
                    "WINDOW(dropped): incident=%s camera=%s reason=no_ring",
                    job.incident_id,
                    job.camera_id,
                )
                continue

            ok = await self._wait_for_postroll(ring, job.window_end)
            items = ring.window(job.window_start, job.window_end)

            if not items:
                res = WindowResult(
                    incident_id=job.incident_id,
                    camera_id=job.camera_id,
                    window_start=job.window_start,
                    window_end=job.window_end,
                    selected=[],
                    status=WindowStatus.DROPPED,
                    reason="no_frames",
                )
                self._put_result(res)
                logger.warning(
                    "WINDOW(dropped): incident=%s camera=%s reason=no_frames",
                    job.incident_id,
                    job.camera_id,
                )
                continue

            selected = select_frames_evenly(
                items,
                start=job.window_start,
                end=job.window_end,
                target_fps=self._target_fps,
                max_frames=self._max_frames,
            )

            status = WindowStatus.READY if ok else WindowStatus.PARTIAL
            reason = "ok" if ok else "timeout"

            res = WindowResult(
                incident_id=job.incident_id,
                camera_id=job.camera_id,
                window_start=job.window_start,
                window_end=job.window_end,
                selected=selected,
                status=status,
                reason=reason,
            )
            self._put_result(res)

            logger.info(
                "WINDOW(%s): incident=%s camera=%s selected=%d span=%.2fs reason=%s",
                res.status.value,
                res.incident_id,
                res.camera_id,
                len(res.selected),
                (res.window_end - res.window_start).total_seconds(),
                res.reason,
            )

    def _put_result(self, res: WindowResult) -> None:
        with suppress(asyncio.QueueFull):
            self._results.put_nowait(res)

    async def _wait_for_postroll(self, ring: RingBuffer, end_ts: datetime) -> bool:
        """
        Wait until ring.latest_ts() >= end_ts, bounded by (remaining + grace).
        """
        now = datetime.now(timezone.utc)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)

        remaining = (end_ts - now).total_seconds()
        remaining = max(0.0, remaining)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + remaining + self._wait_grace

        while loop.time() < deadline and not self._stop.is_set():
            latest = ring.latest_ts()
            if latest is not None:
                # normalize tz
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                if latest >= end_ts:
                    return True
            await asyncio.sleep(0.1)

        return False
