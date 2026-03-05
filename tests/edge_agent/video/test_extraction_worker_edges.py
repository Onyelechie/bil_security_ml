import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from edge_agent.triggers.incident_manager import ExtractionJob
from edge_agent.video.extraction_worker import ExtractionWorker
from edge_agent.video.ring_buffer import RingBuffer


@pytest.mark.asyncio
async def test_extraction_worker_drops_when_no_ring():
    def provider(_cam: str):
        return None

    worker = ExtractionWorker(
        ring_provider=provider,
        target_fps=2.0,
        max_frames=10,
        wait_grace_sec=0.01,
        queue_max=10,
        results_max=10,
    )
    await worker.start()
    try:
        now = datetime.now(timezone.utc)
        job = ExtractionJob(
            incident_id="inc1",
            camera_id="1",
            window_start=now - timedelta(seconds=2),
            window_end=now - timedelta(seconds=1),
            created_at_utc=now,
            reason="quiet",
        )
        await worker.enqueue(job)
        res = await asyncio.wait_for(worker.results.get(), timeout=1.0)

        assert res.status.value == "dropped"
        assert res.reason == "no_ring"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_extraction_worker_drops_when_no_frames_in_window():
    ring = RingBuffer(seconds=30)

    def provider(_cam: str):
        return ring

    worker = ExtractionWorker(
        ring_provider=provider,
        target_fps=2.0,
        max_frames=10,
        wait_grace_sec=0.01,
        queue_max=10,
        results_max=10,
    )
    await worker.start()
    try:
        now = datetime.now(timezone.utc)
        # No frames pushed at all → window() returns empty
        job = ExtractionJob(
            incident_id="inc2",
            camera_id="1",
            window_start=now - timedelta(seconds=10),
            window_end=now - timedelta(seconds=9),
            created_at_utc=now,
            reason="quiet",
        )
        await worker.enqueue(job)
        res = await asyncio.wait_for(worker.results.get(), timeout=1.0)

        assert res.status.value == "dropped"
        assert res.reason == "no_frames"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_extraction_worker_returns_partial_on_postroll_timeout_but_has_frames():
    ring = RingBuffer(seconds=30)

    # Use timestamps in the past so worker doesn't actually sleep long
    now = datetime.now(timezone.utc)
    base = now - timedelta(seconds=100)

    # Frames only up to base+2
    for i in range(3):
        ring.push(base + timedelta(seconds=i), np.zeros((2, 2), dtype=np.uint8))

    def provider(_cam: str):
        return ring

    worker = ExtractionWorker(
        ring_provider=provider,
        target_fps=2.0,
        max_frames=10,
        wait_grace_sec=0.02,  # tiny wait, still triggers timeout path
        queue_max=10,
        results_max=10,
    )
    await worker.start()
    try:
        job = ExtractionJob(
            incident_id="inc3",
            camera_id="1",
            window_start=base + timedelta(seconds=0),
            window_end=base + timedelta(seconds=5),  # beyond latest_ts (base+2)
            created_at_utc=now,
            reason="quiet",
        )
        await worker.enqueue(job)
        res = await asyncio.wait_for(worker.results.get(), timeout=1.0)

        assert res.status.value == "partial"
        assert res.reason == "timeout"
        assert len(res.selected) > 0
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_extraction_worker_enqueue_does_not_crash_when_queue_full():
    ring = RingBuffer(seconds=30)

    def provider(_cam: str):
        return ring

    worker = ExtractionWorker(
        ring_provider=provider,
        target_fps=2.0,
        max_frames=10,
        wait_grace_sec=0.01,
        queue_max=1,  # force full quickly
        results_max=10,
    )

    now = datetime.now(timezone.utc)
    job1 = ExtractionJob("a", "1", now, now, now, "quiet")
    job2 = ExtractionJob("b", "1", now, now, now, "quiet")

    # Fill queue without starting worker
    worker._queue.put_nowait(job1)  # noqa: SLF001 (private access ok in test)
    await worker.enqueue(job2)  # should drop oldest and keep newest, no exception

    assert worker._queue.qsize() == 1
