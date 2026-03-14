import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from edge_agent.triggers.incident_manager import ExtractionJob
from edge_agent.video.extraction_worker import ExtractionWorker
from edge_agent.video.ring_buffer import RingBuffer


@pytest.mark.asyncio
async def test_extraction_worker_ready_when_postroll_available():
    ring = RingBuffer(seconds=30)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    # push frames covering 0..10 seconds
    for i in range(11):
        ring.push(t0 + timedelta(seconds=i), np.zeros((2, 2), dtype=np.uint8))

    def provider(cam: str):
        return ring

    worker = ExtractionWorker(
        ring_provider=provider, target_fps=2.0, max_frames=10, wait_grace_sec=0.1
    )
    await worker.start()
    try:
        job = ExtractionJob(
            incident_id="abc",
            camera_id="1",
            window_start=t0 + timedelta(seconds=1),
            window_end=t0 + timedelta(seconds=6),
            created_at_utc=t0,
            reason="quiet",
        )
        await worker.enqueue(job)

        res = await asyncio.wait_for(worker.results.get(), timeout=2.0)
        assert res.status.value in ("ready", "partial")
        assert res.selected  # should have frames
    finally:
        await worker.stop()
