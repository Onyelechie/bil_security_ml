from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from .pipeline_runner import PipelineRunner
from .sender import ServerSender
from .video.ring_buffer import FrameItem

logger = logging.getLogger(__name__)


def run_sample_video(
    *,
    video_path: str,
    pipeline: PipelineRunner,
    sender: ServerSender,
    camera_id: str,
    window_sec: float,
    stride_sec: float,
    target_fps: float,
    max_frames: int,
) -> None:
    """
    Run a CCTV sample video directly through the pipeline.

    Behavior:
    - load the FULL video first
    - split into sequential chunks by frame index
    - analyze ALL frames in each chunk
    - send at most one best alert per chunk
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = max(float(target_fps), 1.0)

    started_at = datetime.now(timezone.utc)
    all_items: list[FrameItem] = []
    frame_index = 0

    logger.info("Loading sample video: %s", path)
    logger.info("Video FPS detected: %.3f", fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = started_at + timedelta(seconds=(frame_index / fps))
            all_items.append(FrameItem(ts=ts, frame=frame))
            frame_index += 1
    finally:
        cap.release()

    if not all_items:
        logger.warning("No frames read from sample video: %s", path)
        return

    duration_sec = len(all_items) / fps
    frames_per_window = max(1, int(round(window_sec * fps)))
    frames_per_stride = max(1, int(round(stride_sec * fps)))
    total_chunks = max(1, math.ceil(len(all_items) / frames_per_stride))

    logger.info(
        "Sample video loaded: frames=%d duration=%.2fs frames_per_window=%d frames_per_stride=%d total_chunks=%d",
        len(all_items),
        duration_sec,
        frames_per_window,
        frames_per_stride,
        total_chunks,
    )

    started_monotonic = time.monotonic()
    try:
        sender.send_heartbeat(started_monotonic)
    except Exception:
        logger.exception("Initial heartbeat failed before sample-video run")

    chunk_num = 0
    start_idx = 0

    while start_idx < len(all_items):
        end_idx = min(len(all_items), start_idx + frames_per_window)
        chunk_items = all_items[start_idx:end_idx]
        chunk_num += 1

        if not chunk_items:
            start_idx += frames_per_stride
            continue

        chunk_start = chunk_items[0].ts
        chunk_end = chunk_items[-1].ts

        logger.info(
            "SAMPLE_CHUNK #%d/%d camera=%s start_idx=%d end_idx=%d raw_frames=%d analyzed=%d span=%.2fs",
            chunk_num,
            total_chunks,
            camera_id,
            start_idx,
            end_idx - 1,
            len(chunk_items),
            len(chunk_items),
            (chunk_end - chunk_start).total_seconds(),
        )

        pipeline.process_frames(camera_id, chunk_items)

        try:
            sender.retry_queued_alerts()
        except Exception:
            logger.exception("Retry queued alerts failed after chunk")

        start_idx += frames_per_stride

    try:
        sender.send_heartbeat(started_monotonic)
    except Exception:
        logger.exception("Final heartbeat failed after sample-video run")

    try:
        sender.retry_queued_alerts()
    except Exception:
        logger.exception("Final retry_queued_alerts failed after sample-video run")

    logger.info("Sample video pipeline run complete: %s", path)
