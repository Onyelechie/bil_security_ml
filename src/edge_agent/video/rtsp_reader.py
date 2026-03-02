from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone

import numpy as np
import imageio_ffmpeg

from ..config import EdgeSettings
from .ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class RtspReader:
    def __init__(self, cfg: EdgeSettings, ring: RingBuffer) -> None:
        self._cfg = cfg
        self._ring = ring
        self._proc: subprocess.Popen[bytes] | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self._cfg.rtsp_url_low:
            logger.warning("RTSP reader not started: RTSP_URL_LOW is not set.")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="rtsp-reader")
        logger.info("RTSP reader started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
        self._kill_proc()
        logger.info("RTSP reader stopped")

    async def _run_loop(self) -> None:
        w = int(self._cfg.frame_width)
        h = int(self._cfg.frame_height)
        fps = float(self._cfg.analysis_fps)
        frame_bytes = w * h  # grayscale

        while not self._stop.is_set():
            try:
                self._start_ffmpeg(w=w, h=h, fps=fps)
                assert self._proc is not None and self._proc.stdout is not None

                while not self._stop.is_set():
                    buf = self._proc.stdout.read(frame_bytes)
                    if not buf or len(buf) < frame_bytes:
                        raise RuntimeError("ffmpeg stream ended")

                    frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w))
                    self._ring.push(datetime.now(timezone.utc), frame)

            except Exception as exc:
                logger.warning("RTSP read error: %s (restarting in 2s)", exc)
                self._kill_proc()
                await asyncio.sleep(2)

    def _start_ffmpeg(self, *, w: int, h: int, fps: float) -> None:
        self._kill_proc()
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        url = self._cfg.rtsp_url_low

        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            url,
            "-vf",
            f"fps={fps},scale={w}:{h},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
        except Exception:
            pass
        self._proc = None
