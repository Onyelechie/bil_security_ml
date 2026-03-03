from __future__ import annotations

import asyncio
import logging
import subprocess  # nosec B404
from datetime import datetime, timezone
from urllib.parse import urlparse

import imageio_ffmpeg
import numpy as np

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
        logger.info(
            "RTSP reader started (stream=%s)",
            self._stream_label(self._cfg.rtsp_url_low),
        )

    async def stop(self) -> None:
        self._stop.set()

        task = self._task
        self._task = None

        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._kill_proc()
        logger.info("RTSP reader stopped")

    async def _run_loop(self) -> None:
        w = int(self._cfg.frame_width)
        h = int(self._cfg.frame_height)
        fps = float(self._cfg.analysis_fps)
        frame_bytes = w * h  # grayscale

        stream = self._stream_label(self._cfg.rtsp_url_low)

        attempt = 0
        backoff_s = 2.0  # exponential backoff up to 10s

        while not self._stop.is_set():
            attempt += 1
            first_frame = True

            try:
                self._start_ffmpeg(w=w, h=h, fps=fps)
                if self._proc is None or self._proc.stdout is None:
                    raise RuntimeError("ffmpeg process failed to start (stdout missing)")

                logger.info("RTSP connect attempt=%d stream=%s", attempt, stream)

                while not self._stop.is_set():
                    # Give extra time for the *first* decoded frame (keyframe wait), then tighter stall checks.
                    timeout_s = 30.0 if first_frame else 5.0

                    try:
                        buf = await asyncio.wait_for(
                            asyncio.to_thread(self._proc.stdout.read, frame_bytes),
                            timeout=timeout_s,
                        )
                    except asyncio.TimeoutError:
                        # More user-friendly classification
                        if first_frame:
                            raise RuntimeError(f"STARTUP_TIMEOUT: no first frame within {timeout_s:.0f}s")
                        raise RuntimeError(f"STALL: no frame bytes for {timeout_s:.0f}s")

                    if not buf or len(buf) < frame_bytes:
                        raise RuntimeError("DISCONNECT: ffmpeg stream ended")

                    frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w))
                    self._ring.push(datetime.now(timezone.utc), frame)

                    if first_frame:
                        logger.info(
                            "RTSP first frame received stream=%s (attempt=%d)",
                            stream,
                            attempt,
                        )
                        first_frame = False
                        # reset backoff after success
                        backoff_s = 2.0

            except Exception as exc:
                stderr_short = self._read_stderr_if_exited_short()

                msg = str(exc)
                if msg.startswith(("STARTUP_TIMEOUT", "STALL", "DISCONNECT")):
                    code = msg.split(":", 1)[0]
                    detail = msg.split(":", 1)[1].strip() if ":" in msg else ""
                else:
                    code = "ERROR"
                    detail = msg

                logger.warning(
                    "RTSP %s: %s stream=%s (attempt=%d). Retrying in %.1fs",
                    code,
                    detail,
                    stream,
                    attempt,
                    backoff_s,
                )
                if stderr_short:
                    logger.warning("ffmpeg stderr (short): %s", stderr_short)

                self._kill_proc()
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 10.0)

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
        logger.debug("Starting ffmpeg: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # nosec B603

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        from contextlib import suppress

        with suppress(Exception):
            self._proc.kill()
        self._proc = None

    @staticmethod
    def _stream_label(url: str) -> str:
        """
        Return a safe label like: 192.168.2.100:8554/Streaming/Channels/102/
        (no username/password)
        """
        try:
            p = urlparse(url)
            host = p.hostname or "unknown-host"
            port = p.port or 554
            path = p.path or "/"
            return f"{host}:{port}{path}"
        except Exception:
            return "unknown-stream"

    def _read_stderr_if_exited_short(self) -> str:
        """
        Read a short stderr snippet only if ffmpeg has exited.
        Prevents blocking on stderr when process is still running.
        """
        if not self._proc or not self._proc.stderr:
            return ""
        if self._proc.poll() is None:
            return ""
        try:
            txt = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
        if not txt:
            return ""
        # keep only first line (demo-friendly)
        return txt.splitlines()[0][:300]
