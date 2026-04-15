from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Callable

from .video.ring_buffer import FrameItem


@dataclass(slots=True)
class EdgeRuntimeSnapshot:
    pipeline_mode: str = "idle"
    stream_state: str = "unknown"
    sender_status: str = "starting"
    ring_buffer_frames: int = 0
    latest_frame_item: FrameItem | None = None
    last_motion_at: datetime | None = None
    last_alert_at: datetime | None = None
    last_error: str | None = None


class EdgeRuntimeState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = EdgeRuntimeSnapshot()
        self._apply_settings_fn: Callable[[dict], list[str]] | None = None
        self._restart_pipeline_fn: Callable[[], dict] | None = None

    def get(self) -> EdgeRuntimeSnapshot:
        with self._lock:
            snap = self._snapshot
            copy = EdgeRuntimeSnapshot(
                pipeline_mode=snap.pipeline_mode,
                stream_state=snap.stream_state,
                sender_status=snap.sender_status,
                ring_buffer_frames=snap.ring_buffer_frames,
                latest_frame_item=snap.latest_frame_item,
                last_motion_at=snap.last_motion_at,
                last_alert_at=snap.last_alert_at,
                last_error=snap.last_error,
            )
        return copy

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._snapshot, key):
                    setattr(self._snapshot, key, value)

    def set_apply_settings_fn(
        self,
        fn: Callable[[dict], list[str]] | None,
    ) -> None:
        with self._lock:
            self._apply_settings_fn = fn

    def apply_settings(self, updates: dict) -> list[str]:
        with self._lock:
            fn = self._apply_settings_fn
        if fn is None:
            return []
        return fn(updates)

    def set_restart_pipeline_fn(
        self,
        fn: Callable[[], dict] | None,
    ) -> None:
        with self._lock:
            self._restart_pipeline_fn = fn

    def request_restart(self) -> dict:
        with self._lock:
            fn = self._restart_pipeline_fn
        if fn is None:
            return {
                "accepted": False,
                "message": "Pipeline restart is not available right now.",
            }
        return fn()
