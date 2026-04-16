from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import EdgeSettings
from .runtime_state import EdgeRuntimeState
from .sender import ServerSender
from .settings_store import load_effective_settings, save_edge_settings


class HealthOut(BaseModel):
    status: str
    time_utc: datetime

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "ok", "time_utc": "2026-02-18T12:00:00Z"}]
        }
    }


class HeartbeatOut(BaseModel):
    edge_pc_id: str
    site_name: str
    status: str
    time_utc: datetime
    uptime_seconds: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "edge_pc_id": "edge-001",
                    "site_name": "Warehouse 1",
                    "status": "online",
                    "time_utc": "2026-02-18T12:00:00Z",
                    "uptime_seconds": 42,
                }
            ]
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def _ui_dir() -> Path:
    return Path(__file__).resolve().parent / "ui"


def _encode_preview_jpeg_b64(frame) -> str | None:
    if frame is None:
        return None
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 70],
    )
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _settings_payload(cfg: EdgeSettings) -> dict:
    return {
        "identity": {
            "site_id": cfg.site_id,
            "site_name": cfg.site_name,
            "edge_pc_id": cfg.edge_pc_id,
            "device_id": cfg.device_id,
            "server_base_url": cfg.server_base_url,
            "default_camera_id": cfg.default_camera_id,
        },
        "stream": {
            "rtsp_url_low": cfg.rtsp_url_low,
            "ring_buffer_seconds": cfg.ring_buffer_seconds,
            "analysis_fps": cfg.analysis_fps,
            "preview_fps": cfg.preview_fps,
            "frame_width": cfg.frame_width,
            "frame_height": cfg.frame_height,
        },
        "motion": {
            "enable_tcp_motion": cfg.enable_tcp_motion,
            "enable_local_motion": cfg.enable_local_motion,
            "motion_fps": cfg.motion_fps,
            "motion_pixel_delta": cfg.motion_pixel_delta,
            "motion_threshold": cfg.motion_threshold,
            "trigger_cooldown_sec": cfg.trigger_cooldown_sec,
            "trigger_merge_window_sec": cfg.trigger_merge_window_sec,
        },
        "incidents": {
            "incident_quiet_sec": cfg.incident_quiet_sec,
            "incident_max_sec": cfg.incident_max_sec,
            "incident_tick_interval_sec": cfg.incident_tick_interval_sec,
            "window_pre_sec": cfg.window_pre_sec,
            "window_post_sec": cfg.window_post_sec,
            "window_target_fps": cfg.window_target_fps,
            "window_max_frames": cfg.window_max_frames,
            "window_wait_grace_sec": cfg.window_wait_grace_sec,
        },
        "detection": {
            "detector_model": cfg.detector_model,
            "detector_weights": cfg.detector_weights,
            "detector_person_conf": cfg.detector_person_conf,
            "detector_vehicle_conf": cfg.detector_vehicle_conf,
            "detector_allowed_classes": cfg.detector_allowed_classes,
        },
        "ptz": {
            "ptz_global_motion_threshold": cfg.ptz_global_motion_threshold,
            "ptz_consecutive_frames": cfg.ptz_consecutive_frames,
            "ptz_suppress_sec": cfg.ptz_suppress_sec,
        },
        "zones": {
            "motion_include_polygons": cfg.motion_include_polygons,
            "motion_exclude_polygons": cfg.motion_exclude_polygons,
        },
        "timers": {
            "heartbeat_interval_sec": cfg.heartbeat_interval_sec,
            "update_interval_sec": cfg.update_interval_sec,
            "retry_interval_sec": cfg.retry_interval_sec,
        },
    }


def create_app(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState | None = None,
) -> FastAPI:
    """
    Create the Edge Agent HTTP API app.
    """
    runtime_state = runtime_state or EdgeRuntimeState()

    app = FastAPI(
        title="BIL Security ML - Edge Agent API",
        version="0.2.0",
        description=(
            "Edge-side health endpoints for install/debug and office connectivity checks."
        ),
        lifespan=lifespan,
    )
    app.state.sender = sender
    app.state.runtime_state = runtime_state

    ui_dir = _ui_dir()
    assets_dir = ui_dir / "assets"
    if assets_dir.exists():
        app.mount(
            "/edge-console/assets",
            StaticFiles(directory=str(assets_dir)),
            name="edge_console_assets",
        )

    started_monotonic = time.monotonic()

    @app.get("/")
    def root():
        index_path = ui_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"status": "edge console running"}

    @app.get("/health", response_model=HealthOut, tags=["health"])
    def health() -> HealthOut:
        return HealthOut(status="ok", time_utc=datetime.now(timezone.utc))

    @app.get("/heartbeat", response_model=HeartbeatOut, tags=["health"])
    def heartbeat() -> HeartbeatOut:
        uptime = int(time.monotonic() - started_monotonic)
        return HeartbeatOut(
            edge_pc_id=cfg.edge_pc_id,
            site_name=cfg.site_name,
            status=sender.get_status(),
            time_utc=datetime.now(timezone.utc),
            uptime_seconds=uptime,
        )

    @app.get("/api/runtime")
    def runtime():
        snap = runtime_state.get()
        latest_ts = snap.latest_frame_item.ts if snap.latest_frame_item else None
        return {
            "pipeline_mode": snap.pipeline_mode,
            "stream_state": snap.stream_state,
            "sender_status": snap.sender_status,
            "ring_buffer_frames": snap.ring_buffer_frames,
            "latest_frame_time_utc": latest_ts,
            "last_motion_at": snap.last_motion_at,
            "last_alert_at": snap.last_alert_at,
            "last_error": snap.last_error,
        }

    @app.get("/api/settings")
    def settings():
        fresh_cfg = load_effective_settings()
        return _settings_payload(fresh_cfg)

    @app.get("/api/preview")
    def preview():
        snap = runtime_state.get()
        fresh_cfg = load_effective_settings()

        if snap.latest_frame_item is None:
            return JSONResponse(
                {
                    "available": False,
                    "message": "No preview frame available yet.",
                }
            )

        image_b64 = _encode_preview_jpeg_b64(snap.latest_frame_item.frame)
        if not image_b64:
            raise HTTPException(
                status_code=500, detail="Failed to encode preview image"
            )

        return {
            "available": True,
            "captured_at": snap.latest_frame_item.ts,
            "image_jpeg_b64": image_b64,
            "include_polygons": fresh_cfg.motion_include_polygons,
            "exclude_polygons": fresh_cfg.motion_exclude_polygons,
        }

    @app.put("/api/settings")
    def update_settings(payload: dict):
        try:
            result = save_edge_settings(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        applied_keys = runtime_state.apply_settings(payload)
        saved_keys = result.get("saved_keys", [])
        restart_required_keys = [key for key in saved_keys if key not in applied_keys]

        result["applied_keys"] = applied_keys
        result["restart_required_keys"] = restart_required_keys
        result["restart_required"] = bool(restart_required_keys)

        if applied_keys and restart_required_keys:
            result["message"] = (
                "Some settings were applied immediately. "
                "Some saved settings still require restart."
            )
        elif applied_keys:
            result["message"] = "Settings applied immediately and saved to local .env."
        else:
            result["message"] = "Settings saved to local .env. Restart required."

        return result

    @app.put("/api/zones")
    def update_zones(payload: dict):
        updates = {
            "motion_include_polygons": payload.get("motion_include_polygons", []),
            "motion_exclude_polygons": payload.get("motion_exclude_polygons", []),
        }

        try:
            result = save_edge_settings(updates)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        applied_keys = runtime_state.apply_settings(updates)
        result["applied_keys"] = applied_keys
        result["restart_required_keys"] = [
            key for key in result.get("saved_keys", []) if key not in applied_keys
        ]
        result["restart_required"] = bool(result["restart_required_keys"])
        result["message"] = (
            "Zones applied immediately and saved to local .env."
            if applied_keys
            else "Zones saved to local .env. Restart required."
        )
        return result

    @app.post("/api/runtime/restart")
    def restart_runtime():
        return runtime_state.request_restart()

    return app
