from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import EdgeSettings

ENV_KEY_MAP: dict[str, str] = {
    "site_id": "SITE_ID",
    "site_name": "SITE_NAME",
    "edge_pc_id": "EDGE_PC_ID",
    "device_id": "DEVICE_ID",
    "server_base_url": "SERVER_BASE_URL",
    "default_camera_id": "DEFAULT_CAMERA_ID",
    "rtsp_url_low": "RTSP_URL_LOW",
    "ring_buffer_seconds": "RING_BUFFER_SECONDS",
    "analysis_fps": "ANALYSIS_FPS",
    "frame_width": "FRAME_WIDTH",
    "frame_height": "FRAME_HEIGHT",
    "enable_tcp_motion": "ENABLE_TCP_MOTION",
    "enable_local_motion": "ENABLE_LOCAL_MOTION",
    "motion_fps": "MOTION_FPS",
    "motion_pixel_delta": "MOTION_PIXEL_DELTA",
    "motion_threshold": "MOTION_THRESHOLD",
    "trigger_cooldown_sec": "TRIGGER_COOLDOWN_SEC",
    "trigger_merge_window_sec": "TRIGGER_MERGE_WINDOW_SEC",
    "incident_quiet_sec": "INCIDENT_QUIET_SEC",
    "incident_max_sec": "INCIDENT_MAX_SEC",
    "incident_tick_interval_sec": "INCIDENT_TICK_INTERVAL_SEC",
    "window_pre_sec": "WINDOW_PRE_SEC",
    "window_post_sec": "WINDOW_POST_SEC",
    "window_target_fps": "WINDOW_TARGET_FPS",
    "window_max_frames": "WINDOW_MAX_FRAMES",
    "window_wait_grace_sec": "WINDOW_WAIT_GRACE_SEC",
    "detector_model": "DETECTOR_MODEL",
    "detector_weights": "DETECTOR_WEIGHTS",
    "detector_person_conf": "DETECTOR_PERSON_CONF",
    "detector_vehicle_conf": "DETECTOR_VEHICLE_CONF",
    "detector_allowed_classes": "DETECTOR_ALLOWED_CLASSES",
    "ptz_global_motion_threshold": "PTZ_GLOBAL_MOTION_THRESHOLD",
    "ptz_consecutive_frames": "PTZ_CONSECUTIVE_FRAMES",
    "ptz_suppress_sec": "PTZ_SUPPRESS_SEC",
    "motion_include_polygons": "MOTION_INCLUDE_POLYGONS",
    "motion_exclude_polygons": "MOTION_EXCLUDE_POLYGONS",
    "heartbeat_interval_sec": "HEARTBEAT_INTERVAL_SEC",
    "update_interval_sec": "UPDATE_INTERVAL_SEC",
    "retry_interval_sec": "RETRY_INTERVAL_SEC",
}


EDITABLE_KEYS = set(ENV_KEY_MAP.keys())


def _env_path() -> Path:
    return Path(".env").resolve()


def _serialize_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _parse_current_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _apply_updates_to_lines(lines: list[str], updates: dict[str, Any]) -> list[str]:
    env_updates = {
        ENV_KEY_MAP[key]: _serialize_env_value(value)
        for key, value in updates.items()
        if key in ENV_KEY_MAP
    }

    seen: set[str] = set()
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue

        key, _old = line.split("=", 1)
        key = key.strip()

        if key in env_updates:
            out.append(f"{key}={env_updates[key]}")
            seen.add(key)
        else:
            out.append(line)

    for key, value in env_updates.items():
        if key not in seen:
            out.append(f"{key}={value}")

    return out


def save_edge_settings(updates: dict[str, Any]) -> dict[str, Any]:
    safe_updates = {k: v for k, v in updates.items() if k in EDITABLE_KEYS}
    if not safe_updates:
        return {"saved_keys": [], "message": "No editable settings provided."}

    # Validate by constructing settings object in memory
    current = EdgeSettings().model_dump()
    merged = {**current, **safe_updates}
    EdgeSettings(**merged)

    env_path = _env_path()
    lines = _parse_current_env_lines(env_path)
    new_lines = _apply_updates_to_lines(lines, safe_updates)
    text = "\n".join(new_lines).rstrip() + "\n"
    env_path.write_text(text, encoding="utf-8")

    return {
        "saved_keys": sorted(safe_updates.keys()),
        "restart_required": True,
        "message": "Settings saved to local .env.",
    }
