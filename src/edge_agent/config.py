from __future__ import annotations

import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

EDGE_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class EdgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(EDGE_ENV_FILE),
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # --- Identity ---
    site_id: str = "site_demo"

    # --- Motion events input (BIL software -> Edge Agent) ---
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 8127

    # --- Central server output (Edge Agent -> Area C) ---
    server_base_url: str = "http://127.0.0.1:8000"

    # --- Periodic timers (seconds) ---
    heartbeat_interval_sec: int = 60
    update_interval_sec: int = 300
    retry_interval_sec: int = 30

    # --- Logging ---
    log_level: str = "INFO"

    # --- Edge HTTP API (Office/Central -> Edge) ---
    edge_http_host: str = "127.0.0.1"
    edge_http_port: int = 8128

    # Identity fields that match what Area C uses
    edge_pc_id: str = "edge_demo"
    site_name: str = "Demo Site"
    device_id: str | None = None
    device_private_key_b64: str = ""

    # --- Trigger source selection ---
    enable_tcp_motion: bool = True
    enable_local_motion: bool = False

    # --- Trigger control (rate limit / dedupe) ---
    trigger_cooldown_sec: int = 10
    trigger_merge_window_sec: float = 2.0

    # --- RTSP ingest (low-res stream for analysis) ---
    rtsp_url_low: str = ""
    ring_buffer_seconds: int = 25

    preview_fps: float = 8.0

    # Frame sampling / scaling for motion detection and window extraction
    analysis_fps: float = 5.0
    frame_width: int = 640
    frame_height: int = 360

    # --- Local motion trigger (cheap) ---
    motion_fps: float = 1.0
    motion_pixel_delta: int = 15
    motion_threshold: float = 0.005
    default_camera_id: str = "1"

    # --- PTZ / camera-motion suppression ---
    # If a very large fraction of the full frame changes, it is likely
    # camera movement rather than scene motion.
    ptz_global_motion_threshold: float = 0.35
    ptz_consecutive_frames: int = 2
    ptz_suppress_sec: float = 3.0

    # --- Incident merging + window extraction ---
    incident_quiet_sec: float = 2.0
    incident_max_sec: float = 12.0
    incident_tick_interval_sec: float = 0.2

    window_pre_sec: float = 1.5
    window_post_sec: float = 4.0
    window_target_fps: float = 5.0
    window_max_frames: int = 40
    window_wait_grace_sec: float = 1.5

    # --- Offline alert queue ---
    offline_queue_dir: str = "storage/offline_queue"

    # --- Detector Selection ---
    detector_model: str = "YOLOv8-Small"
    detector_weights: str | None = None
    detector_person_conf: float = 0.40
    detector_vehicle_conf: float = 0.50
    detector_allowed_classes: str = "person,vehicle"

    # --- Quarantine retention ---
    queue_quarantine_retention_days: int = 7

    # --- Shared storage (optional) ---
    shared_storage_root: str = ""

    # --- Direct CCTV sample runner ---
    sample_camera_id: str = "1"
    sample_window_sec: float = 3.0
    sample_stride_sec: float = 2.0
    sample_target_fps: float = 5.0
    sample_max_frames: int = 30

    # --- Motion zones / masking ---
    # Polygons use normalized coordinates in the range 0..1:
    # [[[x1, y1], [x2, y2], [x3, y3], ...], ...]
    motion_include_polygons: list[list[list[float]]] = []
    motion_exclude_polygons: list[list[list[float]]] = []

    @field_validator(
        "motion_include_polygons", "motion_exclude_polygons", mode="before"
    )
    @classmethod
    def parse_polygon_json(cls, value):
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            return json.loads(value)
        return value
