from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeSettings(BaseSettings):
    """
    Configuration for the Edge Agent.

    If SHARED_STORAGE_ROOT is set, image_path values under that root are safe
    to replay from the offline queue.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
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

    # Frame sampling / scaling for motion detection and window extraction
    analysis_fps: float = 5.0
    frame_width: int = 640
    frame_height: int = 360

    # --- Local motion trigger (cheap) ---
    motion_fps: float = 1.0
    motion_pixel_delta: int = 15
    motion_threshold: float = 0.005
    default_camera_id: str = "1"

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
    detector_vehicle_conf: float = 0.90
    detector_allowed_classes: str = "person"

    # --- Quarantine retention ---
    queue_quarantine_retention_days: int = 7

    # --- Shared storage (optional) ---
    shared_storage_root: str = ""