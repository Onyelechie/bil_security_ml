from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeSettings(BaseSettings):
    """
    Configuration for the Edge Agent.
    """

    # Tell pydantic-settings to load environment variables from .env if present.
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Identity ---
    site_id: str = "site_demo"

    # --- Motion events input (BIL software -> Edge Agent) ---
    # Edge agent will listen on this host/port for TCP motion events.
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 8127

    # --- Central server output (Edge Agent -> Area C) ---
    server_base_url: str = "http://127.0.0.1:8000"

    # --- Periodic timers (seconds) ---
    heartbeat_interval_sec: int = 60  # how often we send "I'm alive" to the server
    update_interval_sec: int = 300  # how often we check for model/config updates

    # --- Logging ---
    log_level: str = "INFO"

    # --- Edge HTTP API (Office/Central -> Edge) ---
    edge_http_host: str = "127.0.0.1"
    edge_http_port: int = 8128

    # Identity fields that match what Area C uses
    edge_pc_id: str = "edge_demo"
    site_name: str = "Demo Site"

    # --- Trigger control (rate limit / dedupe) ---
    trigger_cooldown_sec: int = 10
    trigger_merge_window_sec: float = 2.0

    # --- RTSP ingest (low-res stream for analysis) ---
    rtsp_url_low: str = ""  # set in .env
    ring_buffer_seconds: int = 10  # keep last N seconds of frames

    # Frame sampling / scaling for motion detection and window extraction
    analysis_fps: float = 5.0  # frames per second stored in ring buffer
    frame_width: int = 640
    frame_height: int = 360

    # --- Local motion trigger (cheap) ---
    motion_fps: float = 1.0  # how often we check for motion
    motion_pixel_delta: int = 15  # per-pixel diff threshold (0..255)
    motion_threshold: float = 0.005  # ratio of changed pixels required to trigger
    default_camera_id: str = "1"  # used for local motion events to match TCP camera_id

    # --- Incident merging + window extraction ---
    incident_quiet_sec: float = 2.0  # how long it must be “quiet” before we finalize
    incident_max_sec: float = 20.0  # hard cap in storms
    incident_tick_interval_sec: float = 0.2  # how often we check quiet/max finalize

    window_pre_sec: float = 2.0  # pre-roll
    window_post_sec: float = 6.0  # post-roll
    window_target_fps: float = 5.0  # selection sampling rate
    window_max_frames: int = 40  # cap selected frames
    window_wait_grace_sec: float = 1.5  # extra wait time before calling it PARTIAL


settings = EdgeSettings()
