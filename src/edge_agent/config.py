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
        extra="ignore",  # ignore unknown env vars
    )

    # --- Identity ---
    site_id: str = "site_demo"

    # --- Motion events input (BIL software -> Edge Agent) ---
    # Edge agent will listen on this host/port for TCP motion events.
    tcp_host: str = "172.22.0.5"
    tcp_port: int = 8127

    # --- Central server output (Edge Agent -> Area C) ---
    # Base URL where edge sends alerts and heartbeats.
    server_base_url: str = "http://127.0.0.1:8000"

    # --- Periodic timers (seconds) ---
    heartbeat_interval_sec: int = 60  # how often we send "I'm alive" to the server
    update_interval_sec: int = 300  # how often we check for model/config updates

    # --- Logging ---
    log_level: str = "INFO"

    # --- Edge HTTP API (Office/Central -> Edge) ---
    # Small local API so someone can confirm the edge agent is alive.
    edge_http_host: str = "127.0.0.1"
    edge_http_port: int = 8128

    # Identity fields that match what Area C uses
    edge_pc_id: str = "edge_demo"
    site_name: str = "Demo Site"



# Convenience global settings object.
# This lets other modules do: from edge_agent.config import settings
settings = EdgeSettings()
