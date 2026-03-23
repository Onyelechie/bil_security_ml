import os
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./server.db")

    # Server
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8000))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug_value(cls, value):
        # Accept common deployment shorthands while preserving strictness for unknown values.
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    # CORS (stored as CSV in env)
    cors_origins: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8000",
    )

    # Security
    # Read SECRET_KEY from environment; do not hardcode a production secret here.
    # For development, leave empty and populate `.env` or CI secrets as appropriate.
    secret_key: str = os.getenv("SECRET_KEY", "")
    # Optional admin password used for first-stage admin login (v1). Prefer setting a strong
    # password in env var `ADMIN_PASSWORD` or use a proper user store in production.
    admin_password: str | None = os.getenv("ADMIN_PASSWORD", None)

    # WebSocket alert ingestion
    ws_max_connections: int = int(os.getenv("WS_MAX_CONNECTIONS", 1000))
    ws_alert_queue_size: int = int(os.getenv("WS_ALERT_QUEUE_SIZE", 5000))
    ws_alert_worker_count: int = int(os.getenv("WS_ALERT_WORKER_COUNT", 4))
    ws_max_image_bytes: int = int(os.getenv("WS_MAX_IMAGE_BYTES", 5_000_000))
    ws_image_storage_dir: str = os.getenv(
        "WS_IMAGE_STORAGE_DIR", "storage/ws_alert_images"
    )
    ws_image_retention_hours: int = int(os.getenv("WS_IMAGE_RETENTION_HOURS", 24))
    ws_image_cleanup_interval_hours: int = int(os.getenv("WS_IMAGE_CLEANUP_INTERVAL_HOURS", 24))

    # New unified image storage settings (backwards-compatible with WS_* envs)
    image_storage_dir: str = os.getenv(
        "IMAGE_STORAGE_DIR", os.getenv("WS_IMAGE_STORAGE_DIR", "storage/alert_images")
    )
    image_retention_hours: int = int(
        os.getenv(
            "IMAGE_RETENTION_HOURS",
            os.getenv("WS_IMAGE_RETENTION_HOURS", 24),
        )
    )
    image_cleanup_interval_hours: int = int(
        os.getenv(
            "IMAGE_CLEANUP_INTERVAL_HOURS",
            os.getenv("WS_IMAGE_CLEANUP_INTERVAL_HOURS", 24),
        )
    )
    log_buffer_max_entries: int = int(os.getenv("LOG_BUFFER_MAX_ENTRIES", 5000))

    def __init__(self, **values):
        super().__init__(**values)
        # Runtime safety guards for websocket ingestion limits.
        if self.ws_max_connections < 1:
            raise ValueError("WS_MAX_CONNECTIONS must be >= 1")
        if self.ws_alert_queue_size < 1:
            raise ValueError("WS_ALERT_QUEUE_SIZE must be >= 1")
        if self.ws_alert_worker_count < 1:
            raise ValueError("WS_ALERT_WORKER_COUNT must be >= 1")
        if self.ws_max_image_bytes < 1:
            raise ValueError("WS_MAX_IMAGE_BYTES must be >= 1")
        if not self.ws_image_storage_dir.strip():
            raise ValueError("WS_IMAGE_STORAGE_DIR must not be empty")
        if self.ws_image_retention_hours < 1:
            raise ValueError("WS_IMAGE_RETENTION_HOURS must be >= 1")
        if self.ws_image_cleanup_interval_hours < 1:
            raise ValueError("WS_IMAGE_CLEANUP_INTERVAL_HOURS must be >= 1")
        if not self.image_storage_dir.strip():
            raise ValueError("IMAGE_STORAGE_DIR must not be empty")
        if self.image_retention_hours < 1:
            raise ValueError("IMAGE_RETENTION_HOURS must be >= 1")
        if self.image_cleanup_interval_hours < 1:
            raise ValueError("IMAGE_CLEANUP_INTERVAL_HOURS must be >= 1")
        if self.log_buffer_max_entries < 1:
            raise ValueError("LOG_BUFFER_MAX_ENTRIES must be >= 1")

    def parsed_cors_origins(self) -> List[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]


settings = Settings()
