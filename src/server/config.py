from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Database
    database_url: str = "sqlite:///./server.db"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    auto_apply_migrations: bool = True

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug_value(cls, value):
        # Accept common deployment shorthands while preserving strictness for unknown values.
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "release",
                "prod",
                "production",
                "false",
                "0",
                "no",
                "off",
            }:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    # CORS (stored as CSV in env)
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Security
    # Read SECRET_KEY from environment; do not hardcode a production secret here.
    # For development, leave empty and populate `.env` or CI secrets as appropriate.
    secret_key: str = ""
    # Optional admin password used for first-stage admin login (v1). Prefer setting a strong
    # password in env var `ADMIN_PASSWORD` or use a proper user store in production.
    admin_password: str | None = None

    # WebSocket alert ingestion
    ws_max_connections: int = 1000
    ws_alert_queue_size: int = 5000
    ws_alert_worker_count: int = 4
    ws_max_image_bytes: int = 5_000_000
    ws_image_storage_dir: str = "storage/ws_alert_images"
    ws_image_retention_hours: int = 24
    ws_image_cleanup_interval_hours: int = 24

    # New unified image storage settings (backwards-compatible with WS_* envs)
    image_storage_dir: str = "storage/alert_images"
    image_retention_hours: int = 24
    image_cleanup_interval_hours: int = 24
    log_buffer_max_entries: int = 5000

    @model_validator(mode="after")
    def _apply_legacy_image_fallbacks(self):
        if (
            "image_storage_dir" not in self.model_fields_set
            and "ws_image_storage_dir" in self.model_fields_set
        ):
            self.image_storage_dir = self.ws_image_storage_dir
        if (
            "image_retention_hours" not in self.model_fields_set
            and "ws_image_retention_hours" in self.model_fields_set
        ):
            self.image_retention_hours = self.ws_image_retention_hours
        if (
            "image_cleanup_interval_hours" not in self.model_fields_set
            and "ws_image_cleanup_interval_hours" in self.model_fields_set
        ):
            self.image_cleanup_interval_hours = self.ws_image_cleanup_interval_hours
        return self

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
