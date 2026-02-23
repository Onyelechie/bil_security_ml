from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Database
    database_url: str = "sqlite:///./server.db"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000"
    ]  # Restrict to known local development origins

    # Security
    secret_key: str = "development-secret-key-change-in-production"


settings = Settings()