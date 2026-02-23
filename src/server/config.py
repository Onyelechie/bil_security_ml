
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./server.db")

    # Server
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8000))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # CORS
    cors_origins: List[str] = []

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "development-secret-key-change-in-production")

    def __init__(self, **values):
        super().__init__(**values)
        # Parse CORS_ORIGINS from env (comma-separated string)
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            self.cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        else:
            # fallback to default dev origins if not set
            self.cors_origins = [
                "http://localhost:3000",
                "http://localhost:8000"
            ]

settings = Settings()