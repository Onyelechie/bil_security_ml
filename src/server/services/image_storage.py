from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re


class ImageStorageError(RuntimeError):
    """Raised when the server cannot persist an incoming image."""


class ImageStorageService:
    """Persists binary image payloads for websocket alerts."""

    _SAFE_PART = re.compile(r"[^A-Za-z0-9_-]+")

    def __init__(self, root_dir: str) -> None:
        self._root_dir = Path(root_dir)

    def ensure_ready(self) -> None:
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def save_alert_image(
        self,
        *,
        site_id: str,
        camera_id: str,
        image_bytes: bytes,
        received_at: datetime | None = None,
    ) -> str:
        if received_at is None:
            received_at = datetime.now(timezone.utc)

        safe_site = self._sanitize_part(site_id)
        safe_camera = self._sanitize_part(camera_id)
        timestamp = received_at.strftime("%Y%m%dT%H%M%S%fZ")
        ext = self._guess_extension(image_bytes)
        filename = f"{safe_site}_{safe_camera}_{timestamp}{ext}"
        path = self._root_dir / filename

        try:
            path.write_bytes(image_bytes)
        except OSError as exc:
            raise ImageStorageError("Failed to persist websocket image payload") from exc

        return path.as_posix()

    @classmethod
    def _sanitize_part(cls, value: str) -> str:
        cleaned = cls._SAFE_PART.sub("_", value.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "unknown"

    @staticmethod
    def _guess_extension(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\xFF\xD8\xFF"):
            return ".jpg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".bin"

