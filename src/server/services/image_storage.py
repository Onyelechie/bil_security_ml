from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


class ImageStorageError(RuntimeError):
    """Raised when the server cannot persist an incoming image."""


class ImageStorageService:
    """Persists binary image payloads and manages per-site storage.

    This service keeps images under a root directory with one subfolder per
    site. Site-specific settings (like retention) are stored in a small
    JSON file inside the site folder called `.site_settings.json`.
    """

    _SAFE_PART = re.compile(r"[^A-Za-z0-9_-]+")
    _SITE_SETTINGS_NAME = ".site_settings.json"

    def __init__(self, root_dir: str) -> None:
        self._root_dir = Path(root_dir)

    def ensure_ready(self) -> None:
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def ensure_site_ready(self, site_id: str) -> Path:
        """Ensure a folder exists for `site_id` and return its Path."""
        self.ensure_ready()
        safe_site = self._sanitize_part(site_id)
        site_dir = self._root_dir / safe_site
        site_dir.mkdir(parents=True, exist_ok=True)
        return site_dir

    def save_alert_image(
        self,
        *,
        site_id: str,
        camera_id: str,
        image_bytes: bytes,
        edge_pc_id: str | None = None,
        detections: list[dict] | None = None,
        received_at: datetime | None = None,
    ) -> str:
        """Save an alert image inside the site's folder and return the file path.

        Filenames contain a compact summary of the alert for easier inspection:
        <site>_<camera>_<edge>_<timestamp>_<dets>.<ext>
        """
        if received_at is None:
            received_at = datetime.now(timezone.utc)

        site_dir = self.ensure_site_ready(site_id)

        safe_site = self._sanitize_part(site_id)
        safe_camera = self._sanitize_part(camera_id)
        safe_edge = self._sanitize_part(edge_pc_id or "unknown")
        timestamp = received_at.strftime("%Y%m%dT%H%M%S%fZ")
        ext = self._guess_extension(image_bytes)

        det_summary = ""
        try:
            if detections and isinstance(detections, list):
                parts = []
                for d in detections[:3]:
                    cls = d.get("class") or d.get("class_") or "obj"
                    conf = d.get("confidence")
                    if isinstance(conf, (int, float)):
                        parts.append(f"{cls}-{float(conf):.2f}")
                    else:
                        parts.append(f"{cls}")
                det_summary = "+".join(parts)
        except Exception:
            det_summary = ""

        if det_summary:
            # keep filename reasonably sized
            det_summary = det_summary[:120]
            filename = f"{safe_site}_{safe_camera}_{safe_edge}_{timestamp}_{self._sanitize_part(det_summary)}{ext}"
        else:
            filename = f"{safe_site}_{safe_camera}_{safe_edge}_{timestamp}{ext}"

        path = site_dir / filename
        try:
            path.write_bytes(image_bytes)
        except OSError as exc:
            raise ImageStorageError("Failed to persist image payload") from exc

        return path.as_posix()

    def cleanup_site(
        self,
        *,
        site_id: str,
        hours: int,
        now: datetime | None = None,
    ) -> int:
        """Remove files older than `hours` from a single site's folder."""
        if hours < 1:
            raise ValueError("hours must be >= 1")
        if now is None:
            now = datetime.now(timezone.utc)

        site_dir = self.ensure_site_ready(site_id)
        cutoff = now - timedelta(hours=hours)
        removed = 0

        for path in site_dir.iterdir():
            if not path.is_file():
                continue
            if path.name == self._SITE_SETTINGS_NAME:
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue

        return removed

    def cleanup_all(self, *, default_hours: int, now: datetime | None = None) -> dict[str, int]:
        """Run cleanup for all site folders, returning a mapping of site->removed count.

        Each site may define its own retention in `.site_settings.json`; if present
        that value will be used instead of `default_hours`.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        self.ensure_ready()
        results: dict[str, int] = {}
        for site_dir in self._root_dir.iterdir():
            if not site_dir.is_dir():
                continue
            site_name = site_dir.name
            site_settings = self._read_site_settings(site_dir)
            hours = site_settings.get("image_retention_hours") if site_settings else None
            try:
                use_hours = int(hours) if hours and int(hours) >= 1 else default_hours
            except Exception:
                use_hours = default_hours
            removed = self.cleanup_site(site_id=site_name, hours=use_hours, now=now)
            results[site_name] = removed
        return results

    def _read_site_settings(self, site_dir: Path) -> dict | None:
        settings_file = site_dir / self._SITE_SETTINGS_NAME
        if not settings_file.is_file():
            return None
        try:
            import json

            return json.loads(settings_file.read_text(encoding="utf8"))
        except Exception:
            return None

    @classmethod
    def _sanitize_part(cls, value: str) -> str:
        cleaned = cls._SAFE_PART.sub("_", (value or "").strip())
        cleaned = cleaned.strip("_")
        return cleaned or "unknown"

    @staticmethod
    def _guess_extension(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".bin"
