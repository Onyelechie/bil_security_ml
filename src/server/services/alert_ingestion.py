from __future__ import annotations

import re
import shutil
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from bil_time import now_in_winnipeg

from ..config import settings
from ..models.alert import Alert
from ..schemas import AlertCreate

DEFAULT_EDGE_PC_ID = "edge-001"


class AlertPersistenceError(RuntimeError):
    """Raised when an alert cannot be persisted."""


class AlertIngestionService:
    """Shared alert-ingestion workflow for HTTP and WebSocket transports."""

    def __init__(self, default_edge_pc_id: str = DEFAULT_EDGE_PC_ID) -> None:
        self._default_edge_pc_id = default_edge_pc_id

    def ingest(self, db: Session, alert: AlertCreate) -> Alert:
        edge_id = alert.edge_pc_id or self._default_edge_pc_id
        try:
            self._ensure_edge_pc_exists(db, edge_id)
            # Ensure any provided image_path refers to a file under the configured storage.
            # If the alert contains an absolute/local path that exists on the server, copy it
            # into the per-site storage folder and update the path to the relative storage path.
            image_path_val = alert.image_path
            try:
                repo_root = Path(__file__).resolve().parents[3]
                storage_root_setting = Path(settings.image_storage_dir)

                if storage_root_setting.is_absolute():
                    storage_root = storage_root_setting.resolve()
                else:
                    storage_root = (repo_root / storage_root_setting).resolve()

                def _sanitize_part(value: str) -> str:
                    return (
                        re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip()).strip(
                            "_"
                        )
                        or "unknown"
                    )

                if image_path_val and isinstance(image_path_val, str):
                    # ignore URLs
                    if image_path_val.startswith(
                        "http://"
                    ) or image_path_val.startswith("https://"):
                        pass
                    # special-case websocket-saved images: they are prefixed with
                    # `ws://` by the websocket route to indicate they were already
                    # persisted by the websocket storage instance and should not be
                    # copied into the main storage root again.
                    elif image_path_val.startswith("ws://"):
                        src = Path(image_path_val[len("ws://") :])
                        if not src.is_absolute():
                            # if a relative path was returned, resolve it against cwd
                            src = src.resolve()
                        if src.is_file():
                            # store absolute path as-is (no copy)
                            image_path_val = src.as_posix()
                        else:
                            image_path_val = None
                    else:
                        src = Path(image_path_val)
                        # if relative, resolve against repo root
                        if not src.is_absolute():
                            candidate = (repo_root / src).resolve()
                            if candidate.exists():
                                src = candidate

                        if src.is_file():
                            # copy into storage_root/site
                            site_safe = _sanitize_part(alert.site_id)
                            dst_dir = storage_root / site_safe
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            dst_name = src.name
                            dst = dst_dir / dst_name
                            # avoid clobbering
                            if dst.exists():
                                base = dst.stem
                                ext = dst.suffix
                                for i in range(1, 1000):
                                    cand = dst_dir / f"{base}_{i}{ext}"
                                    if not cand.exists():
                                        dst = cand
                                        break
                            shutil.copy2(str(src), str(dst))
                            # Store path relative to the configured storage root when possible.
                            # This keeps default relative-storage behavior unchanged, but also works
                            # when the storage root is an absolute path outside the repo.
                            try:
                                rel_to_storage = dst.relative_to(
                                    storage_root
                                ).as_posix()
                                if storage_root_setting.is_absolute():
                                    image_path_val = rel_to_storage
                                else:
                                    image_path_val = (
                                        storage_root_setting / rel_to_storage
                                    ).as_posix()
                            except Exception:
                                image_path_val = dst.as_posix()
                        else:
                            # file not found; clear the image path to avoid storing external refs
                            image_path_val = None
            except Exception:
                # if anything goes wrong during normalization, fall back to the provided value
                image_path_val = alert.image_path

            db_alert = Alert(
                site_id=alert.site_id,
                camera_id=alert.camera_id,
                edge_pc_id=edge_id,
                timestamp=alert.timestamp,
                received_at=now_in_winnipeg(),
                detections=[d.model_dump(by_alias=True) for d in alert.detections],
                image_path=image_path_val,
            )
            db.add(db_alert)
            db.commit()
            db.refresh(db_alert)
            return db_alert
        except SQLAlchemyError as exc:
            db.rollback()
            raise AlertPersistenceError("Failed to save alert to database") from exc

    def _ensure_edge_pc_exists(self, db: Session, edge_pc_id: str) -> None:
        dialect = db.bind.dialect.name if db.bind is not None else ""
        if dialect == "sqlite":
            sql = (
                "INSERT OR IGNORE INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
                "VALUES (:edge_pc_id, 'unknown', NULL, 'offline')"
            )
        elif dialect in {"postgresql", "psycopg", "psycopg2"}:
            sql = (
                "INSERT INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
                "VALUES (:edge_pc_id, 'unknown', NULL, 'offline') "
                "ON CONFLICT (edge_pc_id) DO NOTHING"
            )
        else:
            sql = (
                "INSERT INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
                "SELECT :edge_pc_id, 'unknown', NULL, 'offline' "
                "WHERE NOT EXISTS (SELECT 1 FROM edge_pcs WHERE edge_pc_id = :edge_pc_id)"
            )
        db.execute(text(sql), {"edge_pc_id": edge_pc_id})
