from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

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
            db_alert = Alert(
                site_id=alert.site_id,
                camera_id=alert.camera_id,
                edge_pc_id=edge_id,
                timestamp=alert.timestamp,
                detections=[d.model_dump(by_alias=True) for d in alert.detections],
                image_path=alert.image_path,
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

