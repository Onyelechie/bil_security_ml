from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..config import settings
from ..db import SessionLocal
from ..models.alert import Alert
from ..schemas import AlertCreate, AlertOut
from ..services.dashboard_events import publish_dashboard_event
from ..services.alert_ingestion import AlertIngestionService, AlertPersistenceError

# This router handles all endpoints related to alerts sent from edge PCs.
# Prefix: /api/alerts
# Tags: alerts (for OpenAPI grouping)
router = APIRouter(prefix="/api/alerts", tags=["alerts"])
alert_ingestion_service = AlertIngestionService()


def get_db() -> Session:
    """Database dependency for getting a session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
def receive_alert(alert: AlertCreate, request: Request, db: Session = Depends(get_db)):
    """
    Endpoint to receive an alert from an edge PC.

    Expected: JSON body with alert details (site_id, camera_id, timestamp, detections, etc.)
    Action: Stores the alert in the database.
    """
    try:
        db_alert = alert_ingestion_service.ingest(db, alert)
        publish_dashboard_event(
            request.app,
            "alert_received",
            {
                "id": db_alert.id,
                "site_id": db_alert.site_id,
                "camera_id": db_alert.camera_id,
                "edge_pc_id": db_alert.edge_pc_id,
                "timestamp": db_alert.timestamp.isoformat() if db_alert.timestamp else None,
            },
        )
        return db_alert
    except AlertPersistenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get("")
def list_alerts(
    db: Session = Depends(get_db),
    limit: int | None = Query(default=None, ge=1, le=1000),
):
    """
    Endpoint to list all alerts, with optional filters (site, camera, date, etc.).

    Action: Will query the database and return a list of alerts (to be implemented).
    """
    try:
        # TODO: Implement alert listing with filters
        alerts = db.query(Alert).all()
        if limit is not None:
            alerts = alerts[-limit:]
        return {"alerts": [AlertOut.model_validate(alert) for alert in alerts]}
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts from database",
        ) from e


@router.get("/{alert_id}/image")
def get_alert_image(alert_id: str, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter_by(id=alert_id).first()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    if not alert.image_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert has no image")

    repo_root = Path(__file__).resolve().parents[3]
    image_path = Path(alert.image_path)
    if not image_path.is_absolute():
        image_path = repo_root / image_path
    image_path = image_path.resolve()

    storage_root = Path(settings.ws_image_storage_dir)
    if not storage_root.is_absolute():
        storage_root = repo_root / storage_root
    storage_root = storage_root.resolve()

    try:
        image_path.relative_to(storage_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image path is outside configured storage directory",
        ) from exc

    if not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found")
    return FileResponse(image_path)
