from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, UploadFile, File, Form
from fastapi.responses import FileResponse
import asyncio
import json
from datetime import datetime
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


@router.post("/upload", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
async def upload_alert(
    request: Request,
    image: UploadFile = File(...),
    site_id: str = Form(...),
    camera_id: str = Form(...),
    edge_pc_id: str | None = Form(None),
    timestamp: str = Form(...),
    detections: str = Form(...),
    db: Session = Depends(get_db),
):
    """Multipart endpoint to upload an alert image along with metadata.

    Form fields:
    - `site_id`, `camera_id`, `timestamp` (ISO8601), `detections` (JSON array)
    - file field `image` with binary image payload
    """
    try:
        ts = datetime.fromisoformat(timestamp)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp") from exc

    try:
        dets = json.loads(detections)
        if not isinstance(dets, list):
            raise ValueError("detections must be an array")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid detections JSON") from exc

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read uploaded image") from exc

    storage = request.app.state.image_storage
    try:
        saved_path = await asyncio.to_thread(
            storage.save_alert_image,
            site_id=site_id,
            camera_id=camera_id,
            image_bytes=image_bytes,
            edge_pc_id=edge_pc_id,
            detections=dets,
            received_at=ts,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save image") from exc

    # Build AlertCreate and persist
    from ..schemas import AlertCreate as _AlertCreate

    alert_payload = _AlertCreate(
        site_id=site_id,
        camera_id=camera_id,
        edge_pc_id=edge_pc_id,
        timestamp=ts,
        detections=dets,
        image_path=saved_path,
    )

    try:
        db_alert = alert_ingestion_service.ingest(db, alert_payload)
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

    # Accept either the new unified `image_storage_dir` or the legacy `ws_image_storage_dir`.
    storage_dirs = [settings.image_storage_dir, settings.ws_image_storage_dir]
    resolved_roots = []
    for sd in storage_dirs:
        root = Path(sd)
        if not root.is_absolute():
            root = repo_root / root
        try:
            resolved_roots.append(root.resolve())
        except Exception:
            # ignore invalid roots
            continue

    # Ensure image_path is inside at least one of the configured storage roots
    for root in resolved_roots:
        try:
            image_path.relative_to(root)
            break
        except ValueError:
            continue
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image path is outside configured storage directory",
        )

    if not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found")
    return FileResponse(image_path)
