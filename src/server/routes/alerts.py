import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query,
                     Request, UploadFile, status)
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from bil_time import ensure_winnipeg

from ..config import settings
from ..db import SessionLocal
from ..models.alert import Alert
from ..schemas import AlertCreate, AlertOut
from ..services.alert_ingestion import (AlertIngestionService,
                                        AlertPersistenceError)
from ..services.dashboard_events import publish_dashboard_event
from ..services.device_auth import require_signed_device
from ..services.edge_authorization import (is_authorized_edge_pc,
                                           resolve_edge_pc_id)

# This router handles all endpoints related to alerts sent from edge PCs.
# Prefix: /api/alerts
# Tags: alerts (for OpenAPI grouping)
router = APIRouter(prefix="/api/alerts", tags=["alerts"])
alert_ingestion_service = AlertIngestionService()
logger = logging.getLogger(__name__)


def _safe_identity(value: str | None) -> str:
    if not value:
        return "<missing>"
    return value[:64]


def _ensure_edge_sender_authorized(db: Session, edge_pc_id: str | None) -> str:
    resolved_edge_id = resolve_edge_pc_id(edge_pc_id)
    if not is_authorized_edge_pc(db, resolved_edge_id):
        logger.warning(
            "Rejecting alert ingestion from unregistered edge_pc_id=%s",
            _safe_identity(resolved_edge_id),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Edge PC is not authorized to submit alerts",
        )
    return resolved_edge_id


def _alert_sort_key(alert: Alert, sort_by: str) -> datetime:
    primary = alert.received_at if sort_by == "received_at" else alert.timestamp
    secondary = alert.timestamp if sort_by == "received_at" else alert.received_at
    candidate = primary or secondary
    if candidate is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return ensure_winnipeg(candidate)


def _persist_alert(alert: AlertCreate) -> AlertOut:
    db = SessionLocal()
    try:
        db_alert = alert_ingestion_service.ingest(db, alert)
        return AlertOut.model_validate(db_alert)
    finally:
        db.close()


def get_db() -> Session:
    """Database dependency for getting a session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
async def receive_alert(
    alert: AlertCreate, request: Request, db: Session = Depends(get_db)
):
    """
    Endpoint to receive an alert from an edge PC.

    Expected: JSON body with alert details (site_id, camera_id, timestamp, detections, etc.)
    Action: Stores the alert in the database.
    """
    resolved_edge_id = resolve_edge_pc_id(alert.edge_pc_id)
    device_id = request.headers.get("X-Device-Id")
    signature = request.headers.get("X-Device-Signature")
    body = await request.body()
    require_signed_device(
        db,
        device_id=device_id,
        signature_b64=signature,
        message=body,
        expected_edge_pc_id=resolved_edge_id,
    )

    _ensure_edge_sender_authorized(db, alert.edge_pc_id)

    try:
        alert_out = await asyncio.to_thread(_persist_alert, alert)
        publish_dashboard_event(
            request.app,
            "alert_received",
            alert_out.model_dump(mode="json", by_alias=True),
        )
        return alert_out
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp"
        ) from exc

    try:
        dets = json.loads(detections)
        if not isinstance(dets, list):
            raise ValueError("detections must be an array")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid detections JSON"
        ) from exc

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded image",
        ) from exc

    storage = request.app.state.image_storage
    resolved_edge_id = resolve_edge_pc_id(edge_pc_id)
    device_id = request.headers.get("X-Device-Id")
    signature = request.headers.get("X-Device-Signature")
    sha = hashlib.sha256(image_bytes).hexdigest()
    canonical = f"{site_id}|{camera_id}|{edge_pc_id or ''}|{timestamp}|{sha}".encode(
        "utf-8"
    )
    require_signed_device(
        db,
        device_id=device_id,
        signature_b64=signature,
        message=canonical,
        expected_edge_pc_id=resolved_edge_id,
    )

    _ensure_edge_sender_authorized(db, edge_pc_id)

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save image",
        ) from exc

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
        alert_out = await asyncio.to_thread(_persist_alert, alert_payload)
        publish_dashboard_event(
            request.app,
            "alert_received",
            alert_out.model_dump(mode="json", by_alias=True),
        )
        return alert_out
    except AlertPersistenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get("")
def list_alerts(
    db: Session = Depends(get_db),
    sort_by: Literal["received_at", "timestamp"] = Query(
        default="received_at",
        description="Sort alerts by either the server received time or the alert timestamp.",
    ),
    limit: int | None = Query(default=None, ge=1, le=1000),
):
    """
    Endpoint to list all alerts, with optional filters (site, camera, date, etc.).

    Action: Will query the database and return a list of alerts (to be implemented).
    """
    try:
        # TODO: Implement alert listing with filters
        alerts = db.query(Alert).all()
        alerts = sorted(alerts, key=lambda alert: _alert_sort_key(alert, sort_by), reverse=True)
        if limit is not None:
            alerts = alerts[:limit]
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
        )
    if not alert.image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Alert has no image"
        )

    repo_root = Path(__file__).resolve().parents[3]
    stored_path = Path(alert.image_path)

    # Accept either the new unified `image_storage_dir` or the legacy `ws_image_storage_dir`.
    storage_dirs = [settings.image_storage_dir, settings.ws_image_storage_dir]
    resolved_roots = []
    for sd in storage_dirs:
        root = Path(sd)
        if not root.is_absolute():
            root = repo_root / root
        try:
            resolved_roots.append(root.resolve())
        except OSError:
            continue

    candidate_paths = []

    if stored_path.is_absolute():
        try:
            candidate_paths.append(stored_path.resolve())
        except OSError:
            pass
    else:
        # First try resolving relative to each configured storage root.
        for root in resolved_roots:
            try:
                candidate_paths.append((root / stored_path).resolve())
            except OSError:
                continue

        # Then try repo-root relative for backward compatibility.
        try:
            candidate_paths.append((repo_root / stored_path).resolve())
        except OSError:
            pass

    image_path = None
    for candidate in candidate_paths:
        for root in resolved_roots:
            try:
                candidate.relative_to(root)
                image_path = candidate
                break
            except ValueError:
                continue
        if image_path is not None:
            break

    if image_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image path is outside configured storage directory",
        )

    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found"
        )

    return FileResponse(image_path)
    