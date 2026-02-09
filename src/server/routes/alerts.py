
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..db import SessionLocal
from ..models.alert import Alert
from ..schemas import AlertCreate, AlertOut

# This router handles all endpoints related to alerts sent from edge PCs.
# Prefix: /api/alerts
# Tags: alerts (for OpenAPI grouping)
router = APIRouter(prefix="/api/alerts", tags=["alerts"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
def receive_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    """
    Endpoint to receive an alert from an edge PC.
    Expected: JSON body with alert details (site_id, camera_id, timestamp, detections, etc.)
    Action: Stores the alert in the database.
    """
    db_alert = Alert(
        site_id=alert.site_id,
        camera_id=alert.camera_id,
        timestamp=alert.timestamp,
        detections=[d.dict(by_alias=True) for d in alert.detections],
        image_path=alert.image_path
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

@router.get("")
def list_alerts():
    """
    Endpoint to list all alerts, with optional filters (site, camera, date, etc.).
    Action: Will query the database and return a list of alerts (to be implemented).
    """
    # TODO: Implement alert listing
    return {"alerts": []}
