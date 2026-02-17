from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from ..db import SessionLocal
from ..models.edge_pc import EdgePC
from ..schemas import HeartbeatIn, HeartbeatOut
from datetime import datetime

router = APIRouter(prefix="/api/heartbeat", tags=["heartbeat"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=HeartbeatOut, status_code=status.HTTP_201_CREATED)
def heartbeat(payload: HeartbeatIn, db: Session = Depends(get_db)):
    """
    Upsert edge PC heartbeat info.
    """
    edge_pc = db.query(EdgePC).filter_by(edge_pc_id=payload.edge_pc_id).first()
    now = payload.timestamp
    if edge_pc:
        edge_pc.status = payload.status
        edge_pc.last_heartbeat = now
        edge_pc.site_name = payload.site_name
    else:
        edge_pc = EdgePC(
            edge_pc_id=payload.edge_pc_id,
            site_name=payload.site_name,
            status=payload.status,
            last_heartbeat=now
        )
        db.add(edge_pc)
    db.commit()
    db.refresh(edge_pc)
    return {
        "edge_pc_id": edge_pc.edge_pc_id,
        "site_name": edge_pc.site_name,
        "status": edge_pc.status,
        "last_heartbeat": edge_pc.last_heartbeat,
        "message": "Server received heartbeat"
    }
