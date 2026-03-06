from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..models.edge_pc import EdgePC
from ..schemas import EdgePCStatusListOut, EdgePCStatusOut, HeartbeatIn, HeartbeatOut
from ..services.dashboard_events import publish_dashboard_event

router = APIRouter(prefix="/api/heartbeat", tags=["heartbeat"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", response_model=HeartbeatOut, status_code=status.HTTP_201_CREATED)
def heartbeat(payload: HeartbeatIn, request: Request, db: Session = Depends(get_db)):
    """
    Upsert edge PC heartbeat info.
    """
    edge_pc = db.query(EdgePC).filter_by(edge_pc_id=payload.edge_pc_id).first()
    now = datetime.now(timezone.utc)
    if edge_pc:
        edge_pc.status = payload.status
        edge_pc.last_heartbeat = now
        edge_pc.site_name = payload.site_name
    else:
        edge_pc = EdgePC(
            edge_pc_id=payload.edge_pc_id,
            site_name=payload.site_name,
            status=payload.status,
            last_heartbeat=now,
        )
        db.add(edge_pc)
    db.commit()
    db.refresh(edge_pc)
    publish_dashboard_event(
        request.app,
        "heartbeat_received",
        {
            "edge_pc_id": edge_pc.edge_pc_id,
            "site_name": edge_pc.site_name,
            "status": edge_pc.status,
            "last_heartbeat": edge_pc.last_heartbeat.isoformat() if edge_pc.last_heartbeat else None,
        },
    )
    return {
        "edge_pc_id": edge_pc.edge_pc_id,
        "site_name": edge_pc.site_name,
        "status": edge_pc.status,
        "last_heartbeat": edge_pc.last_heartbeat,
        "message": "Server received heartbeat",
    }


@router.get("", response_model=EdgePCStatusListOut)
def list_edge_pcs(db: Session = Depends(get_db)):
    edges = db.query(EdgePC).all()

    def _sort_key(edge: EdgePC) -> datetime:
        ts = edge.last_heartbeat
        if ts is None:
            return datetime.min
        if ts.tzinfo is not None:
            return ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    ordered = sorted(
        edges,
        key=_sort_key,
        reverse=True,
    )
    return {"edges": [EdgePCStatusOut.model_validate(edge) for edge in ordered]}
