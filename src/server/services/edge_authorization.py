from __future__ import annotations

from sqlalchemy.orm import Session

from ..models.edge_pc import EdgePC
from .alert_ingestion import DEFAULT_EDGE_PC_ID


def resolve_edge_pc_id(edge_pc_id: str | None) -> str:
    candidate = (edge_pc_id or "").strip()
    if candidate:
        return candidate
    return DEFAULT_EDGE_PC_ID


def is_authorized_edge_pc(db: Session, edge_pc_id: str) -> bool:
    edge = db.query(EdgePC.edge_pc_id).filter_by(edge_pc_id=edge_pc_id).first()
    return edge is not None
