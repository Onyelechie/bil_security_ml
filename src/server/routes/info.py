from fastapi import APIRouter, Request
from ..config import settings
from ..db import SessionLocal
from ..models.edge_pc import EdgePC

router = APIRouter(prefix="/api", tags=["info"])


@router.get("/server-info")
def server_info(request: Request):
    # basic server info and edge count
    db = SessionLocal()
    try:
        edges = db.query(EdgePC).count()
    finally:
        db.close()
    return {
        "host": settings.host,
        "port": settings.port,
        "debug": settings.debug,
        "edge_count": edges,
    }
