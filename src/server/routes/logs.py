from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..schemas import ServerLogListOut
from ..services.log_buffer import InMemoryLogBuffer
from .auth import get_current_admin

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("", response_model=ServerLogListOut)
def list_logs(
    request: Request,
    current_admin: str = Depends(get_current_admin),
    limit: int = Query(200, ge=1, le=1000),
    after_id: int | None = Query(default=None, ge=1),
    level: str | None = Query(default=None),
):
    log_buffer: InMemoryLogBuffer | None = getattr(
        request.app.state, "log_buffer", None
    )
    if log_buffer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Log buffer unavailable",
        )

    logs = log_buffer.list_entries(limit=limit, after_id=after_id, level=level)
    return {"logs": logs, "latest_id": log_buffer.latest_id}
