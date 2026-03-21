from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["dashboard"])

_STATIC_ROOT = Path(__file__).resolve().parent.parent / "static" / "dashboard"


@router.get("/dashboard", include_in_schema=False)
def dashboard_index():
    return FileResponse(_STATIC_ROOT / "index.html")
