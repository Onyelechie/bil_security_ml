import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

router = APIRouter(prefix="/api/sites", tags=["sites"])
logger = logging.getLogger(__name__)
SITE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class SiteSettingsIn(BaseModel):
    image_retention_hours: int | None = None


class SiteSettingsOut(BaseModel):
    site_name: str
    image_retention_hours: int | None = None


def _validated_site_name(site_name: str) -> str:
    if not isinstance(site_name, str) or not SITE_NAME_PATTERN.fullmatch(site_name):
        logger.warning("Rejected invalid site_name for settings path: %r", site_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid site name",
        )
    return site_name


def _site_settings_path(root: Path, site_name: str) -> Path:
    safe = _validated_site_name(site_name)
    return root / safe / ".site_settings.json"


@router.get("/{site_name}/settings", response_model=SiteSettingsOut)
def get_site_settings(site_name: str, request: Request):
    storage = request.app.state.image_storage
    root = storage._root_dir
    path = _site_settings_path(root, site_name)
    if not path.is_file():
        return {"site_name": site_name, "image_retention_hours": None}
    try:
        data = json.loads(path.read_text(encoding="utf8"))
        return {
            "site_name": site_name,
            "image_retention_hours": data.get("image_retention_hours"),
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read site settings",
        )


@router.put("/{site_name}/settings", response_model=SiteSettingsOut)
def set_site_settings(site_name: str, body: SiteSettingsIn, request: Request):
    storage = request.app.state.image_storage
    root = storage._root_dir
    path = _site_settings_path(root, site_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image_retention_hours": (
                int(body.image_retention_hours)
                if body.image_retention_hours is not None
                else None
            )
        }
        path.write_text(json.dumps(data), encoding="utf8")
        return {
            "site_name": site_name,
            "image_retention_hours": data.get("image_retention_hours"),
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to write site settings",
        )
