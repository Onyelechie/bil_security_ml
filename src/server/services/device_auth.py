from __future__ import annotations

import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..models.device import Device
from .ed25519 import verify_signature_b64

logger = logging.getLogger(__name__)


def _safe_identity(value: str | None) -> str:
    if not value:
        return "<missing>"
    return value[:64]


def require_signed_device(
    db: Session,
    *,
    device_id: str | None,
    signature_b64: str | None,
    message: bytes,
    expected_edge_pc_id: str,
) -> Device:
    if not device_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing device identity",
        )
    if not signature_b64:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing device signature",
        )

    dev = db.query(Device).filter_by(device_id=device_id).first()
    if not dev:
        logger.warning(
            "Rejecting request with unknown device_id=%s",
            _safe_identity(device_id),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown device",
        )
    if not dev.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device revoked or inactive",
        )
    if device_id != expected_edge_pc_id:
        logger.warning(
            "Rejecting request with device_id=%s for edge_pc_id=%s",
            _safe_identity(device_id),
            _safe_identity(expected_edge_pc_id),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device is not authorized for this edge PC",
        )
    if not verify_signature_b64(dev.public_key_b64, message, signature_b64):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature",
        )
    return dev
