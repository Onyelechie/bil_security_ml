from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import base64

from ..db import SessionLocal
from ..models.device import Device
from .auth import get_current_admin

router = APIRouter(prefix="/api/devices", tags=["devices"])


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DeviceCreate(BaseModel):
    device_id: str
    public_key_b64: str
    description: str | None = None


class DeviceRotate(BaseModel):
    public_key_b64: str


@router.post("/enroll", status_code=status.HTTP_201_CREATED)
def enroll_device(
    payload: DeviceCreate,
    db: Session = Depends(get_db),
    _admin: str = Depends(get_current_admin),
):
    # Validate base64 public key
    try:
        decoded_key = base64.b64decode(payload.public_key_b64, validate=True)
        if len(decoded_key) != 32:
            raise ValueError("invalid ed25519 key length")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid public_key_b64",
        ) from exc

    existing = db.query(Device).filter_by(device_id=payload.device_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Device already enrolled",
        )

    dev = Device(
        device_id=payload.device_id,
        public_key_b64=payload.public_key_b64,
        enrolled_at=datetime.utcnow(),
        active=True,
    )
    db.add(dev)
    db.commit()
    return {"device_id": dev.device_id, "enrolled_at": dev.enrolled_at.isoformat()}


@router.get("/{device_id}")
def get_device(
    device_id: str,
    db: Session = Depends(get_db),
    _admin: str = Depends(get_current_admin),
):
    d = db.query(Device).filter_by(device_id=device_id).first()
    if not d:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return {
        "device_id": d.device_id,
        "public_key_b64": d.public_key_b64,
        "enrolled_at": d.enrolled_at.isoformat(),
        "active": d.active,
    }


@router.post("/{device_id}/revoke", status_code=status.HTTP_200_OK)
def revoke_device(
    device_id: str,
    db: Session = Depends(get_db),
    _admin: str = Depends(get_current_admin),
):
    d = db.query(Device).filter_by(device_id=device_id).first()
    if not d:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if not d.active:
        return {
            "device_id": d.device_id,
            "active": d.active,
            "revoked_at": d.revoked_at.isoformat() if d.revoked_at else None,
        }
    d.active = False
    d.revoked_at = datetime.utcnow()
    db.add(d)
    db.commit()
    return {"device_id": d.device_id, "active": d.active, "revoked_at": d.revoked_at.isoformat()}


@router.post("/{device_id}/rotate", status_code=status.HTTP_200_OK)
def rotate_device(
    device_id: str,
    payload: DeviceRotate,
    db: Session = Depends(get_db),
    _admin: str = Depends(get_current_admin),
):
    # Rotate the device public key. Caller must provide the new base64-encoded public key.
    try:
        decoded_key = base64.b64decode(payload.public_key_b64, validate=True)
        if len(decoded_key) != 32:
            raise ValueError("invalid ed25519 key length")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid public_key_b64",
        ) from exc

    d = db.query(Device).filter_by(device_id=device_id).first()
    if not d:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    d.public_key_b64 = payload.public_key_b64
    d.last_key_rotation_at = datetime.utcnow()
    db.add(d)
    db.commit()
    return {
        "device_id": d.device_id,
        "last_key_rotation_at": d.last_key_rotation_at.isoformat(),
    }
