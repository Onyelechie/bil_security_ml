import base64
import hashlib
import json
from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from server.db import SessionLocal
from server.models.device import Device


def _new_keypair() -> tuple[str, str]:
    signing_key = Ed25519PrivateKey.generate()
    private_key_bytes = signing_key.private_bytes_raw()
    public_key_bytes = signing_key.public_key().public_bytes_raw()
    private_key_b64 = base64.b64encode(private_key_bytes).decode("ascii")
    public_key_b64 = base64.b64encode(public_key_bytes).decode("ascii")
    return private_key_b64, public_key_b64


def enroll_device(device_id: str) -> str:
    private_key_b64, public_key_b64 = _new_keypair()
    db = SessionLocal()
    try:
        device = db.query(Device).filter_by(device_id=device_id).first()
        if device is None:
            device = Device(
                device_id=device_id,
                public_key_b64=public_key_b64,
                enrolled_at=datetime.now(timezone.utc),
                active=True,
            )
            db.add(device)
        else:
            device.public_key_b64 = public_key_b64
            device.active = True
            device.revoked_at = None
        db.commit()
    finally:
        db.close()
    return private_key_b64


def sign_message(private_key_b64: str, message: bytes) -> str:
    signing_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(private_key_b64, validate=True)
    )
    signature = signing_key.sign(message)
    return base64.b64encode(signature).decode("ascii")


def post_signed_json(
    client: TestClient,
    url: str,
    payload: dict,
    *,
    device_id: str,
    private_key_b64: str,
):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Device-Id": device_id,
        "X-Device-Signature": sign_message(private_key_b64, body),
    }
    return client.post(url, content=body, headers=headers)


def build_signed_upload_headers(
    *,
    device_id: str,
    edge_pc_id: str,
    site_id: str,
    camera_id: str,
    timestamp: str,
    image_bytes: bytes,
    private_key_b64: str,
) -> dict[str, str]:
    sha = hashlib.sha256(image_bytes).hexdigest()
    canonical = f"{site_id}|{camera_id}|{edge_pc_id}|{timestamp}|{sha}".encode("utf-8")
    return {
        "X-Device-Id": device_id,
        "X-Device-Signature": sign_message(private_key_b64, canonical),
    }


def register_edge(
    client: TestClient, edge_pc_id: str, site_name: str = "Test Site"
) -> str:
    private_key_b64 = enroll_device(edge_pc_id)
    heartbeat_payload = {
        "edge_pc_id": edge_pc_id,
        "site_name": site_name,
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    response = post_signed_json(
        client,
        "/api/heartbeat",
        heartbeat_payload,
        device_id=edge_pc_id,
        private_key_b64=private_key_b64,
    )
    assert response.status_code == 201
    return private_key_b64
