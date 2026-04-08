from base64 import b64encode

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from server.config import settings
from server.main import app


def _set_test_admin_password():
    original = settings.admin_password
    settings.admin_password = "test-admin-password"
    return original


def _new_keypair() -> tuple[str, str]:
    signing_key = Ed25519PrivateKey.generate()
    private_key_b64 = b64encode(signing_key.private_bytes_raw()).decode("ascii")
    public_key_b64 = b64encode(
        signing_key.public_key().public_bytes_raw()
    ).decode("ascii")
    return private_key_b64, public_key_b64


def _admin_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": settings.admin_password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_device_list_and_revoke_flow():
    original_admin_password = _set_test_admin_password()
    try:
        with TestClient(app) as client:
            headers = _admin_headers(client)
            device_id = "edge-dashboard-1"
            _, public_key_b64 = _new_keypair()

            enroll_response = client.post(
                "/api/devices/enroll",
                headers=headers,
                json={
                    "device_id": device_id,
                    "public_key_b64": public_key_b64,
                },
            )
            assert enroll_response.status_code == 201

            list_response = client.get("/api/devices", headers=headers)
            assert list_response.status_code == 200
            devices = list_response.json()["devices"]
            enrolled = next(
                device for device in devices if device["device_id"] == device_id
            )
            assert enrolled["active"] is True
            assert enrolled["revoked_at"] is None

            revoke_response = client.post(f"/api/devices/{device_id}/revoke", headers=headers)
            assert revoke_response.status_code == 200
            revoked = revoke_response.json()
            assert revoked["device_id"] == device_id
            assert revoked["active"] is False
            assert revoked["revoked_at"] is not None

            refreshed = client.get("/api/devices", headers=headers)
            assert refreshed.status_code == 200
            refreshed_device = next(
                device
                for device in refreshed.json()["devices"]
                if device["device_id"] == device_id
            )
            assert refreshed_device["active"] is False
            assert refreshed_device["revoked_at"] is not None
    finally:
        settings.admin_password = original_admin_password


def test_device_list_requires_admin_auth():
    with TestClient(app) as client:
        response = client.get("/api/devices")
        assert response.status_code == 401
