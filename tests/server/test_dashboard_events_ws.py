from datetime import datetime, timezone
import uuid

from fastapi.testclient import TestClient

from server.config import settings
from server.main import app
from tests.server.device_auth_helpers import enroll_device, post_signed_json


def _receive_until(websocket, expected_type: str, max_messages: int = 8) -> dict:
    for _ in range(max_messages):
        message = websocket.receive_json()
        if message.get("type") == expected_type:
            return message
    raise AssertionError(f"Did not receive event type '{expected_type}'")


def _login_dashboard(client: TestClient) -> None:
    response = client.post(
        "/dashboard/login",
        data={"password": settings.admin_password},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_dashboard_ws_receives_heartbeat_event():
    original_admin_password = settings.admin_password
    settings.admin_password = "test-admin-password"
    try:
        with TestClient(app) as client:
            _login_dashboard(client)
            with client.websocket_connect("/ws/dashboard-events") as websocket:
                connected = websocket.receive_json()
                assert connected["type"] == "connected"

                edge_pc_id = f"edge-ws-hb-{uuid.uuid4().hex}"
                heartbeat_payload = {
                    "edge_pc_id": edge_pc_id,
                    "site_name": "Remote Site A",
                    "status": "online",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                private_key_b64 = enroll_device(edge_pc_id)
                response = post_signed_json(
                    client,
                    "/api/heartbeat",
                    heartbeat_payload,
                    device_id=edge_pc_id,
                    private_key_b64=private_key_b64,
                )
                assert response.status_code == 201

                heartbeat_event = _receive_until(websocket, "heartbeat_received")
                assert heartbeat_event["payload"]["edge_pc_id"] == edge_pc_id
    finally:
        settings.admin_password = original_admin_password


def test_dashboard_ws_receives_alert_event():
    original_admin_password = settings.admin_password
    settings.admin_password = "test-admin-password"
    try:
        with TestClient(app) as client:
            _login_dashboard(client)
            with client.websocket_connect("/ws/dashboard-events") as websocket:
                connected = websocket.receive_json()
                assert connected["type"] == "connected"

                # Setup: create an EdgePC by posting a heartbeat first
                edge_pc_id = f"edge-ws-alert-{uuid.uuid4().hex}"
                heartbeat_payload = {
                    "edge_pc_id": edge_pc_id,
                    "site_name": "Remote Site B",
                    "status": "online",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                private_key_b64 = enroll_device(edge_pc_id)
                hb_response = post_signed_json(
                    client,
                    "/api/heartbeat",
                    heartbeat_payload,
                    device_id=edge_pc_id,
                    private_key_b64=private_key_b64,
                )
                assert hb_response.status_code == 201

                # consume the heartbeat event so the websocket is up-to-date
                _ = _receive_until(websocket, "heartbeat_received")

                alert_payload = {
                    "site_id": "site_remote_a",
                    "camera_id": "cam_remote_1",
                    "edge_pc_id": edge_pc_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detections": [{"class": "person", "confidence": 0.99}],
                    "image_path": None,
                }
                response = post_signed_json(
                    client,
                    "/api/alerts",
                    alert_payload,
                    device_id=edge_pc_id,
                    private_key_b64=private_key_b64,
                )
                assert response.status_code == 201

                alert_event = _receive_until(websocket, "alert_received")
                assert alert_event["payload"]["edge_pc_id"] == edge_pc_id
    finally:
        settings.admin_password = original_admin_password
