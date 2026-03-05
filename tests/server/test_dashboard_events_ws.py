from datetime import datetime, timezone
import uuid

from fastapi.testclient import TestClient

from server.main import app


def _receive_until(websocket, expected_type: str, max_messages: int = 8) -> dict:
    for _ in range(max_messages):
        message = websocket.receive_json()
        if message.get("type") == expected_type:
            return message
    raise AssertionError(f"Did not receive event type '{expected_type}'")


def test_dashboard_ws_receives_heartbeat_and_alert_events():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/dashboard-events") as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            edge_pc_id = f"edge-live-{uuid.uuid4()}"
            heartbeat_payload = {
                "edge_pc_id": edge_pc_id,
                "site_name": "Remote Site A",
                "status": "online",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            heartbeat_response = client.post("/api/heartbeat", json=heartbeat_payload)
            assert heartbeat_response.status_code == 201

            heartbeat_event = _receive_until(websocket, "heartbeat_received")
            assert heartbeat_event["payload"]["edge_pc_id"] == edge_pc_id

            alert_payload = {
                "site_id": "site_remote_a",
                "camera_id": "cam_remote_1",
                "edge_pc_id": edge_pc_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detections": [{"class": "person", "confidence": 0.99}],
                "image_path": None,
            }
            alert_response = client.post("/api/alerts", json=alert_payload)
            assert alert_response.status_code == 201

            alert_event = _receive_until(websocket, "alert_received")
            assert alert_event["payload"]["edge_pc_id"] == edge_pc_id
