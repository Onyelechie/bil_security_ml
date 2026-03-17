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


def test_dashboard_ws_receives_heartbeat_event():
    with TestClient(app) as client:
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
            response = client.post("/api/heartbeat", json=heartbeat_payload)
            assert response.status_code == 201

            heartbeat_event = _receive_until(websocket, "heartbeat_received")
            assert heartbeat_event["payload"]["edge_pc_id"] == edge_pc_id


def test_dashboard_ws_receives_alert_event():
    with TestClient(app) as client:
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
            hb_response = client.post("/api/heartbeat", json=heartbeat_payload)
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
            response = client.post("/api/alerts", json=alert_payload)
            assert response.status_code == 201

            alert_event = _receive_until(websocket, "alert_received")
            assert alert_event["payload"]["edge_pc_id"] == edge_pc_id
