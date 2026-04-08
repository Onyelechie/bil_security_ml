from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from server.db import init_db
from server.main import app
from server.services import alert_ingestion as alert_ingestion_module
from tests.server.device_auth_helpers import post_signed_json, register_edge

# Initialize database tables before tests
init_db()

client = TestClient(app)


def test_receive_alert():
    edge_pc_id = "edge-test-1"
    private_key_b64 = register_edge(client, edge_pc_id, site_name="Alert Test Site")
    payload = {
        "site_id": "site_001",
        "camera_id": "cam_01",
        "edge_pc_id": edge_pc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [
            {"class": "person", "confidence": 0.95},
            {"class": "vehicle", "confidence": 0.88},
        ],
        "image_path": None,
    }
    response = post_signed_json(
        client,
        "/api/alerts",
        payload,
        device_id=edge_pc_id,
        private_key_b64=private_key_b64,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["site_id"] == payload["site_id"]
    assert data["camera_id"] == payload["camera_id"]
    assert data["detections"][0]["class"] == "person"
    assert data["detections"][1]["class"] == "vehicle"
    assert "id" in data
    returned_ts = datetime.fromisoformat(data["timestamp"])
    assert returned_ts.utcoffset() == ZoneInfo("America/Winnipeg").utcoffset(
        returned_ts
    )
    assert "received_at" in data
    received_ts = datetime.fromisoformat(data["received_at"])
    assert received_ts.tzinfo is not None


def test_list_alerts():
    initial_response = client.get("/api/alerts")
    assert initial_response.status_code == 200
    initial_count = len(initial_response.json().get("alerts", []))

    edge_pc_id = "edge-test-2"
    private_key_b64 = register_edge(client, edge_pc_id, site_name="Alert Test Site")
    payload = {
        "site_id": "site_002",
        "camera_id": "cam_02",
        "edge_pc_id": edge_pc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [{"class": "person", "confidence": 0.9}],
        "image_path": "/path/to/image.jpg",
    }
    create_response = post_signed_json(
        client,
        "/api/alerts",
        payload,
        device_id=edge_pc_id,
        private_key_b64=private_key_b64,
    )
    assert create_response.status_code == 201

    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert len(data["alerts"]) == initial_count + 1
    assert data["alerts"][0]["site_id"] == payload["site_id"]
    listed_ts = datetime.fromisoformat(data["alerts"][0]["timestamp"])
    assert listed_ts.utcoffset() == ZoneInfo("America/Winnipeg").utcoffset(listed_ts)
    listed_received = datetime.fromisoformat(data["alerts"][0]["received_at"])
    assert listed_received.tzinfo is not None


def test_list_alerts_supports_sort_modes(monkeypatch):
    edge_pc_id = "edge-test-sort"
    private_key_b64 = register_edge(client, edge_pc_id, site_name="Sort Test Site")
    received_times = iter(
        [
            datetime(
                2026, 1, 1, 12, 0, tzinfo=ZoneInfo("America/Winnipeg")
            ),
            datetime(
                2026, 1, 1, 11, 0, tzinfo=ZoneInfo("America/Winnipeg")
            ),
        ]
    )
    monkeypatch.setattr(
        alert_ingestion_module,
        "now_in_winnipeg",
        lambda: next(received_times),
    )

    first_payload = {
        "site_id": "site_received_first",
        "camera_id": "cam_01",
        "edge_pc_id": edge_pc_id,
        "timestamp": datetime(
            2026, 1, 1, 10, 0, tzinfo=ZoneInfo("America/Winnipeg")
        ).isoformat(),
        "detections": [{"class": "person", "confidence": 0.95}],
        "image_path": None,
    }
    second_payload = {
        "site_id": "site_received_second",
        "camera_id": "cam_02",
        "edge_pc_id": edge_pc_id,
        "timestamp": datetime(
            2026, 1, 1, 11, 0, tzinfo=ZoneInfo("America/Winnipeg")
        ).isoformat(),
        "detections": [{"class": "person", "confidence": 0.9}],
        "image_path": None,
    }

    response_one = post_signed_json(
        client,
        "/api/alerts",
        first_payload,
        device_id=edge_pc_id,
        private_key_b64=private_key_b64,
    )
    assert response_one.status_code == 201
    response_two = post_signed_json(
        client,
        "/api/alerts",
        second_payload,
        device_id=edge_pc_id,
        private_key_b64=private_key_b64,
    )
    assert response_two.status_code == 201

    by_received = client.get("/api/alerts?sort_by=received_at")
    assert by_received.status_code == 200
    received_alerts = by_received.json()["alerts"]
    received_sites = [
        alert["site_id"]
        for alert in received_alerts
        if alert["site_id"] in {"site_received_first", "site_received_second"}
    ]
    assert received_sites == ["site_received_first", "site_received_second"]

    by_timestamp = client.get("/api/alerts?sort_by=timestamp")
    assert by_timestamp.status_code == 200
    timestamp_alerts = by_timestamp.json()["alerts"]
    timestamp_sites = [
        alert["site_id"]
        for alert in timestamp_alerts
        if alert["site_id"] in {"site_received_first", "site_received_second"}
    ]
    assert timestamp_sites == ["site_received_second", "site_received_first"]


def test_receive_alert_rejects_unregistered_edge():
    payload = {
        "site_id": "site_unauth",
        "camera_id": "cam_unauth",
        "edge_pc_id": "edge-not-registered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [{"class": "person", "confidence": 0.91}],
    }
    response = client.post("/api/alerts", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing device identity"


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
