from fastapi.testclient import TestClient
from datetime import datetime, timezone, timedelta

from server.main import app
from server.db import init_db

init_db()
client = TestClient(app)


def test_heartbeat_create_and_update():
    payload = {
        "edge_pc_id": "edge-001",
        "site_name": "Site A",
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # First heartbeat (create)
    response = client.post("/api/heartbeat", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["edge_pc_id"] == payload["edge_pc_id"]
    assert data["site_name"] == payload["site_name"]
    assert data["status"] == payload["status"]
    assert "last_heartbeat" in data
    assert data["message"] == "Server received heartbeat"

    # Update heartbeat (change status)
    payload["status"] = "idle"
    payload["timestamp"] = (
        datetime.now(timezone.utc) + timedelta(seconds=10)
    ).isoformat()
    response = client.post("/api/heartbeat", json=payload)
    assert response.status_code == 201
    data2 = response.json()
    assert data2["status"] == "idle"
    assert data2["last_heartbeat"] > data["last_heartbeat"]
    assert data2["message"] == "Server received heartbeat"


def test_heartbeat_missing_fields():
    payload = {
        "edge_pc_id": "edge-002",
        # Missing site_name
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    response = client.post("/api/heartbeat", json=payload)
    assert response.status_code == 422
