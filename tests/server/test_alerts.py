from fastapi.testclient import TestClient
from datetime import datetime, timezone

from server.main import app
from server.db import init_db

# Initialize database tables before tests
init_db()

client = TestClient(app)


def test_receive_alert():
    payload = {
        "site_id": "site_001",
        "camera_id": "cam_01",
        "edge_pc_id": "edge-test-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [
            {"class": "person", "confidence": 0.95},
            {"class": "vehicle", "confidence": 0.88},
        ],
        "image_path": None,
    }
    response = client.post("/api/alerts", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["site_id"] == payload["site_id"]
    assert data["camera_id"] == payload["camera_id"]
    assert data["detections"][0]["class"] == "person"
    assert data["detections"][1]["class"] == "vehicle"
    assert "id" in data  # Check that ID was generated


def test_list_alerts():
    #Get existing alerts (if any)
    inital_response = client.get("/api/alerts")
    assert inital_response.status_code == 200
    initial_count = len(inital_response.json().get("alerts", []))

    # create a new alert
    payload = {
        "site_id": "site_002",
        "camera_id": "cam_02",
        "edge_pc_id": "edge-test-2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [{"class": "person", "confidence": 0.9}],
        "image_path": "/path/to/image.jpg",
    }
    client.post("/api/alerts", json=payload)

    # Now test listing alerts
    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert len(data["alerts"]) == initial_count + 1 # At least one more than before
    assert data["alerts"][-1]["site_id"] == payload["site_id"] # The last alert should be the one we just added

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
