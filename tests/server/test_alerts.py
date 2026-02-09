import pytest
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [
            {"class": "person", "confidence": 0.95},
            {"class": "vehicle", "confidence": 0.88}
        ],
        "image_path": None
    }
    response = client.post("/api/alerts", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["site_id"] == payload["site_id"]
    assert data["camera_id"] == payload["camera_id"]
    assert data["detections"][0]["class"] == "person"
    assert data["detections"][1]["class"] == "vehicle"
