from fastapi.testclient import TestClient

from edge_agent.config import EdgeSettings
from edge_agent.edge_api import create_app


def test_health_endpoint():
    cfg = EdgeSettings(edge_pc_id="edge-001", site_name="Site A")
    app = create_app(cfg)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "time_utc" in data


def test_heartbeat_endpoint():
    cfg = EdgeSettings(edge_pc_id="edge-001", site_name="Site A")
    app = create_app(cfg)
    client = TestClient(app)

    r = client.get("/heartbeat")
    assert r.status_code == 200
    data = r.json()

    assert data["edge_pc_id"] == "edge-001"
    assert data["site_name"] == "Site A"
    assert data["status"] == "online"
    assert "time_utc" in data
    assert isinstance(data["uptime_seconds"], int)
    assert data["uptime_seconds"] >= 0
