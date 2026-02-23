import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from edge_agent.config import EdgeSettings
from edge_agent.edge_api import create_app


def _parse_iso_z(ts: str) -> datetime:
    """
    Parse ISO-8601 timestamps that may end with 'Z' (UTC).
    Python's datetime.fromisoformat() doesn't accept trailing 'Z', so convert to +00:00.
    """
    if not isinstance(ts, str):
        raise TypeError(f"timestamp must be str, got {type(ts)}")
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@pytest.fixture
def client():
    cfg = EdgeSettings(edge_pc_id="edge-001", site_name="Site A")
    app = create_app(cfg)
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "time_utc" in data

    # Validate timestamp format (supports trailing 'Z')
    try:
        _parse_iso_z(data["time_utc"])
    except (ValueError, TypeError) as e:
        pytest.fail(f"time_utc is not a valid ISO 8601 timestamp: {data.get('time_utc')} ({e})")


def test_heartbeat_endpoint(client):
    response = client.get("/heartbeat")
    assert response.status_code == 200

    data = response.json()
    assert data["edge_pc_id"] == "edge-001"
    assert data["site_name"] == "Site A"
    assert data["status"] == "online"
    assert "time_utc" in data

    # Validate timestamp format (supports trailing 'Z')
    _parse_iso_z(data["time_utc"])

    assert isinstance(data["uptime_seconds"], int)
    assert data["uptime_seconds"] >= 0
