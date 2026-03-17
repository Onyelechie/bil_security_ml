from fastapi.testclient import TestClient

from server.main import app


def test_logs_endpoint_returns_structured_entries():
    with TestClient(app) as client:
        # Generate at least one request log context.
        health_response = client.get("/")
        assert health_response.status_code == 200

        response = client.get("/api/logs?limit=200")
        assert response.status_code == 200
        data = response.json()

        assert "logs" in data
        assert "latest_id" in data
        assert isinstance(data["logs"], list)
        assert isinstance(data["latest_id"], int)

        if data["logs"]:
            entry = data["logs"][-1]
            assert "id" in entry
            assert "timestamp" in entry
            assert "level" in entry
            assert "logger" in entry
            assert "message" in entry
