from fastapi.testclient import TestClient

from server.config import settings
from server.main import app


def test_logs_endpoint_returns_structured_entries():
    original_admin_password = settings.admin_password
    settings.admin_password = "test-admin-password"
    try:
        with TestClient(app) as client:
            # Generate at least one request log context.
            health_response = client.get("/")
            assert health_response.status_code == 200

            token_response = client.post(
                "/api/auth/token",
                data={"username": "admin", "password": settings.admin_password},
            )
            assert token_response.status_code == 200
            token = token_response.json()["access_token"]

            response = client.get(
                "/api/logs?limit=200",
                headers={"Authorization": f"Bearer {token}"},
            )
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
    finally:
        settings.admin_password = original_admin_password
