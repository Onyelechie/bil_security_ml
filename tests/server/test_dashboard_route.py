from fastapi.testclient import TestClient

from server.main import app


def test_dashboard_route_serves_html():
    with TestClient(app) as client:
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "BIL Server Console" in response.text
