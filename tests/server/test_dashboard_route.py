from fastapi.testclient import TestClient

from server.config import settings
from server.main import app


def _with_test_admin_password():
    original = settings.admin_password
    settings.admin_password = "test-admin-password"
    return original


def test_dashboard_route_redirects_when_not_logged_in():
    with TestClient(app) as client:
        response = client.get("/dashboard", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/dashboard/login"
        static_bypass = client.get("/static/dashboard/index.html", follow_redirects=False)
        assert static_bypass.status_code == 404


def test_dashboard_route_serves_html_after_login():
    original_admin_password = _with_test_admin_password()
    try:
        with TestClient(app) as client:
            login_response = client.post(
                "/dashboard/login",
                data={"password": settings.admin_password},
                follow_redirects=False,
            )
            assert login_response.status_code == 303
            assert login_response.headers["location"] == "/dashboard"

            response = client.get("/dashboard")
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
            assert "BIL Server Console" in response.text

            asset_response = client.get("/dashboard/assets/app.js")
            assert asset_response.status_code == 200
            assert "javascript" in asset_response.headers.get("content-type", "")
    finally:
        settings.admin_password = original_admin_password
