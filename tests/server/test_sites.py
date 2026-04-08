from fastapi.testclient import TestClient

from server.main import app


def test_get_site_settings_rejects_path_traversal_name():
    with TestClient(app) as client:
        response = client.get("/api/sites/%2E%2E/settings")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid site name"


def test_put_site_settings_rejects_invalid_site_name():
    with TestClient(app) as client:
        response = client.put(
            "/api/sites/site-with-dash/settings",
            json={"image_retention_hours": 12},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid site name"
