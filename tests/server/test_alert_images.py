from datetime import datetime, timezone
from pathlib import Path
import uuid

from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


def test_get_alert_image_serves_stored_file():
    storage_root = Path("storage/ws_alert_images")
    storage_root.mkdir(parents=True, exist_ok=True)

    filename = f"test_alert_image_{uuid.uuid4().hex}.png"
    image_path = storage_root / filename
    image_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    image_path.write_bytes(image_bytes)

    try:
        payload = {
            "site_id": "site_img",
            "camera_id": "cam_img",
            "edge_pc_id": "edge-img-1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detections": [{"class": "person", "confidence": 0.99}],
            "image_path": image_path.as_posix(),
        }
        create_response = client.post("/api/alerts", json=payload)
        assert create_response.status_code == 201
        alert_id = create_response.json()["id"]

        response = client.get(f"/api/alerts/{alert_id}/image")
        assert response.status_code == 200
        assert response.content == image_bytes
    finally:
        if image_path.exists():
            image_path.unlink()
