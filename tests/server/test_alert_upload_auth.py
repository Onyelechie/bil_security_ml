import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from server.main import app
from tests.server.device_auth_helpers import (
    build_signed_upload_headers,
    enroll_device,
    register_edge,
)


def _upload_payload(edge_pc_id: str) -> tuple[dict, dict, bytes]:
    image_bytes = b"\x89PNG\r\n\x1a\nupload-bytes"
    data = {
        "site_id": "upload_site",
        "camera_id": "upload_cam",
        "edge_pc_id": edge_pc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": json.dumps([{"class": "person", "confidence": 0.98}]),
    }
    files = {
        "image": ("upload.png", image_bytes, "image/png"),
    }
    return data, files, image_bytes


def test_alert_upload_rejects_unregistered_edge():
    with TestClient(app) as client:
        edge_pc_id = "edge-upload-unauthorized"
        private_key_b64 = enroll_device(edge_pc_id)
        data, files, image_bytes = _upload_payload(edge_pc_id)
        headers = build_signed_upload_headers(
            device_id=edge_pc_id,
            edge_pc_id=edge_pc_id,
            site_id=data["site_id"],
            camera_id=data["camera_id"],
            timestamp=data["timestamp"],
            image_bytes=image_bytes,
            private_key_b64=private_key_b64,
        )
        response = client.post("/api/alerts/upload", data=data, files=files, headers=headers)
        assert response.status_code == 403
        assert response.json()["detail"] == "Edge PC is not authorized to submit alerts"


def test_alert_upload_accepts_registered_edge():
    with TestClient(app) as client:
        edge_pc_id = "edge-upload-authorized"
        private_key_b64 = register_edge(client, edge_pc_id, site_name="Upload Auth Test Site")
        data, files, image_bytes = _upload_payload(edge_pc_id)
        headers = build_signed_upload_headers(
            device_id=edge_pc_id,
            edge_pc_id=edge_pc_id,
            site_id=data["site_id"],
            camera_id=data["camera_id"],
            timestamp=data["timestamp"],
            image_bytes=image_bytes,
            private_key_b64=private_key_b64,
        )
        response = client.post("/api/alerts/upload", data=data, files=files, headers=headers)
        assert response.status_code == 201
        body = response.json()
        assert body["edge_pc_id"] == edge_pc_id
        assert "id" in body
        uploaded_ts = datetime.fromisoformat(body["timestamp"])
        assert uploaded_ts.utcoffset() == ZoneInfo("America/Winnipeg").utcoffset(uploaded_ts)
