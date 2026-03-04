from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.main import app
from server.services.image_storage import ImageStorageService
from server.services.ws_alert_dispatcher import AlertQueueFullError
from server.services.ws_connection_manager import WebSocketConnectionManager


def _alert_payload(edge_pc_id: str, site_id: str, camera_id: str) -> dict:
    return {
        "site_id": site_id,
        "camera_id": camera_id,
        "edge_pc_id": edge_pc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [{"class": "person", "confidence": 0.99}],
        "image_path": None,
    }


def _alert_meta_frame(edge_pc_id: str, site_id: str, camera_id: str) -> dict:
    return {
        "type": "alert_meta",
        "alert": _alert_payload(edge_pc_id=edge_pc_id, site_id=site_id, camera_id=camera_id),
    }


def test_websocket_alert_ingestion_ack():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"
            assert connected["status"] == "ok"

            websocket.send_json(_alert_payload("edge-ws-1", "site_ws_1", "cam_ws_1"))
            ack = websocket.receive_json()
            assert ack["type"] == "ack"
            assert ack["status"] == "ok"
            assert ack["alert"]["edge_pc_id"] == "edge-ws-1"
            assert ack["alert"]["site_id"] == "site_ws_1"
            assert "id" in ack["alert"]


def test_websocket_alert_validation_error():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as websocket:
            websocket.receive_json()
            websocket.send_json({"site_id": "missing-required-fields"})
            err = websocket.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "validation_error"
            assert isinstance(err["errors"], list)
            assert len(err["errors"]) > 0


def test_websocket_handles_multiple_connections():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as ws1, client.websocket_connect("/ws/alerts") as ws2:
            ws1.receive_json()
            ws2.receive_json()

            ws1.send_json(_alert_payload("edge-ws-2a", "site_ws_2a", "cam_ws_2a"))
            ws2.send_json(_alert_payload("edge-ws-2b", "site_ws_2b", "cam_ws_2b"))

            ack1 = ws1.receive_json()
            ack2 = ws2.receive_json()

            assert ack1["type"] == "ack"
            assert ack2["type"] == "ack"
            assert ack1["alert"]["edge_pc_id"] == "edge-ws-2a"
            assert ack2["alert"]["edge_pc_id"] == "edge-ws-2b"


def test_websocket_invalid_json_message_returns_error():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as websocket:
            websocket.receive_json()
            websocket.send_text("{not-valid-json")
            err = websocket.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "invalid_message"


def test_websocket_alert_meta_then_binary_ingestion_ack(tmp_path):
    with TestClient(app) as client:
        original_storage = app.state.ws_image_storage
        image_storage = ImageStorageService(str(tmp_path))
        image_storage.ensure_ready()
        app.state.ws_image_storage = image_storage
        try:
            with client.websocket_connect("/ws/alerts") as websocket:
                websocket.receive_json()
                websocket.send_json(_alert_meta_frame("edge-ws-bin-1", "site_ws_bin_1", "cam_ws_bin_1"))
                meta_ack = websocket.receive_json()
                assert meta_ack["type"] == "meta_received"
                assert meta_ack["status"] == "ok"

                img = b"\x89PNG\r\n\x1a\n\x00\x01binary-image-content"
                websocket.send_bytes(img)
                ack = websocket.receive_json()
                assert ack["type"] == "ack"
                assert ack["status"] == "ok"
                assert ack["alert"]["edge_pc_id"] == "edge-ws-bin-1"
                assert ack["alert"]["image_path"] is not None
                image_path = Path(ack["alert"]["image_path"])
                assert image_path.exists()
                assert image_path.parent == tmp_path
                assert "site_ws_bin_1" in image_path.name
                assert "cam_ws_bin_1" in image_path.name
        finally:
            app.state.ws_image_storage = original_storage


def test_websocket_binary_without_meta_returns_error():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as websocket:
            websocket.receive_json()
            websocket.send_bytes(b"\x00\x01orphan-image")
            err = websocket.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "meta_missing"


def test_websocket_binary_too_large_returns_error():
    with TestClient(app) as client:
        original_limit = app.state.ws_max_image_bytes
        app.state.ws_max_image_bytes = 4
        try:
            with client.websocket_connect("/ws/alerts") as websocket:
                websocket.receive_json()
                websocket.send_json(_alert_meta_frame("edge-ws-bin-2", "site_ws_bin_2", "cam_ws_bin_2"))
                websocket.receive_json()
                websocket.send_bytes(b"\x01\x02\x03\x04\x05")
                err = websocket.receive_json()
                assert err["type"] == "error"
                assert err["code"] == "image_too_large"
        finally:
            app.state.ws_max_image_bytes = original_limit


def test_websocket_queue_full_returns_error():
    class _QueueFullDispatcher:
        async def submit(self, payload: dict) -> dict:  # noqa: ARG002
            raise AlertQueueFullError("Alert queue is full")

    with TestClient(app) as client:
        original_dispatcher = app.state.ws_alert_dispatcher
        app.state.ws_alert_dispatcher = _QueueFullDispatcher()
        try:
            with client.websocket_connect("/ws/alerts") as websocket:
                websocket.receive_json()
                websocket.send_json(_alert_payload("edge-qf-1", "site_qf", "cam_qf"))
                err = websocket.receive_json()
                assert err["type"] == "error"
                assert err["code"] == "queue_full"
                assert "queue" in err["message"].lower()
        finally:
            app.state.ws_alert_dispatcher = original_dispatcher


def test_websocket_rejects_when_max_connections_reached():
    with TestClient(app) as client:
        original_manager = app.state.ws_connection_manager
        app.state.ws_connection_manager = WebSocketConnectionManager(max_connections=1)
        try:
            with client.websocket_connect("/ws/alerts") as ws1:
                ws1.receive_json()
                with pytest.raises(WebSocketDisconnect) as exc_info:
                    with client.websocket_connect("/ws/alerts") as ws2:
                        ws2.receive_json()
                assert exc_info.value.code == 1013
        finally:
            app.state.ws_connection_manager = original_manager
