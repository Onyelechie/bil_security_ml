# tests/test_sender.py
import base64
import json
import os
import time
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from edge_agent.config import EdgeSettings
from edge_agent.sender import ServerSender
from tests.temp_dirs import repo_temp_dir


@pytest.fixture
def tmp_path():
    with repo_temp_dir("edge_sender_") as path:
        yield path


@pytest.fixture
def settings(tmp_path) -> EdgeSettings:
    """Provides a standard EdgeSettings for testing."""
    signing_key = Ed25519PrivateKey.generate()
    private_key_b64 = base64.b64encode(signing_key.private_bytes_raw()).decode("ascii")
    return EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        device_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
        device_private_key_b64=private_key_b64,
        offline_queue_dir=str(tmp_path / "offline_queue"),
    )


@pytest.fixture
def sender(settings: EdgeSettings, mocker) -> ServerSender:
    """
    Provides a ServerSender instance with a mocked session.
    Note: `mocker` is from pytest-mock; no need to annotate it as MagicMock.
    """
    sender_instance = ServerSender(settings)
    sender_instance._session = mocker.MagicMock()
    return sender_instance


def test_send_heartbeat_structure(sender: ServerSender):
    """Test that heartbeat sends correct JSON to the correct URL."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    success = sender.send_heartbeat()

    assert success is True
    sender._session.post.assert_called_once()

    args, kwargs = sender._session.post.call_args
    url = args[0]
    body = kwargs["data"]
    payload = json.loads(body.decode("utf-8"))
    headers = kwargs["headers"]

    assert url == "http://mock-server/api/heartbeat"
    assert payload["edge_pc_id"] == "test-edge-1"
    assert payload["status"] == "starting"
    assert "timestamp" in payload
    assert headers["X-Device-Id"] == "test-edge-1"
    assert "X-Device-Signature" in headers
    mock_resp.raise_for_status.assert_called_once()


def test_send_heartbeat_with_uptime(sender: ServerSender):
    """Test that providing a start time includes uptime_seconds (timing-tolerant)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    started = time.monotonic() - 10
    sender.send_heartbeat(started_monotonic=started)

    _, kwargs = sender._session.post.call_args
    payload = json.loads(kwargs["data"].decode("utf-8"))

    assert "uptime_seconds" in payload
    assert 10 <= payload["uptime_seconds"] < 12


def test_send_heartbeat_reflects_status_change(sender: ServerSender):
    """Test that status updates are reflected in the heartbeat payload."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    sender.set_status("online")
    sender.send_heartbeat()

    _, kwargs = sender._session.post.call_args
    payload = json.loads(kwargs["data"].decode("utf-8"))

    assert payload["status"] == "online"


def test_send_alert_structure(sender: ServerSender):
    """Test that send_alert sends a server-compatible payload to the correct URL."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    detections = [{"class": "person", "confidence": 0.9}]
    success = sender.send_alert(camera_id="cam-1", detections=detections)

    assert success is True
    sender._session.post.assert_called_once()

    args, kwargs = sender._session.post.call_args
    assert args[0] == "http://mock-server/api/alerts"
    payload = json.loads(kwargs["data"].decode("utf-8"))
    assert payload["site_id"] == "site-1"
    assert payload["edge_pc_id"] == "test-edge-1"
    assert payload["camera_id"] == "cam-1"
    assert payload["detections"] == detections
    assert "timestamp" in payload
    sent_at = datetime.fromisoformat(payload["timestamp"])
    assert (
        sent_at.utcoffset()
        == sent_at.astimezone(ZoneInfo("America/Winnipeg")).utcoffset()
    )
    assert kwargs["headers"]["X-Device-Id"] == "test-edge-1"
    mock_resp.raise_for_status.assert_called_once()


def test_send_alert_omits_image_path_without_shared_root(
    settings: EdgeSettings, tmp_path, mocker
):
    sender = ServerSender(settings)
    sender._session = mocker.MagicMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"test")

    success = sender.send_alert(
        camera_id="cam-1",
        detections=[{"class": "person", "confidence": 0.9}],
        image_path=str(image_path),
    )

    assert success is True
    _, kwargs = sender._session.post.call_args
    payload = json.loads(kwargs["data"].decode("utf-8"))
    assert "image_path" not in payload


def test_send_alert_includes_image_path_with_shared_root(tmp_path, mocker):
    signing_key = Ed25519PrivateKey.generate()
    private_key_b64 = base64.b64encode(signing_key.private_bytes_raw()).decode("ascii")
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    image_path = shared_root / "img.jpg"
    image_path.write_bytes(b"test")

    settings = EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        device_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
        device_private_key_b64=private_key_b64,
        offline_queue_dir=str(tmp_path / "offline_queue"),
        shared_storage_root=str(shared_root),
    )
    sender = ServerSender(settings)
    sender._session = mocker.MagicMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    success = sender.send_alert(
        camera_id="cam-1",
        detections=[{"class": "person", "confidence": 0.9}],
        image_path=str(image_path),
    )

    assert success is True
    _, kwargs = sender._session.post.call_args
    payload = json.loads(kwargs["data"].decode("utf-8"))
    assert payload["image_path"] == str(image_path)


def test_send_heartbeat_handles_server_error(sender, mocker):
    """
    Test that send_heartbeat properly handles a server error and logs it.
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    sender._session.post.return_value = mock_resp

    mocked_logger = mocker.patch("edge_agent.sender.logger")

    success = sender.send_heartbeat()

    assert success is False
    mocked_logger.error.assert_called()
    assert any(
        "Failed to send heartbeat" in str(call.args[0])
        for call in mocked_logger.error.call_args_list
    )


def test_send_alert_handles_request_exception(sender, mocker):
    """
    Test that send_alert properly handles a requests exception and logs it.
    """
    sender._session.post.side_effect = requests.RequestException("Connection timed out")

    mocked_logger = mocker.patch("edge_agent.sender.logger")

    success = sender.send_alert(
        camera_id="cam-2", detections=[{"class": "loitering", "confidence": 0.5}]
    )

    assert success is False
    mocked_logger.error.assert_called()
    assert any(
        "Failed to send alert" in str(call.args[0])
        for call in mocked_logger.error.call_args_list
    )
    files = os.listdir(sender.queue_dir)
    assert any(f.startswith("alert_") and f.endswith(".json") for f in files)


def test_send_alert_rejects_invalid_detections(sender, mocker):
    """
    Test that send_alert rejects invalid detections and logs an error.
    """
    mocked_logger = mocker.patch("edge_agent.sender.logger")

    success = sender.send_alert(camera_id="cam-3", detections=[{"class": "person"}])

    assert success is False
    sender._session.post.assert_not_called()
    mocked_logger.error.assert_called()


def test_send_heartbeat_requires_private_key(settings: EdgeSettings, mocker):
    sender = ServerSender(settings.model_copy(update={"device_private_key_b64": ""}))
    sender._session = mocker.MagicMock()

    success = sender.send_heartbeat()

    assert success is False
    sender._session.post.assert_not_called()


def test_send_alert_drops_on_4xx(sender: ServerSender):
    """
    Test that send_alert does not queue alerts on client errors (4xx)
    and logs appropriately.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    http_err = requests.HTTPError("400 Bad Request")
    http_err.response = mock_resp
    sender._session.post.return_value = mock_resp
    mock_resp.raise_for_status.side_effect = http_err

    detections = [{"class": "person", "confidence": 0.9}]
    success = sender.send_alert(camera_id="cam-1", detections=detections)

    assert success is False
    assert os.listdir(sender.queue_dir) == []


def test_retry_queued_alerts_resends_and_deletes(sender: ServerSender, tmp_path):
    """
    Test that retry_queued_alerts resends queued alerts
    and deletes them on success.
    """
    payload = {
        "site_id": "site-1",
        "edge_pc_id": "test-edge-1",
        "camera_id": "cam-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "detections": [{"class": "person", "confidence": 0.9}],
    }
    queue_file = os.path.join(sender.queue_dir, "alert_20260101_000000_000000.json")
    with open(queue_file, "w") as f:
        json.dump(payload, f)

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    sender.retry_queued_alerts()

    sender._session.post.assert_called_once()
    assert not os.path.exists(queue_file)


def test_retry_queued_alerts_quarantines_invalid_json_and_continues(
    sender: ServerSender,
):
    bad_file = os.path.join(sender.queue_dir, "alert_20260101_000000_000000.json")
    good_file = os.path.join(sender.queue_dir, "alert_20260101_000000_000001.json")
    with open(bad_file, "w") as f:
        f.write("{not valid json")
    with open(good_file, "w") as f:
        json.dump(
            {
                "site_id": "site-1",
                "edge_pc_id": "test-edge-1",
                "camera_id": "cam-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "detections": [{"class": "person", "confidence": 0.9}],
            },
            f,
        )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    sender.retry_queued_alerts()

    sender._session.post.assert_called_once()
    assert not os.path.exists(good_file)

    bad_dir = os.path.join(sender.queue_dir, "bad")
    quarantined = [
        f for f in os.listdir(bad_dir) if f.startswith("alert_20260101_000000_000000")
    ]
    assert quarantined


def test_retry_queued_alerts_quarantines_on_4xx_and_continues(sender: ServerSender):
    first_file = os.path.join(sender.queue_dir, "alert_20260101_000000_000000.json")
    second_file = os.path.join(sender.queue_dir, "alert_20260101_000000_000001.json")
    payload = {
        "site_id": "site-1",
        "edge_pc_id": "test-edge-1",
        "camera_id": "cam-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "detections": [{"class": "person", "confidence": 0.9}],
    }
    with open(first_file, "w") as f:
        json.dump(payload, f)
    with open(second_file, "w") as f:
        json.dump(payload, f)

    bad_resp = MagicMock()
    bad_resp.status_code = 400
    bad_err = requests.HTTPError("400 Bad Request")
    bad_err.response = bad_resp
    bad_resp.raise_for_status.side_effect = bad_err

    ok_resp = MagicMock()
    ok_resp.raise_for_status.return_value = None

    sender._session.post.side_effect = [bad_resp, ok_resp]

    sender.retry_queued_alerts()

    assert sender._session.post.call_count == 2
    assert not os.path.exists(second_file)

    bad_dir = os.path.join(sender.queue_dir, "bad")
    quarantined = [
        f for f in os.listdir(bad_dir) if f.startswith("alert_20260101_000000_000000")
    ]
    assert quarantined


def test_retry_queued_alerts_quarantines_when_signature_missing(
    settings: EdgeSettings, tmp_path, mocker
):
    sender = ServerSender(
        settings.model_copy(
            update={
                "device_private_key_b64": "",
                "offline_queue_dir": str(tmp_path / "offline_queue"),
            }
        )
    )
    sender._session = mocker.MagicMock()

    payload = {
        "site_id": "site-1",
        "edge_pc_id": "test-edge-1",
        "camera_id": "cam-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "detections": [{"class": "person", "confidence": 0.9}],
    }
    queue_file = os.path.join(sender.queue_dir, "alert_20260101_000000_000000.json")
    with open(queue_file, "w") as f:
        json.dump(payload, f)

    sender.retry_queued_alerts()

    sender._session.post.assert_not_called()
    bad_dir = os.path.join(sender.queue_dir, "bad")
    quarantined = [
        f for f in os.listdir(bad_dir) if f.startswith("alert_20260101_000000_000000")
    ]
    assert quarantined


def test_cleanup_quarantine_removes_old_files(tmp_path):
    settings = EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
        offline_queue_dir=str(tmp_path / "offline_queue"),
        queue_quarantine_retention_days=7,
    )
    sender = ServerSender(settings)

    bad_dir = os.path.join(sender.queue_dir, "bad")
    os.makedirs(bad_dir, exist_ok=True)
    old_file = os.path.join(bad_dir, "old.json")
    new_file = os.path.join(bad_dir, "new.json")
    with open(old_file, "w") as f:
        f.write("{}")
    with open(new_file, "w") as f:
        f.write("{}")

    old_mtime = time.time() - (8 * 86400)
    os.utime(old_file, (old_mtime, old_mtime))

    sender._cleanup_quarantine()

    assert not os.path.exists(old_file)
    assert os.path.exists(new_file)


def test_queue_payload_keeps_image_path_in_shared_root(tmp_path):
    shared_root = tmp_path / "shared_storage"
    shared_root.mkdir()
    image_path = shared_root / "img.jpg"
    image_path.write_bytes(b"test")

    settings = EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
        offline_queue_dir=str(tmp_path / "offline_queue"),
        shared_storage_root=str(shared_root),
    )
    sender = ServerSender(settings)

    payload = {"image_path": str(image_path)}
    queued = sender._queue_payload(payload)

    assert queued.get("image_path") == str(image_path)


def test_queue_payload_drops_image_path_outside_shared_root(tmp_path):
    shared_root = tmp_path / "shared_storage"
    shared_root.mkdir()
    other_root = tmp_path / "other_storage"
    other_root.mkdir()
    image_path = other_root / "img.jpg"
    image_path.write_bytes(b"test")

    settings = EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
        offline_queue_dir=str(tmp_path / "offline_queue"),
        shared_storage_root=str(shared_root),
    )
    sender = ServerSender(settings)

    payload = {"image_path": str(image_path)}
    queued = sender._queue_payload(payload)

    assert "image_path" not in queued


def test_queue_payload_drops_missing_shared_image(tmp_path):
    shared_root = tmp_path / "shared_storage"
    shared_root.mkdir()
    image_path = shared_root / "missing.jpg"

    settings = EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
        offline_queue_dir=str(tmp_path / "offline_queue"),
        shared_storage_root=str(shared_root),
    )
    sender = ServerSender(settings)

    payload = {"image_path": str(image_path)}
    queued = sender._queue_payload(payload)

    assert "image_path" not in queued
