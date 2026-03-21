# tests/test_sender.py
import time
from unittest.mock import MagicMock

import pytest
import requests

from edge_agent.config import EdgeSettings
from edge_agent.sender import ServerSender


@pytest.fixture
def settings() -> EdgeSettings:
    """Provides a standard EdgeSettings for testing."""
    return EdgeSettings(
        server_base_url="http://mock-server",
        edge_pc_id="test-edge-1",
        site_name="Test Site",
        site_id="site-1",
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
    json_body = kwargs["json"]

    assert url == "http://mock-server/api/heartbeat"
    assert json_body["edge_pc_id"] == "test-edge-1"
    assert json_body["status"] == "starting"
    assert "timestamp" in json_body

    # Optional: ensure raise_for_status was used
    mock_resp.raise_for_status.assert_called_once()


def test_send_heartbeat_with_uptime(sender: ServerSender):
    """Test that providing a start time includes uptime_seconds (timing-tolerant)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    started = time.monotonic() - 10
    sender.send_heartbeat(started_monotonic=started)

    _, kwargs = sender._session.post.call_args
    json_body = kwargs["json"]

    assert "uptime_seconds" in json_body
    assert 10 <= json_body["uptime_seconds"] < 12  # Allow small buffer


def test_send_heartbeat_reflects_status_change(sender: ServerSender):
    """Test that status updates are reflected in the heartbeat payload."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    sender.set_status("online")
    sender.send_heartbeat()

    _, kwargs = sender._session.post.call_args
    json_body = kwargs["json"]

    assert json_body["status"] == "online"


def test_send_alert_structure(sender: ServerSender):
    """Test that send_alert sends a server-compatible payload to the correct URL."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    sender._session.post.return_value = mock_resp

    detections = [{"class": "person", "confidence": 0.9}]
    success = sender.send_alert(camera_id="cam-1", detections=detections)

    assert success is True
    sender._session.post.assert_called_once()

    # Check that the payload sent matches the AlertCreate schema
    args, kwargs = sender._session.post.call_args
    assert args[0] == "http://mock-server/api/alerts"
    assert "json" in kwargs
    payload = kwargs["json"]
    assert payload["site_id"] == "site-1"
    assert payload["edge_pc_id"] == "test-edge-1"
    assert payload["camera_id"] == "cam-1"
    assert payload["detections"] == detections
    assert "timestamp" in payload
    mock_resp.raise_for_status.assert_called_once()


def test_send_heartbeat_handles_server_error(sender, mocker):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    sender._session.post.return_value = mock_resp

    mocked_logger = mocker.patch("edge_agent.sender.logger")

    success = sender.send_heartbeat()

    assert success is False
    mocked_logger.error.assert_called()  # or .exception if you switch to exception()
    # Optional: assert on message content
    assert any(
        "Failed to send heartbeat" in str(call.args[0])
        for call in mocked_logger.error.call_args_list
    )


def test_send_alert_handles_request_exception(sender, mocker):
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


def test_send_alert_rejects_invalid_detections(sender, mocker):
    mocked_logger = mocker.patch("edge_agent.sender.logger")

    success = sender.send_alert(camera_id="cam-3", detections=[{"class": "person"}])

    assert success is False
    sender._session.post.assert_not_called()
    mocked_logger.error.assert_called()
