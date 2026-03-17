import pytest

from server.config import Settings


def test_ws_worker_count_must_be_positive():
    with pytest.raises(ValueError, match="WS_ALERT_WORKER_COUNT"):
        Settings(ws_alert_worker_count=0)


def test_ws_queue_size_must_be_positive():
    with pytest.raises(ValueError, match="WS_ALERT_QUEUE_SIZE"):
        Settings(ws_alert_queue_size=0)


def test_ws_max_connections_must_be_positive():
    with pytest.raises(ValueError, match="WS_MAX_CONNECTIONS"):
        Settings(ws_max_connections=0)


def test_ws_max_image_bytes_must_be_positive():
    with pytest.raises(ValueError, match="WS_MAX_IMAGE_BYTES"):
        Settings(ws_max_image_bytes=0)


def test_ws_image_storage_dir_must_not_be_empty():
    with pytest.raises(ValueError, match="WS_IMAGE_STORAGE_DIR"):
        Settings(ws_image_storage_dir="   ")


def test_ws_image_retention_hours_must_be_positive():
    with pytest.raises(ValueError, match="WS_IMAGE_RETENTION_HOURS"):
        Settings(ws_image_retention_hours=0)


def test_ws_image_cleanup_interval_hours_must_be_positive():
    with pytest.raises(ValueError, match="WS_IMAGE_CLEANUP_INTERVAL_HOURS"):
        Settings(ws_image_cleanup_interval_hours=0)


def test_log_buffer_max_entries_must_be_positive():
    with pytest.raises(ValueError, match="LOG_BUFFER_MAX_ENTRIES"):
        Settings(log_buffer_max_entries=0)
