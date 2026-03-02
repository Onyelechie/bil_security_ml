from server.config import Settings


def test_ws_worker_count_must_be_positive():
    try:
        Settings(ws_alert_worker_count=0)
        assert False, "Expected ValueError for ws_alert_worker_count=0"
    except ValueError as exc:
        assert "WS_ALERT_WORKER_COUNT" in str(exc)


def test_ws_queue_size_must_be_positive():
    try:
        Settings(ws_alert_queue_size=0)
        assert False, "Expected ValueError for ws_alert_queue_size=0"
    except ValueError as exc:
        assert "WS_ALERT_QUEUE_SIZE" in str(exc)


def test_ws_max_connections_must_be_positive():
    try:
        Settings(ws_max_connections=0)
        assert False, "Expected ValueError for ws_max_connections=0"
    except ValueError as exc:
        assert "WS_MAX_CONNECTIONS" in str(exc)


def test_ws_max_image_bytes_must_be_positive():
    try:
        Settings(ws_max_image_bytes=0)
        assert False, "Expected ValueError for ws_max_image_bytes=0"
    except ValueError as exc:
        assert "WS_MAX_IMAGE_BYTES" in str(exc)


def test_ws_image_storage_dir_must_not_be_empty():
    try:
        Settings(ws_image_storage_dir="   ")
        assert False, "Expected ValueError for empty ws_image_storage_dir"
    except ValueError as exc:
        assert "WS_IMAGE_STORAGE_DIR" in str(exc)
