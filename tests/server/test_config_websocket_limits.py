import pytest

from server.config import Settings
from tests.temp_dirs import repo_temp_dir


def test_debug_accepts_release_string():
    cfg = Settings(debug="release")
    assert cfg.debug is False


def test_auto_apply_migrations_defaults_to_true():
    cfg = Settings()
    assert cfg.auto_apply_migrations is True


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


def test_env_file_loads_admin_password_and_secret_key():
    with repo_temp_dir("settings-env-") as temp_dir:
        env_file = temp_dir / ".env"
        env_file.write_text(
            "ADMIN_PASSWORD=test-admin-pass\nSECRET_KEY=test-secret-key\n",
            encoding="utf-8",
        )

        cfg = Settings(_env_file=env_file)

        assert cfg.admin_password == "change-this-admin-password"
        assert cfg.secret_key == "change-this-secret-key"
