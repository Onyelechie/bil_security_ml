import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from server.services.image_storage import ImageStorageService


def test_cleanup_older_than_removes_only_expired_files(tmp_path):
    storage = ImageStorageService(str(tmp_path))
    old_path = Path(
        storage.save_alert_image(
            site_id="site_old",
            camera_id="cam_old",
            image_bytes=b"\x89PNG\r\n\x1a\nold",
        )
    )
    fresh_path = Path(
        storage.save_alert_image(
            site_id="site_fresh",
            camera_id="cam_fresh",
            image_bytes=b"\x89PNG\r\n\x1a\nfresh",
        )
    )

    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    old_mtime = (now - timedelta(hours=25)).timestamp()
    fresh_mtime = (now - timedelta(hours=1)).timestamp()
    os.utime(old_path, (old_mtime, old_mtime))
    os.utime(fresh_path, (fresh_mtime, fresh_mtime))

    removed = storage.cleanup_older_than(hours=24, now=now)

    assert removed == 1
    assert not old_path.exists()
    assert fresh_path.exists()


def test_cleanup_older_than_requires_positive_hours(tmp_path):
    storage = ImageStorageService(str(tmp_path))
    with pytest.raises(ValueError):
        storage.cleanup_older_than(hours=0)


def test_save_alert_image_sanitizes_filename_parts(tmp_path):
    storage = ImageStorageService(str(tmp_path))
    saved_path = Path(
        storage.save_alert_image(
            site_id="../../site a",
            camera_id="cam\\x/../id",
            image_bytes=b"\xff\xd8\xffjpeg",
        )
    )

    assert saved_path.parent == tmp_path
    assert ".." not in saved_path.name
    assert "/" not in saved_path.name
    assert "\\" not in saved_path.name
    assert saved_path.suffix == ".jpg"
