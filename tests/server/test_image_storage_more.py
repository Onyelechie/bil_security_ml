import os
from datetime import datetime, timedelta, timezone

from src.server.services.image_storage import ImageStorageService
from tests.temp_dirs import repo_temp_dir

PNG_1X1 = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"
)


def test_save_alert_image_and_guess_extension():
    with repo_temp_dir("image-storage-more-") as temp_dir:
        root = temp_dir / "images"
        svc = ImageStorageService(str(root))

        path = svc.save_alert_image(
            site_id="My Site!",
            camera_id="Cam#1",
            edge_pc_id="edge-01",
            image_bytes=PNG_1X1,
            detections=[
                {"class": "person", "confidence": 0.73},
            ],
        )

        path_p = (root / os.path.basename(path)).resolve()
        assert path_p.as_posix().startswith(root.resolve().as_posix())
        assert path_p.suffix == ".png"
        assert os.path.isfile(path)
        assert "My_Site" in os.path.basename(path)
        assert "Cam_1" in os.path.basename(path)


def test_cleanup_older_than_removes_files():
    with repo_temp_dir("image-storage-more-") as temp_dir:
        root = temp_dir / "images"
        root.mkdir()
        svc = ImageStorageService(str(root))

        recent = root / "recent.png"
        recent.write_bytes(PNG_1X1)

        old = root / "old.png"
        old.write_bytes(PNG_1X1)
        old_time = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
        os.utime(old, (old_time, old_time))

        site_dir = root / "site_foo"
        site_dir.mkdir()
        child_old = site_dir / "c_old.png"
        child_old.write_bytes(PNG_1X1)
        os.utime(child_old, (old_time, old_time))

        removed = svc.cleanup_older_than(hours=24)
        assert removed == 2
        assert recent.exists()
        assert not old.exists()
        assert not child_old.exists()
