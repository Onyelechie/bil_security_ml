from datetime import datetime
from pathlib import Path

from src.server.config import settings
from src.server.db import SessionLocal
from src.server.schemas import AlertCreate, Detection
from src.server.services.alert_ingestion import AlertIngestionService
from tests.temp_dirs import repo_temp_dir


def test_ws_prefixed_path_is_accepted_and_stored():
    with repo_temp_dir("alert-ingest-") as temp_dir:
        saved = temp_dir / "ws_saved.png"
        saved.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data")

        svc = AlertIngestionService()
        alert = AlertCreate(
            site_id="site_test",
            camera_id="cam_x",
            timestamp=datetime.now(),
            detections=[Detection(class_="person", confidence=0.5)],
            image_path=f"ws://{saved.as_posix()}",
        )

        db = SessionLocal()
        try:
            db_alert = svc.ingest(db=db, alert=alert)
            assert db_alert.image_path is not None
            assert saved.as_posix() in db_alert.image_path
        finally:
            db.close()


def test_local_file_is_copied_into_storage(monkeypatch):
    with repo_temp_dir("alert-copy-") as temp_dir:
        src_file = temp_dir / "external.png"
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data")

        monkeypatch.setattr(settings, "image_storage_dir", "storage/test_alert_images")

        svc = AlertIngestionService()
        alert = AlertCreate(
            site_id="site-copy",
            camera_id="cam-copy",
            timestamp=datetime.now(),
            detections=[Detection(class_="person", confidence=0.7)],
            image_path=src_file.as_posix(),
        )

        db = SessionLocal()
        try:
            db_alert = svc.ingest(db=db, alert=alert)
            assert db_alert.image_path is not None
            assert "storage/test_alert_images" in db_alert.image_path
            repo_root = Path(__file__).resolve().parents[2]
            stored_path = (repo_root / Path(db_alert.image_path)).resolve()
            assert stored_path.is_file()
        finally:
            db.close()
