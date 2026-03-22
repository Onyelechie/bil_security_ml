import os
from pathlib import Path
from datetime import datetime

from src.server.services.alert_ingestion import AlertIngestionService
from src.server.services.image_storage import ImageStorageService
from src.server.schemas import AlertCreate, Detection
from src.server.db import SessionLocal
from src.server.config import settings


def test_ws_prefixed_path_is_accepted_and_stored(tmp_path):
    # create a file that simulates a websocket-saved image
    saved = tmp_path / "ws_saved.png"
    saved.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data")

    svc = AlertIngestionService()
    # prepare an alert with image_path prefixed with ws://
    alert = AlertCreate(
        site_id="site_test",
        camera_id="cam_x",
        timestamp=datetime.now(),
        detections=[Detection(class_="person", confidence=0.5)],
        image_path=f"ws://{saved.as_posix()}",
    )

    # Use a DB session from test setup (alembic migrations applied by conftest)
    db = SessionLocal()
    try:
        db_alert = svc.ingest(db=db, alert=alert)
        # image_path should be the absolute path to the saved file
        assert db_alert.image_path is not None
        assert saved.as_posix() in db_alert.image_path
    finally:
        db.close()


def test_local_file_is_copied_into_storage(tmp_path, monkeypatch):
    # Create a source file outside repo
    src_file = tmp_path / "external.png"
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data")

    # Set storage to a relative path under repo so ingestion will produce a repo-relative path
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
        # stored path should contain the configured storage directory
        assert "storage/test_alert_images" in db_alert.image_path
        # file should exist on disk (resolve relative to repo root)
        # the ingestion code stores a repo-relative path when possible
        # repo root is two levels up from tests/server/<file>
        repo_root = Path(__file__).resolve().parents[2]
        stored_path = (repo_root / Path(db_alert.image_path)).resolve()
        assert stored_path.is_file()
    finally:
        db.close()
