# tests/conftest.py
import os
import sys
import uuid
from pathlib import Path

import pytest

from alembic import command
from alembic.config import Config

project_root = Path(__file__).resolve().parents[1]

# Keep project imports working during test collection.
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def _make_temp_sqlite_url(tmp_dir: Path) -> str:
    # Unique DB every run -> no "file in use" collisions.
    db_path = tmp_dir / f"_pytest_{os.getpid()}_{uuid.uuid4().hex}.db"
    return "sqlite:///" + db_path.as_posix()


_TEST_DB_DIR = project_root / ".pytest-db"
_TEST_DB_DIR.mkdir(exist_ok=True)
_DEFAULT_TEST_DATABASE_URL = _make_temp_sqlite_url(_TEST_DB_DIR)

# Set a dedicated test database before test modules import application settings/engines.
os.environ.setdefault("DATABASE_URL", _DEFAULT_TEST_DATABASE_URL)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Apply Alembic migrations to the test database schema before any tests run.

    Rules:
    - If DATABASE_URL is set externally, use it (do NOT delete anything).
    - Otherwise use a unique sqlite DB in a dedicated temp directory for this run.
    """
    db_url = os.environ.get("DATABASE_URL", _DEFAULT_TEST_DATABASE_URL)

    if db_url.endswith("server.db"):
        raise RuntimeError(
            f"Refusing to run tests against server.db (DATABASE_URL={db_url}). "
            "Unset DATABASE_URL or point it to a dedicated test DB."
        )

    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_cfg, "head")
    yield


@pytest.fixture(scope="session", autouse=True)
def setup_test_app_state(setup_test_db):
    """
    Initialize the app state needed by tests that import TestClient(app)
    without entering the client context manager.
    """
    from server.config import settings
    from server.main import app
    from server.services.image_storage import ImageStorageService

    storage = ImageStorageService(settings.image_storage_dir)
    storage.ensure_ready()
    app.state.image_storage = storage
    app.state.ws_image_storage = storage
    app.state.ws_max_image_bytes = settings.ws_max_image_bytes
    yield
