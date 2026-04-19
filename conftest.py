import os
import sys
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def _make_temp_sqlite_url(tmp_dir: Path) -> str:
    db_path = tmp_dir / f"_pytest_{os.getpid()}_{uuid.uuid4().hex}.db"
    return "sqlite:///" + db_path.as_posix()


def _sqlite_db_name(db_url: str) -> str | None:
    if not isinstance(db_url, str):
        return None

    prefix = "sqlite:///"
    if not db_url.lower().startswith(prefix):
        return None

    raw_path = db_url[len(prefix):]
    return Path(raw_path).name.lower()


_TEST_DB_DIR = project_root / ".pytest-db"
_TEST_DB_DIR.mkdir(exist_ok=True)
_DEFAULT_TEST_DATABASE_URL = _make_temp_sqlite_url(_TEST_DB_DIR)

os.environ.setdefault("DATABASE_URL", _DEFAULT_TEST_DATABASE_URL)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    db_url = os.environ.get("DATABASE_URL", _DEFAULT_TEST_DATABASE_URL)

    if _sqlite_db_name(db_url) == "server.db":
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
    from server.config import settings
    from server.main import app
    from server.services.image_storage import ImageStorageService

    storage = ImageStorageService(settings.image_storage_dir)
    storage.ensure_ready()
    app.state.image_storage = storage
    app.state.ws_image_storage = storage
    app.state.ws_max_image_bytes = settings.ws_max_image_bytes
    yield
