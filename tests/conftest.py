# tests/conftest.py
import os
import sys
from pathlib import Path
import uuid

import pytest
from alembic import command
from alembic.config import Config

project_root = Path(__file__).resolve().parents[1]

# If you keep these, fine; but we can also do it via pytest.ini (see below)
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def _make_temp_sqlite_url(tmp_dir: Path) -> str:
    # Unique DB every run -> no “file in use” collisions
    db_path = tmp_dir / f"_pytest_{os.getpid()}_{uuid.uuid4().hex}.db"
    # sqlite URL wants forward slashes
    return "sqlite:///" + db_path.as_posix()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db(tmp_path_factory: pytest.TempPathFactory):
    """
    Apply Alembic migrations to the test database schema before any tests run.

    Rules:
    - If DATABASE_URL is set externally, use it (do NOT delete anything).
    - Otherwise create a unique sqlite DB in pytest's temp dir each run.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        tmp_dir = tmp_path_factory.getbasetemp()
        db_url = _make_temp_sqlite_url(Path(tmp_dir))
        os.environ["DATABASE_URL"] = db_url

    # Safety: never run tests against server.db
    if db_url.endswith("server.db"):
        raise RuntimeError(
            f"Refusing to run tests against server.db (DATABASE_URL={db_url}). "
            "Unset DATABASE_URL or point it to a dedicated test DB."
        )

    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_cfg, "head")
    yield
