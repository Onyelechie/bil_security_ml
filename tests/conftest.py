import os
import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

# ---------- Import paths (optional now if pytest.ini has pythonpath, but harmless) ----------
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# ---------- Test DB setup ----------
DEFAULT_TEST_DB_URL = "sqlite:///./_pytest.db"


def _pick_test_db_url() -> str:
    """
    Choose the DB URL used by tests.
    - If CI (or you) set DATABASE_URL, we respect it.
    - Otherwise we force a local sqlite test DB file.
    """
    return os.environ.get("DATABASE_URL") or DEFAULT_TEST_DB_URL


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Apply Alembic migrations to the *test* database schema before any tests run.

    Key goal: ensure migrations run against the same DATABASE_URL the server code uses in tests,
    so we never see 'no such table' on a fresh DB.
    """
    db_url = _pick_test_db_url()
    os.environ["DATABASE_URL"] = db_url

    # Safety: never migrate the real dev DB by accident
    if db_url.endswith("server.db"):
        raise RuntimeError(
            f"Refusing to run tests against server.db (DATABASE_URL={db_url}). "
            f"Unset DATABASE_URL or set it to a dedicated test DB."
        )

    # If we’re using the default local test DB file, start fresh each run
    if db_url == DEFAULT_TEST_DB_URL:
        db_path = project_root / "_pytest.db"
        if db_path.exists():
            db_path.unlink()

    alembic_cfg = Config(str(project_root / "alembic.ini"))
    # Force Alembic to use the exact same URL tests will use
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_cfg, "head")
    yield
