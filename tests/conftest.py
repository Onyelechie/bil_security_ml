import sys
import os
import pytest
from sqlalchemy import create_engine
from alembic.config import Config
from alembic import command

# Add src and project root to path for all tests
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from server.config import settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Apply Alembic migrations to setup the test database schema."""
    # Run Alembic migrations to populate tables
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    yield
