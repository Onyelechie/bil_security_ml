from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

from .config import settings
from .models.base import Base
from sqlalchemy import event

engine = create_engine(
    settings.database_url,
    connect_args=({"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}),
)

# Ensure SQLite enforces foreign key constraints at the connection level.
# PostgreSQL enforces FKs by default; SQLite requires the PRAGMA to be set per connection.
if settings.database_url.startswith("sqlite"):
    def _enable_sqlite_foreign_keys(dbapi_con, connection_record):
        try:
            cursor = dbapi_con.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception as exc:
            # Log a warning rather than silently swallowing the exception so
            # static analysis tools (Bandit) do not flag a bare pass and
            # operators have visibility into the failure.
            logging.getLogger(__name__).warning(
                "Could not enable SQLite foreign_keys PRAGMA on connect: %s", exc
            )

    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
