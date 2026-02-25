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
    # Ensure all model modules are imported so their tables are registered
    # with the declarative metadata before creating tables. This guards against
    # different import orders in tests or when the app is embedded.
    try:
        # Import models to ensure they register with Base.metadata
        # (these imports are idempotent)
        from .models import alert as _alert  # noqa: F401
        from .models import edge_pc as _edge_pc  # noqa: F401
    except Exception:
        # If model imports fail, proceed to create tables; errors will surface
        # during create_all or at runtime.
        logging.getLogger(__name__).debug("Could not import model modules before create_all")

    Base.metadata.create_all(bind=engine)

    # Ensure sentinel row exists so runtime fallback to 'edge-001' will not
    # violate FK constraints. This is idempotent.
    try:
        from sqlalchemy import text

        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
                        "VALUES ('edge-001', 'unknown', NULL, 'offline')"
                    )
                )
            else:
                # PostgreSQL and others: use ON CONFLICT DO NOTHING
                conn.execute(
                    text(
                        "INSERT INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
                        "VALUES ('edge-001', 'unknown', NULL, 'offline') "
                        "ON CONFLICT (edge_pc_id) DO NOTHING"
                    )
                )
    except Exception:
        logging.getLogger(__name__).exception("Failed to ensure sentinel 'edge-001' exists in database")
