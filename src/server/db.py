from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

from .config import settings
from sqlalchemy import event, inspect, text

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
    """Perform runtime DB checks without managing schema.

    Schema creation/mutations are owned by Alembic migrations. This runtime
    initializer only performs lightweight safety setup that should be
    idempotent when schema already exists.
    """

    # Ensure sentinel row exists so runtime fallback to 'edge-001' will not
    # violate FK constraints. This is idempotent.
    try:
        with engine.begin() as conn:
            if not inspect(conn).has_table("edge_pcs"):
                logging.getLogger(__name__).warning(
                    "edge_pcs table not found; schema is not initialized. "
                    "Run `alembic upgrade head` before starting the server."
                )
                return

            conn.execute(
                text(
                    "INSERT INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
                    "SELECT 'edge-001', 'unknown', NULL, 'offline' "
                    "WHERE NOT EXISTS (SELECT 1 FROM edge_pcs WHERE edge_pc_id = 'edge-001')"
                )
            )
    except Exception:
        logging.getLogger(__name__).exception("Failed to ensure sentinel 'edge-001' exists in database")
