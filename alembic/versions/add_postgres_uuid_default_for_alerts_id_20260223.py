"""Add Postgres server-side UUID default for alerts.id

Revision ID: add_postgres_uuid_default_for_alerts_id_20260223
Revises: enforce_edge_pc_id_not_null_20260223
Create Date: 2026-02-23 21:40:00.000000

This migration sets a server-side default for `alerts.id` when running on
Postgres. It is a no-op on other dialects (SQLite) to keep local development
environments unaffected.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_postgres_uuid_default_for_alerts_id_20260223"
down_revision: Union[str, Sequence[str], None] = "enforce_edge_pc_id_not_null_20260223"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name
    # Only apply for Postgres
    if dialect in ("postgresql", "psycopg2"):
        # Ensure uuid-ossp extension is available (may require superuser on some hosts)
        op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        # Set default to uuid_generate_v4() cast to text (alerts.id is a text column)
        op.execute("ALTER TABLE alerts ALTER COLUMN id SET DEFAULT (uuid_generate_v4())::text;")


def downgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name
    if dialect in ("postgresql", "psycopg2"):
        op.execute("ALTER TABLE alerts ALTER COLUMN id DROP DEFAULT;")
