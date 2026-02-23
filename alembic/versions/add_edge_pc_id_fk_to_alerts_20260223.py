"""Add edge_pc_id column and FK to alerts

Revision ID: add_edge_pc_id_fk_to_alerts_20260223
Revises: add_indexes_to_alerts_20260223
Create Date: 2026-02-23 00:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "add_edge_pc_id_fk_to_alerts_20260223"
down_revision: Union[str, Sequence[str], None] = "add_indexes_to_alerts_20260223"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade: add `edge_pc_id` column, index and FK to `alerts`.

    Use batch_alter_table for SQLite compatibility (rebuild table).
    """
    # NOTE FOR MAINTAINERS:
    # The following `DROP INDEX IF EXISTS` is a best-effort guard to help local
    # developer workflows where a previous, partially-applied migration run
    # may have left temporary indexes or artifacts behind. It is present to
    # make the migration idempotent in local SQLite development environments.
    #
    # In production (Postgres, etc.) this guard is a no-op and can be kept for
    # safety. If you control all target DBs and prefer a cleaner migration
    # history, consider removing the guard and regenerating the migration
    # before a production release.
    try:
        op.execute("DROP INDEX IF EXISTS ix_alerts_edge_pc_id")
    except Exception:
        # best-effort drop (some sqlite versions may behave differently)
        pass

    with op.batch_alter_table("alerts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("edge_pc_id", sa.String(), nullable=True))
        # Some local environments may already have the index present due to
        # prior partial runs; creating the same index would raise an error.
        # We intentionally swallow that error here to avoid blocking local
        # developer work. Keep this behavior only if you need local idempotency.
        try:
            batch_op.create_index("ix_alerts_edge_pc_id", ["edge_pc_id"])
        except Exception:
            pass
        batch_op.create_foreign_key(
            "fk_alerts_edge_pc_id_edge_pcs",
            "edge_pcs",
            ["edge_pc_id"],
            ["edge_pc_id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    """Downgrade: drop FK, index, and column.

    Use batch_alter_table to rebuild table on SQLite.
    """
    with op.batch_alter_table("alerts", schema=None) as batch_op:
        batch_op.drop_constraint("fk_alerts_edge_pc_id_edge_pcs")
        batch_op.drop_index("ix_alerts_edge_pc_id")
        batch_op.drop_column("edge_pc_id")
