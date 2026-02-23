"""Enforce non-null edge_pc_id on alerts (backfill)

Revision ID: enforce_edge_pc_id_not_null_20260223
Revises: add_edge_pc_id_fk_to_alerts_20260223
Create Date: 2026-02-23 21:15:00.000000

This migration:
 - ensures a sentinel edge_pc row ('001') exists
 - backfills existing alerts with '001' where edge_pc_id IS NULL
 - alters the column to be NOT NULL using batch mode (SQLite-safe)

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "enforce_edge_pc_id_not_null_20260223"
down_revision: Union[str, Sequence[str], None] = "add_edge_pc_id_fk_to_alerts_20260223"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Backfill alerts.edge_pc_id and enforce NOT NULL."""
    conn = op.get_bind()

    # NOTE FOR MAINTAINERS:
    # We create a sentinel `edge_pcs` row with `edge_pc_id='edge-001'` and
    # backfill alerts that lack an `edge_pc_id` to that sentinel. This is a
    # pragmatic migration strategy to preserve historical alerts and avoid
    # blocking upgrades for deployed installations where edge agents have
    # not yet been updated to provide `edge_pc_id`.
    #
    # The sentinel is intended as a temporary compatibility measure. After
    # agents are updated, consider adding a follow-up job to re-assign
    # better provenance values or to remove/aggregate sentinel-marked alerts
    # if appropriate for your analytics expectations.
    insert_sql = (
        "INSERT OR IGNORE INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
        "VALUES ('edge-001', 'unknown', NULL, 'offline')"
    )
    conn.execute(sa.text(insert_sql))

    # Backfill alerts where edge_pc_id is null
    update_sql = "UPDATE alerts SET edge_pc_id = 'edge-001' " "WHERE edge_pc_id IS NULL"
    conn.execute(sa.text(update_sql))

    # Alter column to NOT NULL using batch_alter_table for SQLite
    with op.batch_alter_table("alerts", schema=None) as batch_op:
        batch_op.alter_column(
            "edge_pc_id",
            existing_type=sa.String(),
            nullable=False,
        )


def downgrade() -> None:
    """Revert NOT NULL enforcement (make column nullable again)."""
    with op.batch_alter_table("alerts", schema=None) as batch_op:
        batch_op.alter_column(
            "edge_pc_id",
            existing_type=sa.String(),
            nullable=True,
        )

    # Attempt to remove sentinel if it is no longer referenced by any alert.
    # This is a best-effort cleanup for downgrades; in practice you may want
    # to run a dedicated cleanup job once agents are updated.
    conn = op.get_bind()
    delete_sql = (
        "DELETE FROM edge_pcs WHERE edge_pc_id = 'edge-001' AND NOT EXISTS ("
        "SELECT 1 FROM alerts WHERE edge_pc_id = 'edge-001')"
    )
    conn.execute(sa.text(delete_sql))
