"""Remove sentinel '001' from edge_pcs when safe

Revision ID: remove_unknown_sentinel_20260223
Revises: enforce_edge_pc_id_not_null_20260223
Create Date: 2026-02-23 22:00:00.000000

This migration provides a safe, idempotent step to remove the sentinel
`edge_pcs` row with `edge_pc_id = '001'` if it is no longer referenced
by any `alerts` rows. It is intended to be run only after an audit confirms
there are no alerts still pointing at the sentinel (or after those alerts
have been remediated with mapping/assignment).

If alerts continue to reference the sentinel the migration will log a
message and be a no-op.
"""

from typing import Sequence, Union

import os
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "remove_unknown_sentinel_20260223"
down_revision: Union[str, Sequence[str], None] = "enforce_edge_pc_id_not_null_20260223"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    # This migration is intentionally guarded. Removing the sentinel automatically
    # as part of the normal `upgrade head` chain is unsafe for fresh installs
    # (there are zero alerts, causing the sentinel to be removed and breaking
    # FK-backed inserts). To perform removal, the operator must explicitly set
    # the environment variable `ALLOW_REMOVE_EDGE_SENTINEL=1` before running
    # the migration. This keeps the default upgrade path safe.

    if os.environ.get("ALLOW_REMOVE_EDGE_SENTINEL") != "1":
        print("Skipping sentinel removal: set ALLOW_REMOVE_EDGE_SENTINEL=1 to enable manual sentinel cleanup")
        return

    # Check whether any alerts still reference the sentinel
    res = conn.execute(sa.text("SELECT COUNT(*) as c FROM alerts WHERE edge_pc_id = 'edge-001'"))
    row = res.mappings().first()
    cnt = row["c"] if row else 0
    if cnt:
        print(f"Found {cnt} alerts still referencing 'edge-001'; skipping sentinel removal")
        return

    # No references remain; remove sentinel row if present
    conn.execute(sa.text("DELETE FROM edge_pcs WHERE edge_pc_id = 'edge-001'"))


def downgrade() -> None:
    # Re-create the sentinel row if someone downgrades; this mirrors the
    # original compatibility approach used during backfill.
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "INSERT INTO edge_pcs (edge_pc_id, site_name, last_heartbeat, status) "
            "SELECT 'edge-001', 'unknown', NULL, 'offline' "
            "WHERE NOT EXISTS (SELECT 1 FROM edge_pcs WHERE edge_pc_id = 'edge-001')"
        )
    )
