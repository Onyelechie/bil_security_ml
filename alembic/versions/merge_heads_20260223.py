"""Merge multiple heads into a single head

Revision ID: merge_heads_20260223
Revises: add_postgres_uuid_default_for_alerts_id_20260223, remove_unknown_sentinel_20260223
Create Date: 2026-02-23 22:30:00.000000

This is an empty merge migration which stitches together two parallel
heads created during development so that `alembic upgrade head` can run
without requiring callers to specify multiple heads. The migration is
idempotent and contains no schema operations.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "merge_heads_20260223"
down_revision: Union[str, Sequence[str], None] = (
    "add_postgres_uuid_default_for_alerts_id_20260223",
    "remove_unknown_sentinel_20260223",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This is a no-op merge migration; it exists to unify multiple heads.
    pass


def downgrade() -> None:
    # Downgrading a merge migration is a no-op because it only records
    # repository history (it does not change schema state).
    pass
