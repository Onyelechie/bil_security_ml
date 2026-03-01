"""Add indexes to alerts table

Revision ID: add_indexes_to_alerts_20260223
Revises: 6ee722e50a95
Create Date: 2026-02-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_indexes_to_alerts_20260223"
down_revision: Union[str, Sequence[str], None] = "6ee722e50a95"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create indexes on alerts."""
    op.create_index("ix_alerts_timestamp", "alerts", ["timestamp"])
    op.create_index("ix_alerts_site_id", "alerts", ["site_id"])
    op.create_index("ix_alerts_camera_id", "alerts", ["camera_id"])
    op.create_index("ix_alerts_site_timestamp", "alerts", ["site_id", "timestamp"])


def downgrade() -> None:
    """Downgrade schema: drop alerts indexes."""
    op.drop_index("ix_alerts_site_timestamp", table_name="alerts")
    op.drop_index("ix_alerts_camera_id", table_name="alerts")
    op.drop_index("ix_alerts_site_id", table_name="alerts")
    op.drop_index("ix_alerts_timestamp", table_name="alerts")
