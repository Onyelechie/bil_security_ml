"""Add received_at to alerts

Revision ID: 20260406_add_alert_received_at
Revises: 20260322_add_device_revocation
Create Date: 2026-04-06 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "20260406_add_alert_received_at"
down_revision = "20260322_add_device_revocation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("alerts"):
        return
    columns = {col["name"] for col in inspector.get_columns("alerts")}
    if "received_at" not in columns:
        op.add_column("alerts", sa.Column("received_at", sa.DateTime(), nullable=True))
        op.execute("UPDATE alerts SET received_at = timestamp WHERE received_at IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("alerts"):
        return
    columns = {col["name"] for col in inspector.get_columns("alerts")}
    if "received_at" in columns:
        op.drop_column("alerts", "received_at")
