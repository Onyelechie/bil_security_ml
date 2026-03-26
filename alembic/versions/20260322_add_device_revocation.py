"""Add revoked_at and last_key_rotation_at to devices

Revision ID: 20260322_add_device_revocation
Revises: merge_heads_20260223
Create Date: 2026-03-22 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "20260322_add_device_revocation"
down_revision = "merge_heads_20260223"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("devices"):
        op.create_table(
            "devices",
            sa.Column("device_id", sa.String(length=128), nullable=False),
            sa.Column("public_key_b64", sa.Text(), nullable=False),
            sa.Column("enrolled_at", sa.DateTime(), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.Column("last_key_rotation_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("device_id"),
        )
        op.create_index(op.f("ix_devices_device_id"), "devices", ["device_id"], unique=False)
        return
    columns = {col["name"] for col in inspector.get_columns("devices")}
    if "revoked_at" not in columns:
        op.add_column("devices", sa.Column("revoked_at", sa.DateTime(), nullable=True))
    if "last_key_rotation_at" not in columns:
        op.add_column(
            "devices",
            sa.Column("last_key_rotation_at", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("devices"):
        return
    columns = {col["name"] for col in inspector.get_columns("devices")}
    if columns == {
        "device_id",
        "public_key_b64",
        "enrolled_at",
        "active",
        "revoked_at",
        "last_key_rotation_at",
    }:
        try:
            op.drop_index(op.f("ix_devices_device_id"), table_name="devices")
        except Exception:
            pass
        op.drop_table("devices")
        return
    if "last_key_rotation_at" in columns:
        op.drop_column("devices", "last_key_rotation_at")
    if "revoked_at" in columns:
        op.drop_column("devices", "revoked_at")
