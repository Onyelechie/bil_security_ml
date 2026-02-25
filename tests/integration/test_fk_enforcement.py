import os
import sqlite3
from alembic.config import Config
from alembic import command


def test_fk_enforcement_with_migrations(tmp_path):
    db_file = tmp_path / "ci_fk_test.db"
    db_url = f"sqlite:///{db_file}"
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url

    # Run alembic upgrade head programmatically so we don't rely on console scripts
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")

    # Connect with foreign_keys pragma on and verify sentinel exists, and insert succeeds
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM edge_pcs WHERE edge_pc_id='edge-001'")
    count = cur.fetchone()[0]
    assert count > 0, "Sentinel edge-001 should exist after migrations"

    # Insert an alert referencing the sentinel; should succeed with FK enforcement
    cur.execute(
        (
            "INSERT INTO alerts(id, site_id, camera_id, timestamp, detections, "
            "image_path, edge_pc_id) VALUES(?, ?, ?, ?, ?, ?, ?)"
        ),
        ("ci-test", "s", "c", "2026-02-24 00:00:00", "[]", None, "edge-001"),
    )
    conn.commit()
    conn.close()
