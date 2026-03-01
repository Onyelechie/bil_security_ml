#!/usr/bin/env python3
"""Manual helper to remove the 'edge-001' sentinel only when safe.

This script checks whether any alerts reference 'edge-001'. If none do, it
will remove the sentinel from `edge_pcs` when run with `--yes`.

Usage:
  python scripts/remove_sentinel_if_safe.py --dry-run
  python scripts/remove_sentinel_if_safe.py --yes

This is intentionally a manual operation; the migration that *can* remove
sentinel is now guarded by the ALLOW_REMOVE_EDGE_SENTINEL env var. Use this
script for controlled, auditable cleanup.
"""
from __future__ import annotations

import argparse
import sys

import sqlalchemy as sa
from server.db import engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Actually delete the sentinel if safe")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete; just report")
    args = parser.parse_args()

    conn = engine.connect()
    try:
        res = conn.execute(sa.text("SELECT COUNT(*) as c FROM alerts WHERE edge_pc_id = 'edge-001'"))
        row = res.mappings().first()
        cnt = int(row["c"]) if row else 0
        print(f"Alerts referencing 'edge-001': {cnt}")
        if cnt > 0:
            print("Cannot remove sentinel: some alerts still reference 'edge-001'.")
            return 2

        print("No alerts reference 'edge-001'.")
        if args.dry_run or not args.yes:
            print("Dry-run or not confirmed; no changes made. Re-run with --yes to delete the sentinel.")
            return 0

        # Perform deletion
        conn.execute(sa.text("DELETE FROM edge_pcs WHERE edge_pc_id = 'edge-001'"))
        print("Sentinel 'edge-001' removed from edge_pcs.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
