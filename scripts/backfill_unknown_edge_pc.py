"""Backfill and cleanup script for alerts with `edge_pc_id='unknown'`.

Usage examples:

# Dry-run showing counts
python scripts/backfill_unknown_edge_pc.py --dry-run

# Apply mappings from CSV (columns: site_id,camera_id,edge_pc_id)
python scripts/backfill_unknown_edge_pc.py --mapping mappings.csv

# Assign a default edge_pc_id to all unknown alerts
python scripts/backfill_unknown_edge_pc.py --assign-default edge-1234

# After mapping/assigning, remove sentinel if unused:
python scripts/backfill_unknown_edge_pc.py --cleanup-sentinel

This script connects to the app's configured DB using `src/server/db.py` engine.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import sqlalchemy as sa

# Ensure `src` is importable (we modify path before importing app modules)
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))


def read_mapping_csv(path: Path) -> Dict[Tuple[str, str], str]:
    """Read mapping CSV with columns: site_id,camera_id,edge_pc_id

    Returns mapping keyed by (site_id, camera_id) -> edge_pc_id
    """
    mapping: Dict[Tuple[str, str], str] = {}
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            site = row.get("site_id")
            camera = row.get("camera_id")
            edge = row.get("edge_pc_id")
            if not site or not camera or not edge:
                print(f"Skipping invalid mapping row: {row}")
                continue
            mapping[(site, camera)] = edge
    return mapping


def dry_run_report(conn: sa.engine.Connection) -> int:
    res = conn.execute(sa.text("SELECT COUNT(*) as c FROM alerts WHERE edge_pc_id = 'unknown'"))
    row = res.mappings().first()
    count = row["c"] if row else 0
    print(f"Alerts with edge_pc_id='unknown': {count}")
    return count


def apply_mappings(
    conn: sa.engine.Connection,
    mapping: Dict[Tuple[str, str], str],
    dry_run: bool = True,
) -> int:
    updated = 0
    for (site, camera), edge in mapping.items():
        # Count to report
        select_sql = (
            "SELECT COUNT(*) as c FROM alerts WHERE edge_pc_id = 'unknown' "
            "AND site_id = :site AND camera_id = :camera"
        )
        res = conn.execute(sa.text(select_sql), {"site": site, "camera": camera})
        cnt = res.mappings().first()["c"]
        if cnt == 0:
            print(f"No matching unknown alerts for site={site} camera={camera}")
            continue
        print(f"Mapping {cnt} alerts for site={site} camera={camera} -> edge_pc_id={edge}")
        if not dry_run:
            update_sql = (
                "UPDATE alerts SET edge_pc_id = :edge WHERE edge_pc_id = 'unknown' "
                "AND site_id = :site AND camera_id = :camera"
            )
            conn.execute(sa.text(update_sql), {"edge": edge, "site": site, "camera": camera})
        updated += cnt
    return updated


def assign_default(conn: sa.engine.Connection, default_edge: str, dry_run: bool = True) -> int:
    res = conn.execute(sa.text("SELECT COUNT(*) as c FROM alerts WHERE edge_pc_id = 'unknown'"))
    cnt = res.mappings().first()["c"]
    print(f"Assigning default edge_pc_id={default_edge} to {cnt} alerts")
    if cnt and not dry_run:
        update_sql = "UPDATE alerts SET edge_pc_id = :edge " "WHERE edge_pc_id = 'unknown'"
        conn.execute(sa.text(update_sql), {"edge": default_edge})
    return cnt


def cleanup_sentinel(conn: sa.engine.Connection, dry_run: bool = True) -> bool:
    # Delete sentinel if not referenced
    res = conn.execute(sa.text("SELECT COUNT(*) as c FROM alerts WHERE edge_pc_id = 'unknown'"))
    cnt = res.mappings().first()["c"]
    if cnt:
        print(f"Cannot remove sentinel: {cnt} alerts still reference 'unknown'")
        return False
    print("No alerts reference 'unknown'. Removing sentinel row from edge_pcs (if present)")
    if not dry_run:
        conn.execute(sa.text("DELETE FROM edge_pcs WHERE edge_pc_id = 'unknown'"))
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", type=Path, help="CSV file mapping (site_id,camera_id,edge_pc_id)")
    p.add_argument(
        "--assign-default",
        type=str,
        help="Assign this edge_pc_id to all unknown alerts",
    )
    p.add_argument(
        "--cleanup-sentinel",
        action="store_true",
        help="Remove sentinel 'unknown' from edge_pcs if unreferenced",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be changed without applying",
    )
    args = p.parse_args()

    # Import application DB after we've adjusted `sys.path` above
    from server import db as server_db

    engine = server_db.engine
    with engine.connect() as conn:
        print("Connected to DB")
        _ = dry_run_report(conn)
        total_updated = 0
        if args.mapping:
            if not args.mapping.exists():
                print(f"Mapping file not found: {args.mapping}")
                sys.exit(2)
            mapping = read_mapping_csv(args.mapping)
            updated = apply_mappings(conn, mapping, dry_run=args.dry_run)
            total_updated += updated
        if args.assign_default:
            updated = assign_default(conn, args.assign_default, dry_run=args.dry_run)
            total_updated += updated
        if args.cleanup_sentinel:
            ok = cleanup_sentinel(conn, dry_run=args.dry_run)
            print("Cleanup sentinel result:", ok)

        if args.dry_run:
            print("Dry run complete. No changes were applied.")
        else:
            print(f"Completed. Total alerts updated: {total_updated}")


if __name__ == "__main__":
    main()
