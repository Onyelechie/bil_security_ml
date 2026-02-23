"""Deprecated wrapper for backward compatibility.

This file used to implement the backfill helper. The file has been
renamed to `scripts/backfill_edge_001.py`. This stub prints a deprecation
notice and delegates to the new script so older automation that calls the
original filename will continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

NEW = Path(__file__).with_name("backfill_edge_001.py")


def main() -> None:
    print(
        "WARNING: scripts/backfill_unknown_edge_pc.py is deprecated;"
        " use scripts/backfill_edge_001.py instead. Delegating..."
    )
    # Execute the new script in a subprocess-like manner by replacing argv
    sys.argv[0] = str(NEW)
    # Import and call the main from the new module
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from backfill_edge_001 import main as new_main  # type: ignore

    new_main()


if __name__ == "__main__":
    main()
