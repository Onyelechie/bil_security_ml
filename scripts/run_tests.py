from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _prepend_pythonpath(env: dict[str, str], paths: list[Path]) -> None:
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join(str(path) for path in paths)
    if existing:
        env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = prefix


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    project_root = Path(__file__).resolve().parents[1]

    env = os.environ.copy()
    _prepend_pythonpath(env, [project_root / "src", project_root])

    cmd = [sys.executable, "-m", "pytest", "-v", *argv]
    return subprocess.call(cmd, cwd=project_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
