# PR: feat(migrations + config): rename sentinel to `edge-001`, add backfill script, and unify Alembic heads

## Summary

- Standardize sentinel/default `edge_pc_id` to `edge-001` and provide tooling to backfill and remediate historical rows.
- Move server configuration to environment-backed settings and update `.env.example`.
- Add Alembic scaffolding and a merge migration so `alembic upgrade head` runs reliably in CI.

## Key changes

- API: fallback to `edge-001` when `edge_pc_id` is missing (`src/server/routes/alerts.py`).
- Config: `src/server/config.py` reads from environment variables; `.env.example` updated.
- Migrations: added/updated Alembic revisions; `merge_heads_20260223` unifies parallel heads.
- Scripts: new `scripts/backfill_edge_001.py` for remediation and `scripts/cleanup_db.py` helper.
- CI/workflows: updated `.github/workflows/pr-discord-notify.yml` to avoid accidental execution of variable content (e.g. branch names like `edge-001`) by enabling strict shell flags and JSON-encoding payloads safely.
- Docs & CI: `README.md`, `alembic/README`, and CI workflow updated (run migrations before tests).

## Notable commits

- `b35f6d0` rename sentinel to 'edge-001' and add backfill script
- `15264e7` add merge migration to unify Alembic heads
- `0e5c226` remove deprecated backfill wrapper
- `db48b0b` set sentinel edge_pc_id across code, migrations and docs

## Migration & deployment notes

- Run migrations before deploying: `python -m alembic upgrade head` (merge revision ensures a single head).
- Audit and remediate `edge-001` references with `scripts/backfill_edge_001.py --dry-run` before running sentinel-removal in production.

## Tests

- Local test suite: 13 passed.

---

Tell me if you want this file committed to the `Server` branch, or if you want me to open the PR on GitHub using this description.
