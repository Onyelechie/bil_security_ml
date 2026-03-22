# Changelog — Server branch (selected entries)

## 2026-03-21 — Server branch
- Feature: Per-site image storage with automatic directory creation on registration.
- Feature: `POST /api/alerts/upload` multipart HTTP ingestion endpoint.
- Fix: Ingestion normalization — server now copies local absolute image paths into configured storage and persists storage-relative paths.
- Feature: Per-site retention settings exposed via dashboard and background cleanup task.
- UX: Dashboard Settings view (full-width) and Overview improvements (connections, registered PCs, ports).
- Devops: Added `scripts/test_alert_upload.py` and `scripts/fix_alert_image.py` to help upload and remediate legacy alerts.

## 2026-03-10 — Earlier entries
- Initial project scaffolding and core WebSocket ingestion (`/ws/alerts`) implementation.

--
Generated from development work in March 2026. See `README.md` and `docs/` for detailed operational guidance.
