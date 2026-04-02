# bil_security_ml

## On-Device Intrusion Detection and False Alarm Filtering

This project is part of COMP 4560: Industrial Project (Winter 2026) in collaboration with BIL Security.

### Project Overview

Security monitoring systems often generate false alarms due to environmental factors such as weather, vegetation, or animals. This project develops an on-device alarm filtering solution that analyzes live camera feeds to detect meaningful intrusion events (presence of people or vehicles) while filtering out non-critical motion.

The system operates on constrained hardware (Windows PCs with i5/i7 processors and ~4GB RAM) and supports multiple simultaneous camera feeds accessed via RTSP.

### Objectives

- Investigate computer vision and AI techniques for on-device intrusion detection
- Filter false positives caused by weather, animals, or vegetation
- Support configurable sensitivity and monitoring zones
- Evaluate performance under hardware constraints

### Team Members

- Stephen Ugbah
- Bhavik Jain
- Subhash Yadav
- Ebere Onyelechie

### Methodology

The project follows a structured approach:

1. **Research and Setup** (Jan 6 - Jan 24, 2026): Initial research on intrusion detection techniques, setup development environment
2. **Research (continued)** (Jan 25 - Feb 7, 2026): Evaluate AI models and finalize architecture
3. **Prototype Development** (Feb 8 - Feb 28, 2026): Implement detection pipeline and performance profiling
4. **Evaluation and Refinement** (Mar 1 - Mar 21, 2026): Test with real data and optimize
5. **Finishing Touches** (Mar 22 - Apr 6, 2026): Finalize documentation and presentation

### Deliverables

- Working proof-of-concept system
- Open-source source code
- Technical documentation
- Final presentation

### Technologies

- Computer Vision
- Machine Learning / AI
- RTSP stream processing
- On-device inference on constrained hardware

### Development Setup

### Configuration (.env)

Before running the server or migrations, copy the example environment file and update values for your environment:

```powershell
copy .env.example .env
# On Unix/macOS: cp .env.example .env
```

Important variables (see `.env.example`): `DATABASE_URL`, `HOST`, `PORT`, `DEBUG`, `SECRET_KEY`, `ADMIN_PASSWORD`, `CORS_ORIGINS` (comma-separated), `WS_MAX_CONNECTIONS`, `WS_ALERT_QUEUE_SIZE`, `WS_ALERT_WORKER_COUNT`, `WS_MAX_IMAGE_BYTES`, `WS_IMAGE_STORAGE_DIR`, `WS_IMAGE_RETENTION_HOURS`, `WS_IMAGE_CLEANUP_INTERVAL_HOURS`, `LOG_BUFFER_MAX_ENTRIES`.

Important startup note:
- `.env` values are loaded when the server process starts. If you change `ADMIN_PASSWORD`, `SECRET_KEY`, or other env-backed settings, restart uvicorn before testing login or protected routes.

Current edge auth model:
- Edge heartbeat and HTTP alerts must be signed by an enrolled device.
- `DEVICE_ID` must match `EDGE_PC_ID`.
- The edge PC receives the private key at provisioning time; the matching public key is enrolled on the server.
- See [docs/AUTHENTICATION.md](/c:/Users/ebere/Documents/bil_security_ml/docs/AUTHENTICATION.md) for the exact workflow.

### Edge Agent Configuration (.env)

Copy `.env.example` and edit `.env`. **Do not commit credentials**.

Edge Agent key variables (see `.env.example`):

**Core**
- `SITE_ID`, `EDGE_PC_ID`, `DEVICE_ID`, `SITE_NAME`
- `TCP_HOST`, `TCP_PORT` (external motion events)
- `SERVER_BASE_URL` (central server)
- `DEVICE_PRIVATE_KEY_B64` (required for signed heartbeat/alerts)

Provisioning rule:
- `DEVICE_ID` must equal `EDGE_PC_ID`, and the matching public key must be enrolled on the server before the edge agent starts.

**Trigger source selection**
- `ENABLE_TCP_MOTION` (`true`/`false`) - consume external TCP/VMS motion events
- `ENABLE_LOCAL_MOTION` (`true`/`false`) - compute motion locally from RTSP frames

You may enable either source or both. Both feed the same downstream incident / extraction / inference / alert pipeline.

**Trigger control (rate limit / dedupe)**
- `TRIGGER_COOLDOWN_SEC` (e.g., `10`)
- `TRIGGER_MERGE_WINDOW_SEC` (e.g., `2.0`)

**RTSP ingest (low-res stream for analysis/inference)**
- `RTSP_URL_LOW` (low stream, e.g. `/Streaming/Channels/102`)
- `RING_BUFFER_SECONDS` (recommended single-camera starting point: `25`)
- `ANALYSIS_FPS` (recommended: `5`)
- `FRAME_WIDTH`, `FRAME_HEIGHT` (recommended: `640x360`)

**Frame handling**
- The edge agent now keeps a **low-resolution color ring buffer** for extraction/inference.
- Local motion detection still runs on **grayscale**, derived from the buffered color frames.
- This keeps motion detection cheap while improving YOLO classification quality versus grayscale-only inference.

**Local motion trigger**
- `MOTION_FPS` (e.g., `1` to `2`)
- `MOTION_PIXEL_DELTA` (e.g., `10` to `25`)
- `MOTION_THRESHOLD` (e.g., `0.003` to `0.01`)
- `DEFAULT_CAMERA_ID` (used for local motion labeling in single-camera mode)

**Incident + extraction tuning**
- `INCIDENT_QUIET_SEC`
- `INCIDENT_MAX_SEC`
- `WINDOW_PRE_SEC`
- `WINDOW_POST_SEC`
- `WINDOW_TARGET_FPS`
- `WINDOW_MAX_FRAMES`
- `WINDOW_WAIT_GRACE_SEC`

Recommended indoor single-camera starting profile:
- `RING_BUFFER_SECONDS=25`
- `INCIDENT_MAX_SEC=12.0`
- `WINDOW_PRE_SEC=1.5`
- `WINDOW_POST_SEC=4.0`
- `WINDOW_TARGET_FPS=5.0`
- `WINDOW_MAX_FRAMES=40`

**Detector configuration**
- `DETECTOR_MODEL` (e.g. `YOLOv8-Small`)
- `DETECTOR_WEIGHTS` (optional explicit path)
- `DETECTOR_PERSON_CONF` (recommended indoor starting point: `0.40`)
- `DETECTOR_VEHICLE_CONF` (recommended indoor starting point: `0.90`)
- `DETECTOR_ALLOWED_CLASSES` (comma-separated, e.g. `person` or `person,vehicle`)

Notes:
- For cluttered indoor cameras, `DETECTOR_ALLOWED_CLASSES=person` is usually the best starting profile.
- The evaluator prefers a valid **person** detection over a valid **vehicle** detection when both appear in the same extracted window.

**Shared storage (optional)**
- `SHARED_STORAGE_ROOT` (absolute path to a shared mount visible to both edge and server)
- When set, alert snapshots saved under this root can be referenced by `image_path` and displayed by the server/dashboard.
- Queued alerts keep `image_path` only if the file exists and resolves under this root.
- Queued alerts that are invalid JSON or rejected with 4xx are quarantined under `OFFLINE_QUEUE_DIR/bad/`.
- Quarantined payloads are deleted after `QUEUE_QUARANTINE_RETENTION_DAYS` (default `7`).

FFmpeg note:
- The edge agent uses `imageio-ffmpeg`, which provides an ffmpeg binary automatically (no separate system ffmpeg install required).

Matplotlib note (accuracy eval):
- If `accuracy/eval_accuracy.py` hangs or errors with a Matplotlib cache permission issue on Windows,
  set `MPLCONFIGDIR` to a writable folder, e.g. `$env:MPLCONFIGDIR="$env:TEMP\mpl-cache"`.

### Database Migrations

This project uses Alembic for database schema migrations. If you change any models, you must generate and apply a migration:

1. **Generate migration script:**
  ```bash
  .venv\Scripts\activate  # On Windows
  python -m alembic revision --autogenerate -m "Describe your change"
  ```
2. **Apply migration:**
  ```bash
  python -m alembic upgrade head
  ```

This ensures your database schema matches your models. See the `alembic/` folder for migration scripts.

CI note: The repository's CI workflow runs database migrations before running tests so
the test environment mirrors migration state used locally. Ensure migrations are
committed before opening a PR.

### Notes about databases and migrations

- SQLite is used as the default development database for convenience. It is NOT recommended for production deployments where durability, concurrency, and advanced SQL features are required. For production, prefer PostgreSQL or another production-grade RDBMS and set `DATABASE_URL` accordingly.
- The `detections` column uses SQLAlchemy's `JSON` type; behavior differs by database. On SQLite it will be stored as text - JSON operators are available on PostgreSQL but not on SQLite.
- If you are migrating an existing database into Alembic-managed migrations, follow these steps:
  1. **Backup your DB** (copy the SQLite file or take a dump for other DBs).
  2. If the existing schema already matches models and you don't need to run migrations, mark the DB as up-to-date with:
    ```bash
    python -m alembic stamp head
    ```
  3. If you need to apply new migrations, review autogenerated migration scripts carefully before applying, then:
    ```bash
    python -m alembic upgrade head
    ```

Security note

- Do not run the server in production with an empty or placeholder `SECRET_KEY` (for example, `your-secret-key-here`). The server warns at startup if `DEBUG` is `False` and `SECRET_KEY` is empty or still set to the placeholder value.

#### Prerequisites
- Python 3.13 is the tested path for this repo right now. Python 3.14 caused dependency-install friction during review.
- Virtual environment (recommended)

#### Installation
```bash
# Clone the repository
git clone https://github.com/Onyelechie/bil_security_ml.git
cd bil_security_ml

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/Mac:
# source .venv/bin/activate
```
# Install dependencies
**Runtime (to run the system):**
```bash
pip install -r requirements.txt
```

**Dev/Test (to run tests + lint/type checks):**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```


#### Running the Central Server (Area C)
```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# Optional host/port (defaults shown):
$env:HOST = "127.0.0.1"
$env:PORT = "8000"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"
# Optional host/port:
# export HOST="127.0.0.1"
# export PORT="8000"

# Run the server (uses HOST/PORT values from env)
# On Windows PowerShell:
$bindHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$bindPort = if ($env:PORT) { $env:PORT } else { "8000" }
python -m uvicorn server.main:app --reload --host $bindHost --port $bindPort
# On Unix/Mac:
# python -m uvicorn server.main:app --reload --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
```

Quick copy-paste command (PowerShell, fixed host/port):
```powershell
$env:PYTHONPATH="$PWD\src"; python -m uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

Admin note:
- Set `ADMIN_PASSWORD` in `.env` before using `/api/auth/token`, `/api/logs`, `/api/devices/enroll`, or the protected dashboard login at `/dashboard`.

#### Route Table (Server)

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/heartbeat` | Upsert edge heartbeat/status |
| GET | `/api/heartbeat` | List known edge PCs with latest status/heartbeat |
| POST | `/api/alerts` | Ingest alert over HTTP |
| GET | `/api/alerts` | List alerts |
| GET | `/api/alerts/{alert_id}/image` | Serve stored alert image bytes |
| GET | `/api/logs` | Read recent bounded server logs |
| WS | `/ws/alerts` | Ingest alerts over WebSocket (`connected` / `meta_received` / `ack` / `error`) |
| GET | `/dashboard` | Protected web monitoring UI (login required) |

Apply migrations before first run:
```bash
python -m alembic -c alembic.ini upgrade head
```

#### Quick Test (HTTP + WebSocket)

HTTP health check:
```powershell
curl.exe http://127.0.0.1:8000/
```

Open the monitoring UI:
```powershell
start http://127.0.0.1:8000/dashboard
```

The dashboard now requires login with the configured `ADMIN_PASSWORD`.

HTTP alert ingestion from PowerShell (recommended):
```powershell
$payload = @{
  site_id    = "site_001"
  camera_id  = "cam_001"
  edge_pc_id = "edge-live-1"
  timestamp  = "2026-03-01T12:00:00Z"
  detections = @(
    @{ "class" = "person"; confidence = 0.98 }
  )
}

$json = $payload | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/alerts" -Method Post -ContentType "application/json" -Body $json
```

Optional `curl.exe` fallback (send JSON from file):
```powershell
$json | Set-Content -Path .\alert.json -Encoding utf8NoBOM
curl.exe -X POST "http://127.0.0.1:8000/api/alerts" -H "Content-Type: application/json" --data-binary "@alert.json"
```

Real-time WebSocket ingestion test (`/ws/alerts`) with metadata + binary image:
```powershell
@'
import asyncio, json
from datetime import datetime, timezone
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/alerts") as ws:
        print("connected:", await ws.recv())

        meta = {
            "type": "alert_meta",
            "alert": {
                "site_id": "site_ws",
                "camera_id": "cam_ws",
                "edge_pc_id": "edge-ws-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detections": [{"class": "person", "confidence": 0.95}]
            }
        }
        await ws.send(json.dumps(meta))
        print("meta ack:", await ws.recv())

        # Send JPEG/PNG bytes in binary frame
        await ws.send(b"\x89PNG\r\n\x1a\n\x00demo-image-bytes")
        print("ack:", await ws.recv())

asyncio.run(main())
'@ | python -
```

Note: JSON-only alert messages are still accepted for backward compatibility, but binary image transport uses the `alert_meta` + binary frame sequence.

WebSocket protocol summary (`WS /ws/alerts`):
- Server sends `connected` immediately after handshake.
- Client can send either:
  - a complete JSON alert payload (backward-compatible path), or
  - metadata-first binary flow:
    1. JSON frame: `{"type":"alert_meta","alert":{...}}`
    2. binary frame: image bytes (JPEG/PNG/GIF/WEBP or raw bytes)
- Server sends `meta_received` after valid metadata.
- Server sends `ack` when alert persistence succeeds.
- Server sends `error` for invalid frames, validation failures, queue pressure, or storage/persistence failures.

Backpressure behavior:
- If internal ingestion queue is full, server returns `{"type":"error","code":"queue_full",...}`.
- The overloaded message is dropped (not queued); connection stays open.
- Client should retry with backoff/jitter.

Image storage behavior:
- Binary image frames are saved under `WS_IMAGE_STORAGE_DIR`.
- Filename format: `<site_id>_<camera_id>_<received_utc_timestamp>.<ext>`.
- `received_utc_timestamp` is server receive time.
- Stored path is written to `alerts.image_path`.
- Storage directory is created automatically if it does not exist.
- Images older than `WS_IMAGE_RETENTION_HOURS` are deleted by a background cleanup task every `WS_IMAGE_CLEANUP_INTERVAL_HOURS` (defaults are both `24`).

If needed:
```bash
pip install websockets
```

WebSocket load test script:
```powershell
# Example: 200 clients, 10 messages each
python scripts/ws_load_test.py --clients 200 --messages-per-client 10

# Smoother ramp-up (20 ms between client starts)
python scripts/ws_load_test.py --clients 200 --messages-per-client 10 --stagger-ms 20
```

The script exits with code `1` if all expected messages are not ACKed.



### Endpoint Purpose

- **Heartbeat** (`POST /api/heartbeat`): Used by edge PCs to report their own status and last-seen time to the server. This endpoint now requires a signed request from an enrolled device.
- **Edge Status List** (`GET /api/heartbeat`): Returns known edge PCs and their current status/last heartbeat for dashboard-style monitoring.
- **Healthcheck** (`GET /`): Used by anyone (user, monitoring system, load balancer) to check if the server itself is running and responsive. Returns a simple status message.
- **Server Logs** (`GET /api/logs`): Returns recent in-memory server logs for operations visibility in the web dashboard. Requires admin auth.
- **WebSocket Alert Ingestion** (`WS /ws/alerts`): Used by edge clients or UI clients to stream alert payloads in real time and receive immediate ACK or error responses. Supports metadata-first + binary image frames.
- **Dashboard** (`GET /dashboard`): Protected web UI to monitor multiple server targets (host/port), view alerts/images, edge status, and logs.

---

### API Documentation

When the server is running, interactive API documentation is available:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

These docs are auto-generated from the code and always up to date. You can try out endpoints, view request/response schemas, and see example payloads directly in your browser.

---

### API Endpoints

#### Heartbeat Endpoint

**POST /api/heartbeat**

Used by edge PCs to report their status. The server records the time it receives the heartbeat as `last_heartbeat` (using its own UTC clock, not the client timestamp).
The request must include:
- `X-Device-Id`
- `X-Device-Signature`

The enrolled `device_id` must match the claimed `edge_pc_id`.

**Request Body (HeartbeatIn):**
```json
{
  "edge_pc_id": "edge-001",
  "site_name": "Warehouse 1",
  "status": "online",
  "timestamp": "2026-02-17T12:34:56Z"
}
```

**Response (HeartbeatOut):**
```json
{
  "edge_pc_id": "edge-001",
  "site_name": "Warehouse 1",
  "status": "online",
  "last_heartbeat": "2026-02-17T12:34:56Z",
  "message": "Server received heartbeat"
}
```


**Model Conventions:**
- `In` models (e.g., `HeartbeatIn`) are for data sent from the client to the server (requests).
- `Out` models (e.g., `HeartbeatOut`) are for data sent from the server to the client (responses). The heartbeat response now includes a `message` field confirming receipt. The `last_heartbeat` field is always set by the server's current UTC time.

#### Alerts Endpoint
- **POST /api/alerts**: Ingests alerts from edge PCs. Requires a signed request from an enrolled device whose `device_id` matches `edge_pc_id`.
- **GET /api/alerts**: Lists alerts (filtering to be implemented).
- **GET /api/alerts/{alert_id}/image**: Returns the stored image bytes for that alert if `image_path` exists and is within configured storage.
- **WS /ws/alerts**: Accepts alert JSON messages (backward compatible) and metadata + binary image frames. Returns `connected`, `meta_received`, `ack`, or `error` frames.

#### Logs Endpoint
- **GET /api/logs**: Returns bounded, structured server logs.
  - Query params:
    - `limit` (default `200`, max `1000`)
    - `after_id` (optional incremental polling cursor)
    - `level` (optional filter, e.g. `INFO`, `WARNING`, `ERROR`)

#### Monitoring Dashboard
- **GET /dashboard**: Built-in web UI for multi-port monitoring.
- Add one or many targets by host/port (for example `127.0.0.1:8000`, `127.0.0.1:8001`).
- For each selected target, the dashboard shows:
  - health status
  - recent alerts + image previews (when available)
  - known edge PCs from heartbeat data

## Recent Changes (March-April 2026)

Summary of notable recent updates:

### Server / Dashboard
- Per-site image storage: images are stored in a configurable `IMAGE_STORAGE_DIR` with automatic creation of per-site folders on site registration.
- Multipart HTTP upload endpoint: `POST /api/alerts/upload` accepts multipart metadata + image upload.
- Ingestion normalization: server copies local/absolute image paths referenced by incoming alerts into the configured storage root and persists a storage-relative path in the DB.
- Per-site retention & cleanup: per-site image retention settings are exposed to the dashboard and background cleanup removes old images.
- Dashboard updates: protected login, `Settings` view for retention and edge enrollment, Overview cleanup, and alerts viewer fixes.
- Alert timestamps are normalized and displayed in `America/Winnipeg`.

### Edge Agent
- Unified live runtime under `python -m edge_agent --run`.
- Added motion-source selection:
  - `ENABLE_TCP_MOTION`
  - `ENABLE_LOCAL_MOTION`
- Both TCP motion and local RTSP-derived motion now feed the same incident / extraction / inference / alert pipeline.
- RTSP ingest now keeps a **low-resolution color ring buffer** for improved inference quality.
- Local motion detection still runs on **grayscale**, derived from the color frames.
- Added configurable detector controls:
  - `DETECTOR_PERSON_CONF`
  - `DETECTOR_VEHICLE_CONF`
  - `DETECTOR_ALLOWED_CLASSES`
- Evaluator now prefers valid **person** detections over valid **vehicle** detections when both are available.
- Window frame selection has been improved to keep bounded compute while biasing more useful frames near the recent part of the incident.
- Recommended indoor defaults were tightened to reduce oversized incident windows and improve latency on edge hardware.

## Edge Agent (Area B)

The edge agent is the on-site Windows service that can:

- listen for motion events over TCP (from BIL software / VMS rules)
- pull frames via RTSP into an in-memory ring buffer
- optionally compute lightweight local motion from RTSP frames
- merge/dedupe noisy triggers into incidents
- extract bounded frame windows around incidents
- run YOLO burst inference on selected frames
- send signed alerts + heartbeats to the central server

### Current runtime model

The live edge pipeline is event-driven:

`RTSP stream -> motion trigger(s) -> incident manager -> extraction worker -> ML evaluator -> pipeline runner -> server`

Supported motion sources:
- **TCP motion**: external XML motion events from BIL/VMS
- **Local motion**: cheap frame-difference motion computed from RTSP frames on the edge

Both motion sources can be enabled together and feed the same downstream pipeline.

### Frame strategy

- The RTSP reader stores **low-resolution color** frames in memory.
- Local motion detection converts buffered frames to **grayscale** only for motion scoring.
- Extracted frames sent to YOLO remain **color**, which improves person detection quality versus grayscale-only inference.

### Running the Edge Agent

> Important: this repo uses a `src/` layout. Set `PYTHONPATH` before running.

```powershell
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"
````

#### Print config

```powershell
python -m edge_agent --print-config
```

#### Run Edge HTTP API (install/debug)

```powershell
python -m edge_agent --http-serve
```

#### Endpoints

* **GET** `http://localhost:8128/health`
  Returns a simple "ok" response if the edge agent is running.
* **GET** `http://localhost:8128/heartbeat`
  Returns edge identity (`edge_pc_id`, `site_name`), a basic status snapshot, and uptime.

> Note: This is the Edge-side heartbeat (server/office -> edge).
> The Central Server heartbeat endpoint is separate (`POST /api/heartbeat`, edge -> server).

#### Run TCP motion listener (prints parsed motion events)

```powershell
python -m edge_agent --tcp-listen
```

#### Run RTSP ingest test

```powershell
python -m edge_agent --rtsp-test
```

This validates:

* RTSP connectivity
* ffmpeg recovery/backoff
* ring buffer fill behavior

#### Run RTSP + local motion debug mode

```powershell
python -m edge_agent --motion-test
```

This validates:

* RTSP ingest
* local motion scoring
* incident creation
* extraction worker behavior

#### Run the unified live pipeline

```powershell
python -m edge_agent --run
```

This is the main runtime entrypoint. Depending on `.env`, it can use:

* TCP motion only
* local motion only
* both TCP and local motion

The unified `--run` path handles:

* trigger ingestion
* incident merging
* extraction
* YOLO evaluation
* alert sending to the server

### Example: local-motion-only live run

```powershell
$env:PYTHONPATH="$PWD\src"
$env:RTSP_URL_LOW="rtsp://<camera>/Streaming/Channels/102"
$env:ENABLE_TCP_MOTION="false"
$env:ENABLE_LOCAL_MOTION="true"
$env:SHARED_STORAGE_ROOT=(Resolve-Path ".\storage\ws_alert_images").Path
python -m edge_agent --run
```

### Example: TCP-motion-only live run

```powershell
$env:PYTHONPATH="$PWD\src"
$env:ENABLE_TCP_MOTION="true"
$env:ENABLE_LOCAL_MOTION="false"
python -m edge_agent --run
```

### Example: both motion sources enabled

```powershell
$env:PYTHONPATH="$PWD\src"
$env:ENABLE_TCP_MOTION="true"
$env:ENABLE_LOCAL_MOTION="true"
python -m edge_agent --run
```

### Detection notes

The evaluator supports configurable alert classes and thresholds:

* `DETECTOR_ALLOWED_CLASSES`
* `DETECTOR_PERSON_CONF`
* `DETECTOR_VEHICLE_CONF`

Recommended indoor single-camera setup:

* `DETECTOR_ALLOWED_CLASSES=person`
* `DETECTOR_PERSON_CONF=0.40`
* `DETECTOR_VEHICLE_CONF=0.90`

This helps reduce false indoor “vehicle” detections in cluttered scenes.

### Frame selection notes

The extraction worker does **not** send an entire raw video file to YOLO.
Instead it:

1. pulls all buffered frames inside the incident window
2. selects a bounded set of representative frames
3. sends only those selected frames into the ML evaluator

Selection is deterministic and now biases more frames toward the more recent portion of the incident while still keeping early/full-window context.

### Performance notes

For an i7-class edge PC, a good starting profile for one indoor camera is:

* `FRAME_WIDTH=640`
* `FRAME_HEIGHT=360`
* `ANALYSIS_FPS=5`
* `RING_BUFFER_SECONDS=25`
* `WINDOW_MAX_FRAMES=40`

This keeps the pipeline responsive while preserving enough information for event-driven inference.

### Alert images

* Confirmed alerts can save an annotated snapshot to disk.
* If `SHARED_STORAGE_ROOT` is configured and visible to both edge and server, the alert payload can include a usable `image_path`.
* The server/dashboard can then display the alert image.

## Running Tests

```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

# Run all tests
python -m pytest

# Edge agent tests only
python -m pytest -q tests/edge_agent

# Focused edge runtime tests
python -m pytest tests/edge_agent/test_main.py -v
python -m pytest tests/edge_agent/test_pipeline_runner.py -v
python -m pytest tests/edge_agent/test_ml_evaluator.py -v
python -m pytest tests/edge_agent/video/test_window_extractor.py -v
````

### Testing Commands

Linux/macOS (with `make` installed):

```bash
make test
```

Windows PowerShell:

```powershell
python scripts/run_tests.py
```

What both commands do:

* Run `pytest -v` using repository test configuration.
* Ensure both `src/` and project root are injected into `PYTHONPATH`.

Prerequisites:

* Install project dependencies before running tests: `pip install -r requirements.txt`.
* Some ML packages (`torch`, `torchvision`, `ultralytics`, OpenVINO-related tooling) are heavier and platform-sensitive. Use the project virtual environment when running edge tests.

Troubleshooting:

* If pytest fails with an error about `--cov` options, install `pytest-cov`.
* If imports fail because your global Python is being used instead of the virtual environment, activate `.venv` first or invoke `.\.venv\Scripts\python.exe` directly.
* If you see import errors for top-level packages, run tests through the helper:

  ```bash
  python scripts/run_tests.py
  ```
