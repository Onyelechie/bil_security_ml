# BIL Security ML - Edge Agent (Area B)

## Project Context Document for AI Agent

**Owner:** Bhavik Jain  
**Area:** B - Edge Agent (on-site Windows PC service)  
**Date:** April 2026  
**Course:** COMP 4560 Industrial Project (Winter 2026)

---

## 1. Purpose

Area B is the **Edge Agent** that runs at customer sites on an on-site Windows PC.

It connects local cameras through RTSP, consumes motion events from existing BIL software over TCP, can also compute motion locally from RTSP frames, applies event-driven filtering and incident logic, runs bounded YOLO confirmation on extracted windows, and sends alert artifacts to the central server.

Area B now also includes a **local edge console** for operators and developers on the edge PC itself. That console is used to:

- view the latest preview frame
- draw motion include and exclude zones
- review runtime state
- tune edge settings
- save settings locally
- request a pipeline restart when needed

The system remains intentionally **event-driven**:

- RTSP is the video input
- one or more motion sources decide when to analyze a bounded incident window
- YOLO is used as a confirmation and filtering step before sending an alert

---

## 2. System Context (Big Picture)

```text
BIL Software (Motion)         RTSP Camera Streams
│                               │
▼                               ▼
TCP Motion Events              RTSP Reader(s)
│                               │
└───────────────┐       ┌───────┘
                ▼       ▼
          EDGE AGENT (Area B)
(ring buffer + incident logic + extraction + inference + rules + local console)
                     │
                     ▼
           Central Server (Area C)
      (alerts storage + dashboard + updates)
````

---

## 3. Responsibilities (Area B)

Area B owns:

- Build edge service runtime for the on-site Windows PC
- Ring buffer for recent frame storage
- TCP listener for BIL motion events
- Optional local motion detection from RTSP frames
- Trigger management using cooldown and merge-window rules
- Incident merging so many noisy triggers become one bounded event
- Window extraction from the ring buffer around an incident
- Detection pipeline using YOLO on selected extracted frames
- Signed heartbeat and alert sending to Area C
- Local runtime state tracking for preview and operator actions
- Local edge console UI and API for preview, zoning, tuning, and restart

---

## 4. Demo Environment

### Original BIL demo environment

- **Motion events:** TCP packets sent to `172.22.0.5:8127`
- **Camera IP:** `172.22.0.10`

### Direct RTSP

- High: `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/101/`
- Low:  `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/102/`

### VMS (Symphony) RTSP-like URLs

- High: `rtsp://172.22.0.5:50010/live?camera=1&user=admin&pass=LlowXGMdQ0cERgI=%3D`
- Low: append `&stream=2`

### Local dev and test camera environment

A local RTSP source can also be used for development and demos.

Recent development work also used a local webcam-to-RTSP test setup to exercise:

- RTSP ingest
- preview rendering
- local motion detection
- zoning and masking
- end-to-end alert behavior

Note: some VMS URLs may not behave like standard RTSP in all players, so the edge reader remains modular and direct RTSP is still the most reliable test path.

---

## 5. Interfaces

### 5.1 Input: TCP motion event

Edge receives motion triggers over TCP and parses XML payloads to extract at least:

- camera id or name
- policy id or policy name
- user string or event type
- received timestamp

### 5.2 Input: RTSP video

Edge maintains a low-resolution **color** ring buffer for configured cameras.

The same RTSP reader also updates the **latest preview frame** used by the local edge console.

Important behavior:

- preview and analysis do **not** use separate camera streams
- the latest decoded frame is used for preview
- analysis frames are pushed into the ring buffer at the configured analysis cadence
- local motion scoring converts frames to grayscale only when needed

### 5.3 Output: Alerts to Area C

Edge sends:

- alert metadata
- detections
- optional image path when shared storage is configured

Signing requirement:

- alert and heartbeat requests must include `X-Device-Id` and `X-Device-Signature`
- `DEVICE_ID` must match `EDGE_PC_ID`
- the matching public key must be enrolled on the server

Shared storage behavior:

- if a shared mount exists between edge and server, edge can save images there and include `image_path`
- configure `SHARED_STORAGE_ROOT` so queued alerts keep `image_path` only when the file exists under the shared root
- invalid JSON or 4xx-rejected queued alerts are quarantined under `OFFLINE_QUEUE_DIR/bad/`
- quarantined payloads are deleted after `QUEUE_QUARANTINE_RETENTION_DAYS`

### 5.4 Output: Heartbeat

Edge periodically sends signed heartbeats to the central server.

### 5.5 Local edge console API

The edge PC exposes a local operator-facing API and UI.

Core endpoints:

- `GET /health`
- `GET /heartbeat`
- `GET /api/runtime`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/preview`
- `PUT /api/zones`
- `POST /api/runtime/restart`

This is separate from the central server heartbeat endpoint `POST /api/heartbeat`.

---

## 6. Project Structure (Area B)

Edge Agent lives in `src/edge_agent/`.

### Core runtime

- `edge_agent/config.py` - pydantic settings
- `edge_agent/main.py` - CLI entrypoints and unified runtime loop
- `edge_agent/ml_evaluator.py` - YOLO burst inference and alert filtering
- `edge_agent/pipeline_runner.py` - connects extracted frames to evaluator and sender
- `edge_agent/sender.py` - signed heartbeat and alert sending with offline queue behavior
- `edge_agent/runtime_state.py` - shared runtime snapshot for preview, UI, and restart/apply hooks
- `edge_agent/settings_store.py` - local save and reload of editable settings from `.env` and local JSON state

### Triggers

- `edge_agent/triggers/tcp_trigger.py` - async TCP motion listener
- `edge_agent/triggers/tcp_parse.py` - XML parsing
- `edge_agent/triggers/types.py` - `MotionEvent`
- `edge_agent/triggers/trigger_manager.py` - merge window and cooldown
- `edge_agent/triggers/incident_manager.py` - incident creation and finalization
- `edge_agent/triggers/local_motion_trigger.py` - local motion from RTSP frames, masks, and PTZ-style suppression

### Video

- `edge_agent/video/rtsp_reader.py` - ffmpeg-based RTSP reader with retry and backoff
- `edge_agent/video/ring_buffer.py` - in-memory buffer of recent **color** frames
- `edge_agent/video/window_extractor.py` - bounded frame selection from incident windows
- `edge_agent/video/extraction_worker.py` - background worker that waits for post-roll and extracts windows

### Edge console

- `edge_agent/edge_api.py` - local edge console API and UI entry
- `edge_agent/ui/` - local edge console static UI assets

---

## 7. Current runtime behavior

### 7.1 Motion sources

The unified live runtime can use:

- **TCP motion**
- **Local motion**
- **Both at once**

Config flags:

- `ENABLE_TCP_MOTION`
- `ENABLE_LOCAL_MOTION`

Both motion sources feed the same downstream:

- TriggerManager
- IncidentManager
- ExtractionWorker
- MLEvaluator
- PipelineRunner
- ServerSender

### 7.2 Incident merging

Motion can be noisy, so the system treats motion as an **incident** per camera.

Behavior:

- incident starts on the first accepted motion event
- while incident is active, any motion can extend it, even if that individual trigger was dropped by cooldown
- incident finalizes when it has been quiet for `INCIDENT_QUIET_SEC`
- incident also finalizes when it reaches `INCIDENT_MAX_SEC`

This helps produce one bounded analysis window for one real event instead of many small alert attempts.

### 7.3 Window extraction

When an incident finalizes, the edge agent creates an extraction job and extracts frames from the ring buffer.

Window boundaries:

- window start = `first_motion - WINDOW_PRE_SEC`
- window end = `last_motion + WINDOW_POST_SEC`

A background worker waits for post-roll frames, then extracts the bounded frame set from the ring buffer.

Results are marked as:

- `ready`
- `partial`
- `dropped`

### 7.4 Frame selection

The edge agent does **not** send a whole raw video file to YOLO.

Instead it:

1. gathers buffered frames inside the incident window
2. selects a bounded representative subset
3. sends only those selected frames to the ML evaluator

Selection is deterministic and capped, while still preserving:

- some early context
- full-span coverage
- slightly more emphasis toward the more recent part of the event

### 7.5 Detection rules

YOLO runs on extracted **color** frames.

Configurable rules:

- `DETECTOR_PERSON_CONF`
- `DETECTOR_VEHICLE_CONF`
- `DETECTOR_ALLOWED_CLASSES`

For cluttered indoor cameras, the recommended starting profile is:

- `DETECTOR_ALLOWED_CLASSES=person`
- `DETECTOR_PERSON_CONF=0.40`
- `DETECTOR_VEHICLE_CONF=0.90`

The evaluator prefers a valid **person** detection over a valid **vehicle** detection when both are present.

### 7.6 Hard-scene filtering

Hard-scene filtering now includes PTZ-style or large-scene-motion suppression inside local motion detection.

This is meant to reduce false motion caused by:

- camera movement
- sweeping or touring views
- very large full-frame changes
- scenes where the camera itself appears to move rather than a real subject moving within the scene

Key settings:

- `PTZ_GLOBAL_MOTION_THRESHOLD`
- `PTZ_CONSECUTIVE_FRAMES`
- `PTZ_SUPPRESS_SEC`

Behavior:

- the trigger first measures a full-frame global change score
- if that score is too large for enough consecutive frames, the system treats it as likely camera movement
- local motion alerts are then temporarily suppressed for the configured period
- normal local motion resumes after the suppression window expires

This is the current hard-scene filtering layer for Area B.

### 7.7 Configurable zones and masking

Area B now supports configurable motion zones.

Two polygon types are supported:

- **include polygons** for regions that count toward motion
- **exclude polygons** for regions that should be ignored

Polygons use normalized coordinates in the range `0..1`.

The motion trigger builds a score mask from these polygons and only scores motion in active pixels. This allows the edge PC to ignore streets, irrelevant background regions, and other noisy areas.

### 7.8 Local edge console behavior

The edge console is served locally from the edge PC and is used for:

- viewing runtime state
- viewing the latest frame
- drawing include and exclude polygons
- editing grouped settings
- saving settings locally
- requesting runtime restart

Saved settings are handled in two layers:

- `.env` for normal editable env-backed values
- `.edge_console_state.json` for JSON-only local state such as polygon lists

At runtime:

- some local-motion-related settings can be applied immediately through `runtime_state.apply_settings(...)`
- other settings are saved locally and marked as restart-required
- restart requests are exposed through `runtime_state.request_restart(...)`

---

## 8. How to run

> Ensure `PYTHONPATH` includes `src/`.

```powershell
$env:PYTHONPATH = "$PWD\src"

# Print resolved config
python -m edge_agent --print-config

# Edge HTTP API and local console
python -m edge_agent --http-serve

# TCP listener only
python -m edge_agent --tcp-listen

# RTSP connectivity and ring-buffer test
python -m edge_agent --rtsp-test

# Local motion debug mode
python -m edge_agent --motion-test

# Sample video directly through pipeline
python -m edge_agent --sample-video .\path\to\video.mp4

# Unified live runtime with console
python -m edge_agent --run
````

### Example: local-motion-driven indoor run

```powershell
$env:PYTHONPATH="$PWD\src"
$env:RTSP_URL_LOW="rtsp://<camera>/Streaming/Channels/102"
$env:ENABLE_TCP_MOTION="false"
$env:ENABLE_LOCAL_MOTION="true"
$env:DETECTOR_ALLOWED_CLASSES="person"
$env:SHARED_STORAGE_ROOT=(Resolve-Path ".\storage\ws_alert_images").Path
python -m edge_agent --run
```

### Local console access

When the edge console is running, open:

```text
http://127.0.0.1:8128
```

Use it to:

* review pipeline status
* view preview frames
* draw zones
* save local settings
* restart the edge pipeline when required

---

## 9. Recommended tuning profile (single indoor camera)

Recommended starting point for a single indoor camera on an i7-class Windows edge PC:

* `FRAME_WIDTH=640`
* `FRAME_HEIGHT=360`
* `ANALYSIS_FPS=5`
* `PREVIEW_FPS=8`
* `RING_BUFFER_SECONDS=25`
* `INCIDENT_MAX_SEC=12.0`
* `WINDOW_PRE_SEC=1.5`
* `WINDOW_POST_SEC=4.0`
* `WINDOW_TARGET_FPS=5.0`
* `WINDOW_MAX_FRAMES=40`
* `DETECTOR_ALLOWED_CLASSES=person`
* `DETECTOR_PERSON_CONF=0.40`
* `DETECTOR_VEHICLE_CONF=0.90`

Why:

* keeps local motion scoring cheap
* preserves color for better inference quality
* bounds memory and CPU use
* reduces false indoor vehicle alerts
* keeps preview responsive without needing a separate stream

---

## 10. Roadmap snapshot

✅ PR1: config, logging, CLI skeleton, and module entrypoint  
✅ PR2: Edge HTTP API with `/health` and `/heartbeat`  
✅ PR3: TCP listener, XML parsing, MotionEvent types, and TriggerManager  
✅ PR4: RTSP reader, ring buffer, and recovery behavior  
✅ PR5: local motion trigger and `--motion-test`  
✅ PR6: incident merging and window extraction from the ring buffer  
✅ PR7: unified `--run`, burst YOLO evaluation, alert sending, and shared-storage snapshot behavior  
✅ PR8: local edge console UI, preview rendering, grouped settings, local save flow, restart flow, configurable zoning and masking, and PTZ-style large-scene-motion suppression
