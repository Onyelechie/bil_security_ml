# BIL Security ML - Edge Agent (Area B)
## Project Context Document for AI Agent

**Owner:** Bhavik Jain  
**Area:** B - Edge Agent (on-site Windows PC service)  
**Date:** April 2026  
**Course:** COMP 4560 Industrial Project (Winter 2026)

---

## 1. Purpose
Area B is the **Edge Agent** that runs at customer sites on an on-site Windows PC.
It connects local cameras (RTSP), consumes motion events from existing BIL software (TCP),
can also compute motion locally from RTSP frames, runs event-driven detection logic,
and sends alert artifacts to the central server (Area C).

This remains intentionally **event-driven**:
- RTSP is the video input
- one or more motion sources decide when to analyze a bounded incident window
- YOLO is used as a confirmation/filtering step before sending an alert

---

## 2. System Context (Big Picture)

```

BIL Software (Motion)         RTSP Camera Streams
│                               │
▼                               ▼
TCP Motion Events              RTSP Reader(s)
│                               │
└───────────────┐       ┌───────┘
                ▼       ▼
           EDGE AGENT (Area B)
(ring buffer + window extraction + inference + rules)
                   │
                   ▼
        Central Server (Area C)
 (alerts storage + dashboard + updates)

````

---

## 3. Responsibilities (Area B)

Area B owns:

* Build edge service (auto-start, auto-restart)
* Ring buffer for frame storage (memory-conscious)
* TCP listener for BIL motion events
* Optional local motion detection from RTSP frames
* Trigger management (cooldown/dedupe / merge window)
* **Incident merging** (many triggers -> one “incident”, no spam)
* **Window extraction** from ring buffer around incident
* Detection pipeline (YOLO burst inference on extracted frames)
* Signed alert sending to Area C
* Heartbeat/health reporting
* Local debug HTTP API

---

## 4. Demo Environment

### Original BIL demo environment

* **Motion events:** TCP packets sent to `172.22.0.5:8127`
* **Camera IP:** `172.22.0.10`

**Direct RTSP**

* High: `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/101/`
* Low:  `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/102/`

**VMS (Symphony) RTSP-like URLs**

* High: `rtsp://172.22.0.5:50010/live?camera=1&user=admin&pass=LlowXGMdQ0cERgI=%3D`
* Low: append `&stream=2`

### Local dev/test camera environment

A local RTSP camera/substream can also be used for development. In recent testing,
the edge pipeline was exercised against a local RTSP source on the same network
using a low-resolution stream and local-motion-driven incidents.

Note: VMS URLs may not be “real” RTSP for some players; keep stream reader code modular and support direct RTSP for stable demos.

---

## 5. Interfaces

### 5.1 Input: TCP motion event (XML)

Edge receives motion triggers over TCP (XML payload), extracts at least:

* camera id / name
* policy id / name
* user string / event type
* timestamp received

### 5.2 Input: RTSP video

Edge maintains a low-resolution **color** ring buffer for configured cameras.
Local motion scoring converts frames to grayscale only when needed for cheap motion detection.

### 5.3 Output: Alerts to Area C

Edge sends:

* alert metadata
* optional snapshot image path (shared storage mode)

Signing requirement:

* Alert and heartbeat requests must include `X-Device-Id` and `X-Device-Signature`
* `DEVICE_ID` must match `EDGE_PC_ID`
* the matching public key must be enrolled on the server

Shared storage option:

* If a shared mount exists between edge and server, edge can save images to that
  mount and include `image_path` in alert payloads
* Configure `SHARED_STORAGE_ROOT` so queued alerts keep `image_path` only when the
  file exists under the shared root
* Invalid JSON or 4xx-rejected queued alerts are quarantined under
  `OFFLINE_QUEUE_DIR/bad/`
* Quarantined payloads are deleted after `QUEUE_QUARANTINE_RETENTION_DAYS`

### 5.4 Output: Heartbeat

Edge periodically sends signed heartbeats to the server.

### 5.5 Input: Edge health/heartbeat (office/server -> edge)

Edge exposes:

* `GET /health`
* `GET /heartbeat`

This is separate from the central server heartbeat endpoint (`POST /api/heartbeat`, edge -> server).

---

## 6. Project Structure (Area B)

Edge Agent lives in `src/edge_agent/`.

* `edge_agent/config.py` - pydantic settings
* `edge_agent/main.py` - CLI entrypoints + unified run loop
* `edge_agent/ml_evaluator.py` - YOLO burst inference + alert filtering
* `edge_agent/pipeline_runner.py` - connects extracted frames to evaluator + sender
* `edge_agent/sender.py` - signed heartbeat/alert sending + offline queue behavior

**Triggers**

* `edge_agent/triggers/tcp_trigger.py` - async TCP motion listener
* `edge_agent/triggers/tcp_parse.py` - XML parsing
* `edge_agent/triggers/types.py` - `MotionEvent`
* `edge_agent/triggers/trigger_manager.py` - merge window + cooldown
* `edge_agent/triggers/incident_manager.py` - incident creation/finalization
* `edge_agent/triggers/local_motion_trigger.py` - local motion from RTSP frames

**Video**

* `edge_agent/video/rtsp_reader.py` - ffmpeg-based RTSP reader with retry/backoff
* `edge_agent/video/ring_buffer.py` - in-memory buffer of recent **color** frames
* `edge_agent/video/window_extractor.py` - bounded frame selection from incident windows
* `edge_agent/video/extraction_worker.py` - background worker that waits for post-roll and extracts windows

**Edge debug API**

* `edge_agent/edge_api.py` - edge debug HTTP API

---

## 7. Current runtime behavior

### 7.1 Motion sources

The unified live runtime can use:

* **TCP motion**
* **Local motion**
* **Both at once**

Config flags:

* `ENABLE_TCP_MOTION`
* `ENABLE_LOCAL_MOTION`

Both motion sources feed the same downstream:

* TriggerManager
* IncidentManager
* ExtractionWorker
* MLEvaluator
* PipelineRunner
* ServerSender

### 7.2 Incident merging

Motion can be noisy. We treat motion as an **incident** per camera:

* Incident starts on the first *accepted* motion event
* While incident is active, **any motion** (even dropped by cooldown) extends it
* Incident finalizes when:

  * it is quiet for `INCIDENT_QUIET_SEC`, or
  * it reaches `INCIDENT_MAX_SEC`

### 7.3 Window extraction

When an incident finalizes, the edge agent creates an extraction job and extracts frames:

* Window start: `first_motion - WINDOW_PRE_SEC`
* Window end: `last_motion + WINDOW_POST_SEC`

A background worker waits (bounded) for post-roll frames and then extracts frames from the ring buffer.

Results are marked:

* `ready`
* `partial`
* `dropped`

### 7.4 Frame selection

The edge agent does **not** send an entire raw video file to YOLO.
Instead it:

1. pulls all buffered frames within the incident window
2. selects a bounded set of representative frames
3. sends only those selected frames to the ML evaluator

Selection remains deterministic and capped, but now biases more useful frames toward the later/recent part of the incident while still keeping some early/full-span context.

### 7.5 Detection rules

YOLO runs on extracted **color** frames.

Configurable rules:

* `DETECTOR_PERSON_CONF`
* `DETECTOR_VEHICLE_CONF`
* `DETECTOR_ALLOWED_CLASSES`

For cluttered indoor cameras, the recommended starting profile is:

* `DETECTOR_ALLOWED_CLASSES=person`
* `DETECTOR_PERSON_CONF=0.40`
* `DETECTOR_VEHICLE_CONF=0.90`

The evaluator prefers a valid **person** detection over a valid **vehicle** detection when both are present.

---

## 8. How to run

> Ensure `PYTHONPATH` includes `src/`.

```powershell
$env:PYTHONPATH = "$PWD\src"

# Print resolved config
python -m edge_agent --print-config

# Edge HTTP API
python -m edge_agent --http-serve

# TCP listener only
python -m edge_agent --tcp-listen

# RTSP connectivity/ring-buffer test
python -m edge_agent --rtsp-test

# Local motion debug mode
python -m edge_agent --motion-test

# Unified live runtime (main mode)
python -m edge_agent --run
```

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

---

## 9. Recommended tuning profile (single indoor camera)

Recommended starting point for an i7-class Windows edge PC:

* `FRAME_WIDTH=640`
* `FRAME_HEIGHT=360`
* `ANALYSIS_FPS=5`
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

* keeps motion detection cheap
* preserves color for better inference
* bounds memory/CPU
* reduces false indoor “vehicle” alerts

---

## 10. Roadmap snapshot

✅ PR1: config/logging/CLI skeleton + module entrypoint  
✅ PR2: Edge HTTP API (`/health`, `/heartbeat`) + tests  
✅ PR3: TCP listener + XML parsing + MotionEvent types + TriggerManager  
✅ PR4: RTSP reader + ring buffer + tests  
✅ PR5: Local motion trigger + `--motion-test` + improved RTSP recovery  
✅ PR6: Incident merging + window extraction from ring buffer  
✅ PR7/PR8-style integration work: unified `--run`, burst YOLO inference, alert sending, shared-storage snapshot path support, local-motion-to-alert integration  