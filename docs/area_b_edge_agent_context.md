# BIL Security ML - Edge Agent (Area B)
## Project Context Document for AI Agent

**Owner:** Bhavik Jain  
**Area:** B - Edge Agent (on-site Windows PC service)  
**Date:** February 2026  
**Course:** COMP 4560 Industrial Project (Winter 2026)

---

## 1. Purpose
Area B is the **Edge Agent** that runs at customer sites on an on-site Windows PC.
It connects local cameras (RTSP), consumes motion events from existing BIL software (TCP),
runs detection/decision logic, and sends alert artifacts to the central server (Area C).

This is intentionally **event-driven**: RTSP is the video input; the motion event is the trigger.

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

- Build edge service (auto-start, auto-restart)
- Ring buffer for frame storage (memory-conscious)
- TCP listener for BIL motion events
- Trigger management (cooldown/dedupe / merge window)
- **Incident merging** (many triggers → one “incident”, no spam)
- **Window extraction** from ring buffer around incident (T-2s..T+6s by default)
- Detection pipeline (inference + rules) *(later)*
- Update checker (poll server for new models/config) *(later)*
- Offline mode (queue alerts when disconnected) *(later)*
- Heartbeat/health reporting (edge → server) *(later)*
- Local debug HTTP API (office/server → edge)

---

## 4. Demo Environment (BIL-provided PC)

- **Motion events:** TCP packets sent to `172.22.0.5:8127`
- **Camera IP:** `172.22.0.10`

**Direct RTSP**
- High: `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/101/`
- Low:  `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/102/`

**VMS (Symphony) RTSP-like URLs**
- High: `rtsp://172.22.0.5:50010/live?camera=1&user=admin&pass=LlowXGMdQ0cERgI=%3D`
- Low: append `&stream=2`

Note: VMS URLs may not be “real” RTSP for some players; keep stream reader code modular and support direct RTSP for stable demos.

---

## 5. Interfaces

### 5.1 Input: TCP motion event (XML)
Edge receives motion triggers over TCP (XML payload), extracts at least:
- camera id / name
- policy id / name
- user string / event type
- timestamp received

### 5.2 Input: RTSP video
Edge maintains ring buffers for configured cameras (default: low stream for analysis/inference).

### 5.3 Output: Alerts to Area C *(later)*
Edge sends alert metadata + optional snapshot image to central server.
(Prefer multipart for image bytes rather than base64 for efficiency.)

### 5.4 Output: Heartbeat + update polling *(later)*
Edge periodically:
- POST heartbeat (health + versions)
- GET update checks (model/config)

### 5.5 Input: Edge health/heartbeat (office/server → edge)
Edge exposes a small HTTP API for install/debug:
- `GET /health`: confirms edge agent process is running
- `GET /heartbeat`: returns edge identity + basic status snapshot + uptime

This is separate from the central server heartbeat endpoint (`POST /api/heartbeat`, edge → server).

---

## 6. Project Structure (Area B)
Edge Agent lives in `src/edge_agent/` (src-layout like the server).

- `edge_agent/config.py` - pydantic settings
- `edge_agent/main.py` - CLI entrypoints

**Triggers**
- `edge_agent/triggers/tcp_trigger.py` - async TCP motion listener
- `edge_agent/triggers/tcp_parse.py` - XML parsing
- `edge_agent/triggers/types.py` - MotionEvent dataclass
- `edge_agent/triggers/trigger_manager.py` - merge window + cooldown (accepted events)
- `edge_agent/triggers/incident_manager.py` - PR6: merges motion into incidents + produces extraction jobs
- `edge_agent/triggers/local_motion_trigger.py` - local motion trigger (dev mode)

**Video**
- `edge_agent/video/rtsp_reader.py` - ffmpeg-based RTSP reader (with recovery/backoff)
- `edge_agent/video/ring_buffer.py` - in-memory buffer of recent frames (thread-safe)
- `edge_agent/video/window_extractor.py` - PR6: deterministic frame selection from a time window
- `edge_agent/video/extraction_worker.py` - PR6: background worker that waits for post-roll and extracts windows

**Edge debug API**
- `edge_agent/edge_api.py` - edge debug HTTP API

---

## 7. PR6 behavior (Incident merging + Window extraction)

### 7.1 Why incident merging exists
Motion triggers can be extremely noisy (wind/rain/trees/headlights). We avoid spamming by treating
motion as an **incident** per camera:

- Incident starts on the first *accepted* motion event (from TriggerManager)
- While incident is active, **any motion** (even events dropped by cooldown) extends the incident
- Incident finalizes when:
  - it is quiet for `INCIDENT_QUIET_SEC`, OR
  - it reaches `INCIDENT_MAX_SEC` (chunking under endless motion)

### 7.2 Window extraction
When an incident finalizes, the edge agent creates an extraction job and extracts frames:

- Window start: `first_motion - WINDOW_PRE_SEC` (default 2s)
- Window end: `last_motion + WINDOW_POST_SEC` (default 6s)

A background worker waits (bounded) for post-roll frames up to `window_end` and then extracts frames from the ring buffer.
Results are marked:
- `ready` (full post-roll reached)
- `partial` (timed out waiting for post-roll)
- `dropped` (no ring/no frames)

### 7.3 Deterministic frame selection
To keep compute bounded and outputs stable for testing, selection:
- samples evenly across the window at `WINDOW_TARGET_FPS`
- caps output at `WINDOW_MAX_FRAMES`
- chooses the **closest available frame** for each target timestamp (works with variable FPS)

---

## 8. How to run (current)

> Ensure `PYTHONPATH` includes `src/`.

```powershell
$env:PYTHONPATH = "$PWD\src"

# Print resolved config
python -m edge_agent --print-config

# Edge HTTP API (install/debug)
python -m edge_agent --http-serve

# TCP motion listener (prints accepted/dropped motion events only)
python -m edge_agent --tcp-listen

# PR6 pipeline: incident merging + window extraction worker
# (uses TCP motion; RTSP optional but needed for non-dropped windows)
python -m edge_agent --run

# RTSP + local motion live test (requires RTSP_URL_LOW)
# PR6 logic is active here too (incident merge + extraction worker logs)
python -m edge_agent --motion-test
````

Notes:

* `--motion-test` is a live dev/test mode for RTSP ingest + ring buffer + local motion. It does not send alerts to the central server yet.
* RTSP ingest uses `imageio-ffmpeg` (no separate system ffmpeg install required).

### PR6 tuning knobs (env vars)

These are configurable via `.env`:

* `INCIDENT_QUIET_SEC` (default `2.0`)
* `INCIDENT_MAX_SEC` (default `20.0`)
* `WINDOW_PRE_SEC` (default `2.0`)
* `WINDOW_POST_SEC` (default `6.0`)
* `WINDOW_TARGET_FPS` (default `5.0`)
* `WINDOW_MAX_FRAMES` (default `40`)
* `WINDOW_WAIT_GRACE_SEC` (default `1.5`)

---

## 9. PR Roadmap

✅ PR1: config/logging/CLI skeleton + module entrypoint (`python -m edge_agent`)   
✅ PR2: Edge HTTP API (/health, /heartbeat) + tests   
✅ PR3: TCP listener + XML parsing + MotionEvent types + TriggerManager (cooldown/dedupe)   
✅ PR4: RTSP reader + ring buffer + tests   
✅ PR5: Local motion trigger + `--motion-test` + improved RTSP recovery logs/backoff  
✅ PR6: Incident merging + window extraction from ring buffer (T-2s..T+6s) + deterministic frame selection

Next:

* PR7: Integrate Area A detector wrapper (YOLO burst inference) + snapshot artifact generation
* PR8: Alert sending to server + offline queue/spool
* PR9: Update polling + model/config versioning