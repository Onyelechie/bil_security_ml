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
- Trigger management (cooldown/dedupe / merge)
- Detection pipeline (window extraction + inference + rules) *(later)*
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
- `edge_agent/triggers/` - TCP trigger, local motion trigger, TriggerManager, types
- `edge_agent/video/` - RTSP reader (ffmpeg) + ring buffer
- `edge_agent/edge_api.py` - edge debug HTTP API

---

## 7. How to run (current)

> Ensure `PYTHONPATH` includes `src/`.

```powershell
$env:PYTHONPATH = "$PWD\src"

# Print resolved config
python -m edge_agent --print-config

# Edge HTTP API (install/debug)
python -m edge_agent --http-serve

# TCP motion listener (prints parsed motion events)
python -m edge_agent --tcp-listen

# RTSP + local motion live test (requires RTSP_URL_LOW)
python -m edge_agent --motion-test
```

Notes:

* `--motion-test` is a live dev/test mode for RTSP ingest + ring buffer + local motion. It does not send alerts to the central server yet.
* RTSP ingest uses `imageio-ffmpeg` (no separate system ffmpeg install required).


---

## 8. PR Roadmap (next)

✅ PR1: config/logging/CLI skeleton + module entrypoint (`python -m edge_agent`)   
✅ PR2: Edge HTTP API (/health, /heartbeat) + tests   
✅ PR3: TCP listener + XML parsing + MotionEvent types + TriggerManager (cooldown/dedupe)   
✅ PR4: RTSP reader + ring buffer + tests   
✅ PR5: Local motion trigger + `--motion-test` + improved RTSP recovery logs/backoff

Next:

* PR6: Window extraction from ring buffer on accepted motion (T-2s..T+6s) + frame selection
* PR7: Integrate Area A detector wrapper (YOLO burst inference) + snapshot artifact generation
* PR8: Alert sending to server + offline queue/spool
* PR9: Update polling + model/config versioning