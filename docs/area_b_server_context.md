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
- Detection pipeline (frame window extraction)
- Update checker (poll server for new models/config)
- Offline mode (queue alerts when disconnected)
- Heartbeat/health reporting to server
- Local configuration management

---

## 4. Demo Environment (BIL-provided PC)
This is the initial on-site setup information:

- **Motion events:** TCP packets sent to `172.22.0.5:8127`
- **Camera IP:** `172.22.0.10`
- **Direct RTSP**
  - High: `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/101/`
  - Low:  `rtsp://admin:LiveCamera1@172.22.0.10:554/Streaming/Channels/102/`
- **VMS (Symphony) RTSP-like URLs**
  - High: `rtsp://172.22.0.5:50010/live?camera=1&user=admin&pass=LlowXGMdQ0cERgI=%3D`
  - Low: append `&stream=2`

Note: VMS URLs may not be “real” RTSP for some players; we will keep stream reader code modular.

---

## 5. Interfaces (Planned)

### 5.1 Input: TCP motion event (XML)
Edge receives motion triggers over TCP (XML payload), extracts at least:
- camera id / name
- policy id / name
- user string / event type
- timestamp received

### 5.2 Input: RTSP video
Edge maintains ring buffers for configured cameras (default: low stream for inference).

### 5.3 Output: Alerts to Area C
Edge sends alert metadata + optional snapshot image to central server.
(We will likely use multipart for image bytes rather than base64 for efficiency.)

### 5.4 Output: Heartbeat + update polling
Edge periodically:
- POST heartbeat (health + versions)
- GET update checks (model/config)

---

## 6. Project Structure (Area B)
Edge Agent lives in `src/edge_agent/` (src-layout like the server).

- `edge_agent/config.py` - pydantic settings
- `edge_agent/main.py`   - entrypoint (skeleton in PR1)
- later PRs will add:
  - ingest/ (TCP listener + XML parsing)
  - video/ (RTSP readers + ring buffer)
  - pipeline/ (scheduler/backpressure)
  - inference/ (detector interface)
  - decision/ (rules + persistence + masks)
  - comms/ (server client + offline spool)

---

## 7. How to run (PR1)
```bash
# Ensure PYTHONPATH includes src/
# PowerShell:
$env:PYTHONPATH = "$PWD\src"

python -m edge_agent.main --print-config
````

---

## 8. PR Roadmap (next)

* PR2: TCP listener + XML parsing + tests
* PR3: RTSP reader interface + ring buffer skeleton
* PR4+: scheduler/backpressure + decision rules + alert sending + offline mode