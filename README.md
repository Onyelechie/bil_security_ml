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

1. **Research and Setup** (Jan 6 – Jan 24, 2026): Initial research on intrusion detection techniques, setup development environment
2. **Research (continued)** (Jan 25 – Feb 7, 2026): Evaluate AI models and finalize architecture
3. **Prototype Development** (Feb 8 – Feb 28, 2026): Implement detection pipeline and performance profiling
4. **Evaluation and Refinement** (Mar 1 – Mar 21, 2026): Test with real data and optimize
5. **Finishing Touches** (Mar 22 – Apr 6, 2026): Finalize documentation and presentation

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

Important variables (see `.env.example`): `DATABASE_URL`, `HOST`, `PORT`, `DEBUG`, `CORS_ORIGINS` (comma-separated), `SECRET_KEY`.


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

#### Prerequisites
- Python 3.9+
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
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

# Run the server
python -m uvicorn server.main:app --reload --port 8000
```



### Endpoint Purpose

- **Heartbeat** (`POST /api/heartbeat`): Used by edge PCs to report their own status and last-seen time to the server. This lets the server track which devices are online and their current state.
- **Healthcheck** (`GET /`): Used by anyone (user, monitoring system, load balancer) to check if the server itself is running and responsive. Returns a simple status message.

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
- **POST /api/alerts**: Ingests alerts from edge PCs (see code for schema).
- **GET /api/alerts**: Lists alerts (filtering to be implemented).

---

### Security & Production Notes
- CORS is now restricted to `http://localhost:3000` and `http://localhost:8000` for development. **Update this for production deployments!**
- No authentication is enabled by default. Add API keys or JWT for production deployments.
- Alert listing filters are marked as TODO and will be implemented in future updates.


## Edge Agent (Area B)

The edge agent is the on-site Windows service that will:
- listen for motion events over TCP
- pull frames via RTSP
- run detection/decision logic
- send alerts + heartbeats to the central server

### Running the Edge Agent (PR1 skeleton)

```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

python -m edge_agent.main --print-config
```

See `docs/area_b_edge_agent_context.md` for architecture + demo environment details.

### Edge Agent HTTP API (PR2)

The edge agent can optionally run a small HTTP API so office staff (or the central server later) can confirm the edge PC is alive.

#### Run the Edge HTTP API
```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

python -m edge_agent.main --http-serve
````

#### Endpoints

* **GET** `http://localhost:8128/health`
  Returns a simple “ok” response if the edge agent is running.
* **GET** `http://localhost:8128/heartbeat`
  Returns edge identity (`edge_pc_id`, `site_name`), a basic status snapshot, and uptime.

> Note: This is the Edge-side heartbeat (server/office → edge).
> The Central Server heartbeat endpoint is separate (`POST /api/heartbeat`, edge → server).


## Running Tests
```bash
# Set Python path for src/ layout
# On Windows PowerShell:
$env:PYTHONPATH = "$PWD\src"
# On Unix/Mac:
# export PYTHONPATH="$PWD/src"

# Run all tests
python -m pytest

# Run only heartbeat tests
python -m pytest tests/server/test_heartbeat.py -v

# Run only edge API tests
python -m pytest tests/edge_agent/test_edge_api.py -v
```

### Technical Notes

- "Motion events" (from the problem statement) can be sent from our existing software via TCP.
- Ideally, supported security cameras should be compliant with ONVIF Profile S and T standards:
  - [ONVIF Profile S](https://www.onvif.org/profiles/profile-s/)
  - [ONVIF Profile T](https://www.onvif.org/profiles/profile-t/)
- These standards define connection protocols for streaming video, including RTSP.
- Any cameras used will always follow one of these two profiles.

### License

This project is released as open source.