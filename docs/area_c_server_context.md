# BIL Security ML - Central Server (Area C)

## Project Context Document for AI Agent

**Owner:** Ebere Onyelechie  
**Area:** C - Central Server + Dashboard  
**Date:** February 4, 2026  
**Course:** COMP 4560 Industrial Project (Winter 2026)

---

## 1. Project Overview

This is a **distributed edge computing system** for security camera intrusion detection, built in collaboration with BIL Security. The system filters false alarms (caused by weather, animals, vegetation) from real security threats (people, vehicles).

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CENTRAL SERVER (Area C)                     │
│                     (BIL Security Office)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Alert       │  │ Dashboard   │  │ Model/Config            │ │
│  │ Aggregator  │  │ & Monitoring│  │ Distribution Service    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         ▲                                      │
         │ Alerts (HTTPS POST)                  │ Updates (HTTPS GET)
         │                                      ▼
┌────────┴────────┐  ┌─────────────────┐  ┌─────────────────┐
│   EDGE PC #1    │  │   EDGE PC #2    │  │   EDGE PC #N    │
│  Customer Site  │  │  Customer Site  │  │  Customer Site  │
│  i5/i7, 4-8GB   │  │  i5/i7, 4-8GB   │  │  i5/i7, 4-8GB   │
│  Runs detection │  │  Runs detection │  │  Runs detection │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### How the System Works

1. **Edge PCs** are deployed at customer sites (one per location)
2. Each edge PC connects to local security cameras via RTSP
3. Existing BIL Security software detects motion and sends events to the edge PC
4. Edge PC runs YOLOv8 object detection on video clips around the motion event
5. If a person/vehicle is detected, edge PC sends an **alert** to the Central Server
6. Edge PCs periodically **poll** the server for model/config updates

### Team Structure (4 people)

| Area | Owner | Responsibility |
|------|-------|----------------|
| A | TBD | Models & Benchmarking - YOLOv8 optimization, model comparison |
| B | TBD | Edge Agent - Python service running at customer sites |
| **C** | **Ebere** | **Central Server + Dashboard - THIS IS YOUR AREA** |
| D | TBD | Integration, Testing & Docs |

---

## 2. Your Responsibility: Area C

You are building the **Central Server** that runs at BIL Security's office. It has 3 main functions:

### 2.1 Alert Ingestion API
- Receive alerts from all edge PCs via HTTPS POST
- Store alerts in a database
- Each alert contains: site_id, camera_id, timestamp, detection details, optional image

### 2.2 Web Dashboard
- View alerts from all customer sites in one place
- Filter by site, camera, date, object type
- Show edge PC status (online/offline, last heartbeat)
- View alert images

### 2.3 Model & Config Distribution
- Serve model files (YOLOv8 weights ~6MB) to edge PCs
- Serve configuration (detection zones, thresholds) per site
- Version tracking - know what version each edge is running
- Edge PCs poll for updates periodically

---

## 3. Technical Requirements

### 3.1 Technology Stack
- **Framework:** FastAPI (Python) - async, fast, auto-generates OpenAPI docs
- **Database:** SQLite for development, PostgreSQL for production
- **Frontend:** Streamlit (rapid prototyping) OR simple HTML/Jinja2 templates
- **File Storage:** Local filesystem for models and alert images

### 3.2 API Endpoints to Build

```
# Alerts
POST /api/alerts              - Receive alert from edge PC
GET  /api/alerts              - List alerts (with filters)
GET  /api/alerts/{id}         - Get single alert details

# Edge PC Management
POST /api/heartbeat           - Edge PC health check (every 1-5 min)
GET  /api/edges               - List all registered edge PCs
GET  /api/edges/{site_id}     - Get edge PC status

# Updates Distribution
GET  /api/updates/check       - Check for available updates
GET  /api/models/{version}    - Download model file
GET  /api/config/{site_id}    - Get site configuration
```

### 3.3 Data Models

#### Alert
```python
{
    "id": "uuid",
    "site_id": "site_001",
    "camera_id": "cam_01",
    "timestamp": "2026-02-04T14:30:00Z",
    "detections": [
        {
            "class": "person",
            "confidence": 0.87,
            "bbox": [100, 200, 300, 400]
        }
    ],
    "image_path": "/storage/alerts/site_001/2026-02-04/img_123.jpg",
    "acknowledged": false,
    "notes": ""
}
```

#### Edge PC Status
```python
{
    "site_id": "site_001",
    "site_name": "ABC Company - Main Entrance",
    "last_heartbeat": "2026-02-04T14:29:55Z",
    "status": "online",  # online, offline, error
    "model_version": "1.2.0",
    "config_version": "2026-01-15",
    "cameras": ["cam_01", "cam_02"],
    "alerts_today": 5
}
```

#### Update Check Response
```python
# Edge sends: GET /api/updates/check?site_id=site_001&model_version=1.2.0&config_version=2026-01-15

# Server responds:
{
    "model": {
        "update_available": true,
        "latest_version": "1.3.0",
        "download_url": "/api/models/1.3.0",
        "file_size_mb": 6.2
    },
    "config": {
        "update_available": false,
        "latest_version": "2026-01-15"
    }
}
```

---

## 4. Project Structure to Create

**You are starting from an empty repo (just LICENSE and README.md).**

Create this structure:

```
server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings (database URL, etc.)
│   ├── database.py          # SQLAlchemy setup
│   │
│   ├── models/              # SQLAlchemy database models
│   │   ├── __init__.py
│   │   ├── alert.py         # Alert table
│   │   └── edge.py          # EdgePC table
│   │
│   ├── schemas/             # Pydantic models for request/response
│   │   ├── __init__.py
│   │   ├── alert.py
│   │   └── edge.py
│   │
│   └── routes/              # API endpoint handlers
│       ├── __init__.py
│       ├── alerts.py        # /api/alerts endpoints
│       ├── heartbeat.py     # /api/heartbeat endpoint
│       ├── edges.py         # /api/edges endpoints
│       └── updates.py       # /api/updates, /api/models, /api/config
│
├── dashboard/               # Web UI
│   └── app.py               # Streamlit dashboard
│
├── storage/                 # File storage
│   ├── models/              # Model files (.pt) to distribute
│   ├── configs/             # Site config JSON files
│   └── alerts/              # Uploaded alert images
│
├── tests/
│   └── test_alerts.py
│
├── requirements.txt
└── README.md
```

---

## 5. Interface with Edge Agent (Area B)

The edge agent will send these requests to your server:

### Alert Submission
```python
POST /api/alerts
Content-Type: application/json

{
    "site_id": "site_001",
    "camera_id": "cam_01",
    "timestamp": "2026-02-04T14:30:00Z",
    "detections": [
        {
            "class": "person",
            "confidence": 0.87,
            "bbox": [100, 200, 300, 400]
        },
        {
            "class": "car",
            "confidence": 0.72,
            "bbox": [400, 300, 600, 500]
        }
    ],
    "image_base64": "iVBORw0KGgo..."  # Optional, base64 encoded JPEG
}
```

### Heartbeat (every 1-5 minutes)
```python
POST /api/heartbeat
Content-Type: application/json

{
    "site_id": "site_001",
    "model_version": "1.2.0",
    "config_version": "2026-01-15",
    "cameras_active": 3,
    "memory_usage_mb": 1200,
    "cpu_percent": 45,
    "uptime_hours": 72.5
}
```

### Update Check
```python
GET /api/updates/check?site_id=site_001&model_version=1.2.0&config_version=2026-01-15
```

---

## 6. Implementation Phases

### Phase 1: Core API (Week 1)
1. Set up FastAPI project structure
2. Create SQLite database with SQLAlchemy
3. Implement `POST /api/alerts` - receive and store alerts
4. Implement `POST /api/heartbeat` - track edge PC status
5. Implement `GET /api/alerts` - list with pagination
6. Implement `GET /api/edges` - list edge PCs with status

### Phase 2: Dashboard (Week 2)
1. Create Streamlit dashboard
2. Alert list page with filters (site, date, object type)
3. Alert detail page showing image and detections
4. Edge status page (which sites are online/offline)

### Phase 3: Update Distribution (Week 3)
1. Implement `GET /api/updates/check`
2. Implement `GET /api/models/{version}` - serve model files
3. Implement `GET /api/config/{site_id}` - serve site configs
4. Track versions per edge in database

### Phase 4: Polish (Week 4)
1. Add API key authentication
2. Alert acknowledgment feature
3. Basic stats (alerts per day chart)
4. Error handling and logging

---

## 7. Quick Start

```bash
# Create project directory
mkdir server
cd server

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install fastapi uvicorn sqlalchemy pydantic python-multipart aiofiles

# Create requirements.txt
pip freeze > requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000

# API docs automatically available at:
# http://localhost:8000/docs
```

---

## 8. Example Code to Start With

### app/main.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routes import alerts, heartbeat, edges, updates

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="BIL Security Central Server",
    description="Aggregates alerts from edge PCs and distributes updates",
    version="0.1.0"
)

# Allow CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(alerts.router, prefix="/api", tags=["Alerts"])
app.include_router(heartbeat.router, prefix="/api", tags=["Heartbeat"])
app.include_router(edges.router, prefix="/api", tags=["Edges"])
app.include_router(updates.router, prefix="/api", tags=["Updates"])

@app.get("/")
def root():
    return {"status": "ok", "message": "BIL Security Central Server"}
```

### app/database.py
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./bil_security.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## 9. Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Framework | FastAPI | Async, auto API docs, Python (same as edge) |
| Database | SQLite | Simple for development, easy to swap to PostgreSQL |
| Auth | API keys in header | Simple, sufficient for internal system |
| Dashboard | Streamlit | Rapid prototyping, Python-native |
| File storage | Local filesystem | Simple, can migrate to S3 later |

---

## 10. Success Criteria

- [ ] Edge PCs can POST alerts and they're stored in database
- [ ] Dashboard shows alerts from multiple sites with filtering
- [ ] Heartbeat updates edge PC status (online/offline detection)
- [ ] Edge PCs can check for and download model updates
- [ ] Server knows what version each edge is running
- [ ] Basic search and filtering in dashboard

---

## 11. Background Context

### What is this project?
A security monitoring system that uses AI (YOLOv8) to filter false alarms. Security cameras often trigger on wind, animals, shadows - this system detects actual people/vehicles.

### Why edge + server architecture?
- **Edge PCs** at customer sites: Run detection locally (no video streaming bandwidth)
- **Central Server** at BIL office: Aggregate alerts, push updates to all sites

### Hardware constraints
- Edge PCs: Intel i5/i7, 4-8GB RAM, NO GPU
- YOLOv8 Nano runs at ~20 FPS on CPU
- Event-driven: only process when motion detected

### What other team members are building
- **Area A:** Optimizing YOLOv8, comparing models, fine-tuning
- **Area B:** Edge agent service (detection, sends alerts to you)
- **Area D:** Testing, documentation, making it all work together

---

## 12. Resources

- FastAPI docs: https://fastapi.tiangolo.com/
- SQLAlchemy docs: https://docs.sqlalchemy.org/
- Streamlit docs: https://docs.streamlit.io/
- Pydantic docs: https://docs.pydantic.dev/

---

**START HERE:** Create the project structure, then implement `POST /api/alerts` and `GET /api/alerts` first. That's the core functionality everything else builds on.
