from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from .config import EdgeSettings


class HealthOut(BaseModel):
    status: str
    time_utc: datetime

    model_config = {"json_schema_extra": {"examples": [{"status": "ok", "time_utc": "2026-02-18T12:00:00Z"}]}}


class HeartbeatOut(BaseModel):
    edge_pc_id: str
    site_name: str
    status: str
    time_utc: datetime
    uptime_seconds: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "edge_pc_id": "edge-001",
                    "site_name": "Warehouse 1",
                    "status": "online",
                    "time_utc": "2026-02-18T12:00:00Z",
                    "uptime_seconds": 42,
                }
            ]
        }
    }


def create_app(cfg: EdgeSettings) -> FastAPI:
    """
    Create the Edge Agent HTTP API app.
    """
    app = FastAPI(
        title="BIL Security ML - Edge Agent API",
        version="0.2.0",
        description="Edge-side health endpoints for install/debug and office connectivity checks.",
    )

    @app.get("/")
    def root():
        return {"status": "edge agent running"}

    # Store start time for uptime calculation
    started_monotonic = time.monotonic()

    @app.get("/health", response_model=HealthOut, tags=["health"])
    def health() -> HealthOut:
        """Liveness check: returns OK if the edge agent process is running."""
        ...
        return HealthOut(status="ok", time_utc=datetime.now(timezone.utc))

    @app.get("/heartbeat", response_model=HeartbeatOut, tags=["health"])
    def heartbeat() -> HeartbeatOut:
        """Returns edge identity + basic status snapshot + uptime."""
        ...
        uptime = int(time.monotonic() - started_monotonic)

        # For now, status is always "online" because PR2 doesn't have motion logic yet.
        # Later PRs can change this based on last motion time, stream status, etc.
        return HeartbeatOut(
            edge_pc_id=cfg.edge_pc_id,
            site_name=cfg.site_name,
            # NOTE(PR2): status is hardcoded because PR2 only implements the Edge HTTP API.
            # TODO(PR3/PR4): derive real status from health signals (TCP listener running,
            # RTSP connectivity, last motion received, queue depth, etc.)
            status="online",
            time_utc=datetime.now(timezone.utc),
            uptime_seconds=uptime,
        )

    return app
