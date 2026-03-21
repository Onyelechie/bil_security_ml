from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ..services.dashboard_events import DashboardEventManager

router = APIRouter(tags=["dashboard-events"])


@router.websocket("/ws/dashboard-events")
async def dashboard_events_websocket(websocket: WebSocket) -> None:
    manager: DashboardEventManager | None = getattr(websocket.app.state, "dashboard_event_manager", None)
    if manager is None:
        await websocket.close(code=1011, reason="Dashboard event subsystem not initialized")
        return

    await manager.connect(websocket)
    try:
        while True:
            incoming = await websocket.receive()
            if incoming.get("type") == "websocket.disconnect":
                return
    finally:
        await manager.disconnect(websocket)
