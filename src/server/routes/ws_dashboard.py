from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ..services.auth import TokenError, verify_access_token
from ..services.dashboard_events import DashboardEventManager

router = APIRouter(tags=["dashboard-events"])
_DASHBOARD_COOKIE = "bil_dashboard_session"


@router.websocket("/ws/dashboard-events")
async def dashboard_events_websocket(websocket: WebSocket) -> None:
    dashboard_session = websocket.cookies.get(_DASHBOARD_COOKIE)
    if not dashboard_session:
        await websocket.close(code=4401, reason="Dashboard login required")
        return
    try:
        verify_access_token(dashboard_session)
    except TokenError:
        await websocket.close(code=4401, reason="Dashboard login required")
        return

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
