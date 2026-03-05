from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket


class DashboardEventManager:
    """Broadcasts lightweight server-side events to dashboard websocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        await websocket.send_json(
            {
                "type": "connected",
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, event: dict[str, Any]) -> None:
        async with self._lock:
            clients = tuple(self._clients)

        dead: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(event)
            except Exception:  # noqa: BLE001
                dead.append(websocket)

        if dead:
            async with self._lock:
                for websocket in dead:
                    self._clients.discard(websocket)

    def broadcast_threadsafe(self, loop: asyncio.AbstractEventLoop, event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(lambda: asyncio.create_task(self.broadcast(event)))


def publish_dashboard_event(app: FastAPI, event_type: str, payload: dict[str, Any]) -> None:
    manager: DashboardEventManager | None = getattr(app.state, "dashboard_event_manager", None)
    loop: asyncio.AbstractEventLoop | None = getattr(app.state, "main_event_loop", None)
    if manager is None or loop is None:
        return
    if loop.is_closed():
        return

    try:
        manager.broadcast_threadsafe(
            loop,
            {
                "type": event_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except RuntimeError:
        # Loop may have already shut down (for example between test sessions).
        return
