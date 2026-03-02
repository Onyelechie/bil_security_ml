from __future__ import annotations

import asyncio
from collections.abc import Iterable

from fastapi import WebSocket


class WebSocketConnectionManager:
    """Tracks active WebSocket clients with a configurable connection cap."""

    def __init__(self, max_connections: int) -> None:
        self._max_connections = max_connections
        self._active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        await websocket.accept()
        async with self._lock:
            if len(self._active_connections) >= self._max_connections:
                await websocket.close(code=1013, reason="Too many active WebSocket connections")
                return False
            self._active_connections.add(websocket)
            return True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._active_connections.discard(websocket)

    async def send_json(self, websocket: WebSocket, payload: dict) -> None:
        await websocket.send_json(payload)

    def snapshot(self) -> Iterable[WebSocket]:
        return tuple(self._active_connections)

    @property
    def active_count(self) -> int:
        return len(self._active_connections)
