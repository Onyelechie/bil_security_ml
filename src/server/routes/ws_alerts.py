from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.ws_alert_dispatcher import (
    AlertDispatchFailure,
    AlertQueueFullError,
    AlertValidationFailure,
    WebSocketAlertDispatcher,
)
from ..services.ws_connection_manager import WebSocketConnectionManager

router = APIRouter(tags=["alerts-websocket"])


def _extract_alert_payload(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise TypeError("Expected a JSON object payload")
    payload = message.get("alert", message)
    if not isinstance(payload, dict):
        raise TypeError("Expected 'alert' to be a JSON object")
    return payload


@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket) -> None:
    manager: WebSocketConnectionManager | None = getattr(websocket.app.state, "ws_connection_manager", None)
    dispatcher: WebSocketAlertDispatcher | None = getattr(websocket.app.state, "ws_alert_dispatcher", None)

    if manager is None or dispatcher is None:
        await websocket.close(code=1011, reason="WebSocket alert subsystem not initialized")
        return

    accepted = await manager.connect(websocket)
    if not accepted:
        return

    try:
        await manager.send_json(
            websocket,
            {
                "type": "connected",
                "status": "ok",
                "message": "WebSocket alert ingestion channel ready",
            },
        )

        while True:
            try:
                incoming = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            except (TypeError, ValueError):
                await manager.send_json(
                    websocket,
                    {
                        "type": "error",
                        "code": "invalid_message",
                        "message": "Expected a valid JSON object message",
                    },
                )
                continue

            try:
                alert_payload = _extract_alert_payload(incoming)
                alert_out = await dispatcher.submit(alert_payload)
            except TypeError as exc:
                await manager.send_json(
                    websocket,
                    {"type": "error", "code": "invalid_message", "message": str(exc)},
                )
                continue
            except AlertValidationFailure as exc:
                await manager.send_json(
                    websocket,
                    {"type": "error", "code": "validation_error", "errors": exc.errors},
                )
                continue
            except AlertQueueFullError as exc:
                await manager.send_json(
                    websocket,
                    {"type": "error", "code": "queue_full", "message": str(exc)},
                )
                continue
            except AlertDispatchFailure as exc:
                await manager.send_json(
                    websocket,
                    {"type": "error", "code": "ingestion_error", "message": str(exc)},
                )
                continue

            await manager.send_json(
                websocket,
                {"type": "ack", "status": "ok", "alert": alert_out},
            )
    finally:
        await manager.disconnect(websocket)
