from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket
from pydantic import ValidationError

from bil_time import ensure_winnipeg, now_in_winnipeg

from ..db import SessionLocal
from ..schemas import AlertCreate
from ..services.dashboard_events import publish_dashboard_event
from ..services.edge_authorization import is_authorized_edge_pc, resolve_edge_pc_id
from ..services.image_storage import ImageStorageError, ImageStorageService
from ..services.ws_alert_dispatcher import (
    AlertDispatchFailure,
    AlertQueueFullError,
    AlertValidationFailure,
    WebSocketAlertDispatcher,
)
from ..services.ws_connection_manager import WebSocketConnectionManager

router = APIRouter(tags=["alerts-websocket"])
logger = logging.getLogger(__name__)


def _safe_identity(value: str | None) -> str:
    if not value:
        return "<missing>"
    return value[:64]


def _is_authorized_edge_sender(edge_pc_id: str | None) -> tuple[bool, str]:
    resolved_edge_id = resolve_edge_pc_id(edge_pc_id)
    db = SessionLocal()
    try:
        return is_authorized_edge_pc(db, resolved_edge_id), resolved_edge_id
    finally:
        db.close()


def _extract_alert_payload(message: Any, *, strip_type: bool = False) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise TypeError("Expected a JSON object payload")
    if "alert" in message:
        payload = message["alert"]
    elif "payload" in message:
        payload = message["payload"]
    elif strip_type and "type" in message:
        payload = {k: v for k, v in message.items() if k != "type"}
    else:
        payload = message
    if not isinstance(payload, dict):
        raise TypeError("Expected alert payload to be a JSON object")
    return payload


@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket) -> None:
    manager: WebSocketConnectionManager | None = getattr(
        websocket.app.state, "ws_connection_manager", None
    )
    dispatcher: WebSocketAlertDispatcher | None = getattr(
        websocket.app.state, "ws_alert_dispatcher", None
    )
    image_storage: ImageStorageService | None = getattr(
        websocket.app.state, "ws_image_storage", None
    )
    max_image_bytes: int = int(
        getattr(websocket.app.state, "ws_max_image_bytes", 5_000_000)
    )

    if manager is None or dispatcher is None or image_storage is None:
        await websocket.close(
            code=1011, reason="WebSocket alert subsystem not initialized"
        )
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
                "max_image_bytes": max_image_bytes,
            },
        )

        pending_alert_payload: dict[str, Any] | None = None
        while True:
            incoming = await websocket.receive()
            incoming_type = incoming.get("type")
            if incoming_type == "websocket.disconnect":
                return
            if incoming_type != "websocket.receive":
                continue

            binary_payload = incoming.get("bytes")
            text_payload = incoming.get("text")

            if binary_payload is not None:
                if pending_alert_payload is None:
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "meta_missing",
                            "message": "Send an 'alert_meta' JSON frame before binary image bytes",
                        },
                    )
                    continue
                if len(binary_payload) == 0:
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "invalid_message",
                            "message": "Binary image frame is empty",
                        },
                    )
                    continue
                if len(binary_payload) > max_image_bytes:
                    pending_alert_payload = None
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "image_too_large",
                            "message": f"Image exceeds max size ({max_image_bytes} bytes)",
                        },
                    )
                    continue

                alert_payload = pending_alert_payload
                pending_alert_payload = None
                site_id = str(alert_payload.get("site_id", "unknown"))
                camera_id = str(alert_payload.get("camera_id", "unknown"))
                raw_timestamp = alert_payload.get("timestamp")
                try:
                    received_at = ensure_winnipeg(
                        datetime.fromisoformat(str(raw_timestamp))
                    )
                except Exception:
                    received_at = now_in_winnipeg()
                try:
                    # Save image using the configured websocket image storage
                    # instance. Prefix the returned path with a special scheme so
                    # the ingestion pipeline can know this file was written by
                    # the websocket storage and avoid copying it again.
                    saved = await asyncio.to_thread(
                        image_storage.save_alert_image,
                        site_id=site_id,
                        camera_id=camera_id,
                        image_bytes=binary_payload,
                        received_at=received_at,
                    )
                    image_path = f"ws://{saved}"
                except ImageStorageError as exc:
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "image_store_failed",
                            "message": str(exc),
                        },
                    )
                    continue

                alert_payload = {**alert_payload, "image_path": image_path}
                authorized, resolved_edge_id = await asyncio.to_thread(
                    _is_authorized_edge_sender,
                    alert_payload.get("edge_pc_id"),
                )
                if not authorized:
                    logger.warning(
                        "Rejecting websocket alert ingestion from unregistered edge_pc_id=%s",
                        _safe_identity(resolved_edge_id),
                    )
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "unauthorized",
                            "status": 403,
                            "message": "Edge PC is not authorized to submit alerts",
                        },
                    )
                    continue
                try:
                    alert_out = await dispatcher.submit(alert_payload)
                except AlertValidationFailure as exc:
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "validation_error",
                            "errors": exc.errors,
                        },
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
                        {
                            "type": "error",
                            "code": "ingestion_error",
                            "message": str(exc),
                        },
                    )
                    continue

                await manager.send_json(
                    websocket,
                    {"type": "ack", "status": "ok", "alert": alert_out},
                )
                publish_dashboard_event(
                    websocket.app,
                    "alert_received",
                    {
                        "id": alert_out.get("id"),
                        "site_id": alert_out.get("site_id"),
                        "camera_id": alert_out.get("camera_id"),
                        "edge_pc_id": alert_out.get("edge_pc_id"),
                        "timestamp": alert_out.get("timestamp"),
                        "received_at": alert_out.get("received_at"),
                    },
                )
                continue

            if text_payload is None:
                await manager.send_json(
                    websocket,
                    {
                        "type": "error",
                        "code": "invalid_message",
                        "message": "Expected a text JSON frame or binary image frame",
                    },
                )
                continue

            try:
                incoming_json = json.loads(text_payload)
            except json.JSONDecodeError:
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
                if (
                    isinstance(incoming_json, dict)
                    and incoming_json.get("type") == "alert_meta"
                ):
                    if pending_alert_payload is not None:
                        await manager.send_json(
                            websocket,
                            {
                                "type": "error",
                                "code": "binary_expected",
                                "message": "Binary frame expected for previously received metadata",
                            },
                        )
                        continue
                    candidate_payload = _extract_alert_payload(
                        incoming_json, strip_type=True
                    )
                    try:
                        pending_alert_payload = AlertCreate.model_validate(
                            candidate_payload
                        ).model_dump(
                            mode="json",
                            by_alias=True,
                            exclude_none=True,
                        )
                    except ValidationError as exc:
                        await manager.send_json(
                            websocket,
                            {
                                "type": "error",
                                "code": "validation_error",
                                "errors": exc.errors(),
                            },
                        )
                        continue
                    await manager.send_json(
                        websocket,
                        {
                            "type": "meta_received",
                            "status": "ok",
                            "message": "Metadata accepted; send binary image frame next",
                        },
                    )
                    continue

                if pending_alert_payload is not None:
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "binary_expected",
                            "message": "Binary frame expected for previously received metadata",
                        },
                    )
                    continue

                alert_payload = _extract_alert_payload(incoming_json)
                authorized, resolved_edge_id = await asyncio.to_thread(
                    _is_authorized_edge_sender,
                    alert_payload.get("edge_pc_id"),
                )
                if not authorized:
                    logger.warning(
                        "Rejecting websocket alert ingestion from unregistered edge_pc_id=%s",
                        _safe_identity(resolved_edge_id),
                    )
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "code": "unauthorized",
                            "status": 403,
                            "message": "Edge PC is not authorized to submit alerts",
                        },
                    )
                    continue
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
            publish_dashboard_event(
                websocket.app,
                "alert_received",
                {
                    "id": alert_out.get("id"),
                    "site_id": alert_out.get("site_id"),
                    "camera_id": alert_out.get("camera_id"),
                    "edge_pc_id": alert_out.get("edge_pc_id"),
                    "timestamp": alert_out.get("timestamp"),
                    "received_at": alert_out.get("received_at"),
                },
            )
    finally:
        await manager.disconnect(websocket)
