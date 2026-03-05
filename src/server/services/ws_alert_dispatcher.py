from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..db import SessionLocal
from ..schemas import AlertCreate, AlertOut
from .alert_ingestion import AlertIngestionService, AlertPersistenceError


class AlertQueueFullError(RuntimeError):
    """Raised when the ingestion queue is full."""


class AlertValidationFailure(RuntimeError):
    """Raised when an incoming payload fails schema validation."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__("Invalid alert payload")
        self.errors = errors


class AlertDispatchFailure(RuntimeError):
    """Raised when persistence fails during queued processing."""


@dataclass(slots=True)
class _AlertJob:
    payload: dict[str, Any]
    result_future: asyncio.Future[dict[str, Any]]


_STOP = object()


class WebSocketAlertDispatcher:
    """Queue-backed dispatcher to process alert payloads from many sockets."""

    def __init__(
        self,
        worker_count: int,
        queue_size: int,
        ingestion_service: AlertIngestionService | None = None,
    ) -> None:
        self._worker_count = worker_count
        self._queue: asyncio.Queue[_AlertJob | object] = asyncio.Queue(
            maxsize=queue_size
        )
        self._workers: list[asyncio.Task] = []
        self._ingestion_service = ingestion_service or AlertIngestionService()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(), name=f"ws-alert-worker-{idx}")
            for idx in range(self._worker_count)
        ]

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            await self._queue.put(_STOP)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._running:
            raise AlertDispatchFailure("Alert dispatcher is not running")

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        job = _AlertJob(payload=payload, result_future=result_future)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise AlertQueueFullError("Alert queue is full") from exc
        return await result_future

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is _STOP:
                    return
                if not isinstance(job, _AlertJob):
                    continue
                try:
                    result = await asyncio.to_thread(self._process_payload, job.payload)
                except Exception as exc:  # noqa: BLE001
                    if not job.result_future.done():
                        job.result_future.set_exception(exc)
                else:
                    if not job.result_future.done():
                        job.result_future.set_result(result)
            finally:
                self._queue.task_done()

    def _process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            alert = AlertCreate.model_validate(payload)
        except ValidationError as exc:
            raise AlertValidationFailure(exc.errors()) from exc

        db = SessionLocal()
        try:
            created_alert = self._ingestion_service.ingest(db, alert)
        except AlertPersistenceError as exc:
            raise AlertDispatchFailure(str(exc)) from exc
        finally:
            db.close()

        return AlertOut.model_validate(created_alert).model_dump(
            mode="json", by_alias=True
        )
