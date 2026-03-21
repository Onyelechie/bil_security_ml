from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging
from threading import Lock
from typing import TypedDict


class LogEntry(TypedDict):
    id: int
    timestamp: str
    level: str
    logger: str
    message: str


class InMemoryLogBuffer:
    """Thread-safe bounded buffer for structured server log entries."""

    def __init__(self, *, max_entries: int = 5000) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=max_entries)
        self._next_id = 1
        self._lock = Lock()

    @property
    def latest_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    def append_record(self, record: logging.LogRecord) -> None:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        entry: LogEntry = {
            "id": 0,
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        with self._lock:
            entry["id"] = self._next_id
            self._next_id += 1
            self._entries.append(entry)

    def list_entries(
        self,
        *,
        limit: int = 200,
        after_id: int | None = None,
        level: str | None = None,
    ) -> list[LogEntry]:
        with self._lock:
            entries = list(self._entries)

        if after_id is not None:
            entries = [entry for entry in entries if entry["id"] > after_id]
        if level is not None:
            normalized = level.upper()
            entries = [entry for entry in entries if entry["level"] == normalized]

        return entries[-limit:]


class InMemoryLogHandler(logging.Handler):
    """Logging handler that forwards LogRecord objects to InMemoryLogBuffer."""

    def __init__(self, buffer: InMemoryLogBuffer) -> None:
        super().__init__(level=logging.NOTSET)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append_record(record)
        except Exception:  # noqa: BLE001
            # Keep logging side-effects from breaking the request path.
            return
