from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from ..config import EdgeSettings
from .tcp_parse import parse_motion_xml
from .types import MotionEvent

logger = logging.getLogger(__name__)


class TcpMotionTrigger:
    """
    Async TCP server that receives VMS motion XML packets and emits MotionEvent objects.
    Minimal deps: stdlib only.
    """

    def __init__(self, cfg: EdgeSettings, *, queue_max: int = 1000) -> None:
        self._cfg = cfg
        self._queue: asyncio.Queue[MotionEvent] = asyncio.Queue(maxsize=queue_max)
        self._server: asyncio.AbstractServer | None = None
        self._bound_host: str | None = None
        self._bound_port: int | None = None

    @property
    def queue(self) -> asyncio.Queue[MotionEvent]:
        return self._queue

    @property
    def bound_port(self) -> int | None:
        return self._bound_port

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._cfg.tcp_host,
            port=self._cfg.tcp_port,
        )
        sockets = self._server.sockets or []

        if sockets:
            host, port = sockets[0].getsockname()[:2]
            self._bound_host = str(host)
            self._bound_port = int(port)

        bind_info = ", ".join(str(s.getsockname()) for s in sockets)
        logger.info("TCP motion listener started on %s", bind_info)

    async def stop(self) -> None:
        if self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None

        self._bound_host = None
        self._bound_port = None

        logger.info("TCP motion listener stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            # Read up to a sane limit; motion payloads should be small.
            data = await asyncio.wait_for(reader.read(64_000), timeout=5.0)
            if not data:
                return

            text = data.decode("utf-8", errors="replace").strip()
            fields = parse_motion_xml(text)

            now = datetime.now(timezone.utc)
            evt = MotionEvent(
                received_at_utc=now,
                site_id=self._cfg.site_id,
                edge_pc_id=self._cfg.edge_pc_id,
                camera_id=fields.get("camera_id"),
                camera_name=fields.get("camera_name"),
                policy_id=fields.get("policy_id"),
                policy_name=fields.get("policy_name"),
                user_string=fields.get("user_string"),
                raw_xml=text,
                source="tcp",
            )

            try:
                self._queue.put_nowait(evt)
            except asyncio.QueueFull:
                logger.warning("Motion queue full; dropping event from %s", peer)

        except Exception:
            logger.exception("Error handling TCP motion packet from %s", peer)
        finally:
            from contextlib import suppress

            with suppress(Exception):
                writer.close()
                await writer.wait_closed()
