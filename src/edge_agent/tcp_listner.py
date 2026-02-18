from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .framing import extract_action_messages
from .motion_parser import parse_action_xml
from .motion_event import MotionEvent

logger = logging.getLogger(__name__)


@dataclass
class MotionTcpServer:
    """
    Thin wrapper around asyncio TCP server so tests can start it on port 0
    (ephemeral), discover the actual port, and stop it cleanly.
    """
    server: asyncio.base_events.Server                                  # the actual asyncio TCP server
    site_id: str                                                        # used when building MotionEvent
    queue: asyncio.Queue[MotionEvent]                                   # parsed events are pushed here
    stop_after: Optional[int]                                           # if set: stop after N events (useful for tests)
    max_buffer_bytes: int                                               # safety limit against garbage input

    # Event used to signal "stop now" when stop_after is reached
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Track how many events we've successfully parsed
    _event_count: int = 0

    # Track all active client handler tasks so we can cancel them on stop()
    _client_tasks: set[asyncio.Task] = field(default_factory=set)

    @property
    def port(self) -> int:
        """
        Get the port actually bound.
        Important for tests when we start with port=0 (ephemeral port).
        """
        sock = self.server.sockets[0]
        return int(sock.getsockname()[1])

    @property
    def host(self) -> str:
        """
        Get the host/IP actually bound.
        """
        sock = self.server.sockets[0]
        return str(sock.getsockname()[0])

    async def run(self) -> None:
        """
        Run until Ctrl+C (serve_forever) or until stop_after events are parsed.
        """
        addrs = ", ".join(str(s.getsockname()) for s in (self.server.sockets or []))
        logger.info("TCP listener started at %s", addrs)

        # Ensures the server is properly started and cleaned up when this block exits
        async with self.server:
            if self.stop_after is None:
                # Normal mode: run forever (until process is killed / Ctrl+C)
                await self.server.serve_forever()
            else:
                # Test mode: wait until we set stop_event after N events
                await self.stop_event.wait()

        logger.info("TCP listener stopped")

    async def stop(self) -> None:
        """
        Stop the server + cancel active client handlers.
        """
        self.server.close()
        await self.server.wait_closed()

        for t in list(self._client_tasks):
            t.cancel()

        # Give cancellations a chance to propagate
        await asyncio.sleep(0)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Reads bytes, frames <Action>...</Action> messages, parses them, and puts MotionEvent into queue.
        """
        task = asyncio.current_task()
        if task:
            self._client_tasks.add(task)

        peer = writer.get_extra_info("peername")
        logger.info("Client connected: %s", peer)

        buf = bytearray()

        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break  # client closed

                buf.extend(data)

                # Safety: prevent unbounded memory usage if garbage arrives
                if len(buf) > self.max_buffer_bytes:
                    logger.warning(
                        "Buffer exceeded %d bytes from %s; clearing buffer to recover",
                        self.max_buffer_bytes, peer
                    )
                    buf.clear()
                    continue

                for xml_text in extract_action_messages(buf):
                    try:
                        event = parse_action_xml(xml_text, site_id=self.site_id)
                    except Exception as e:
                        # Don't crash server on bad packets
                        logger.error("Failed to parse motion XML from %s: %s", peer, e)
                        continue

                    await self.queue.put(event)
                    self._event_count += 1

                    logger.info(
                        "Motion event parsed: camera_id=%s policy_id=%s user=%s",
                        event.camera_id, event.policy_id, event.user_string
                    )

                    if self.stop_after is not None and self._event_count >= self.stop_after:
                        self.stop_event.set()
                        break

                if self.stop_event.is_set():
                    break

        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

            logger.info("Client disconnected: %s", peer)

            if task:
                self._client_tasks.discard(task)


async def start_motion_tcp_server(
    host: str,
    port: int,
    *,
    site_id: str,
    queue: asyncio.Queue[MotionEvent],
    stop_after: Optional[int] = None,
    max_buffer_bytes: int = 1_000_000,
) -> MotionTcpServer:
    """
    Start TCP server and return MotionTcpServer wrapper.
    Use port=0 to bind an ephemeral port (best for tests).
    """

    # We create an empty placeholder to pass the bound handler access to self.
    server_ref: dict[str, MotionTcpServer] = {}

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await server_ref["srv"]._handle_client(reader, writer)

    server = await asyncio.start_server(handler, host=host, port=port)
    srv = MotionTcpServer(
        server=server,
        site_id=site_id,
        queue=queue,
        stop_after=stop_after,
        max_buffer_bytes=max_buffer_bytes,
    )
    server_ref["srv"] = srv
    return srv
