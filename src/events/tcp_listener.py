"""
TCP listener for receiving motion events from external software.
Listens on a configurable port and parses incoming JSON messages into MotionEvent objects.
"""

import asyncio
import json
import logging
from typing import Callable, Optional
from datetime import datetime

from .models import MotionEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCPEventListener:
    """
    Asynchronous TCP server that listens for motion events.
    
    Events are expected in JSON format and are normalized into MotionEvent objects.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        on_event: Optional[Callable[[MotionEvent], None]] = None
    ):
        """
        Initialize the TCP event listener.
        
        Args:
            host: Host address to bind to (default: all interfaces)
            port: Port to listen on (default: 9000)
            on_event: Callback function invoked when an event is received
        """
        self.host = host
        self.port = port
        self.on_event = on_event or self._default_event_handler
        self._server: Optional[asyncio.Server] = None
        self._running = False
    
    def _default_event_handler(self, event: MotionEvent) -> None:
        """Default handler that prints the event."""
        logger.info(f"Received event:\n{event}")
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle an incoming client connection."""
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {addr}")
        
        try:
            while True:
                # Read data until newline or connection closed
                data = await reader.readline()
                if not data:
                    break
                
                message = data.decode('utf-8').strip()
                if not message:
                    continue
                
                logger.debug(f"Received raw message: {message}")
                
                try:
                    # Parse JSON message
                    json_data = json.loads(message)
                    
                    # Convert to MotionEvent
                    event = MotionEvent.from_json(json_data)
                    
                    # Invoke callback
                    self.on_event(event)
                    
                    # Send acknowledgment
                    response = json.dumps({
                        "status": "ok",
                        "event_id": event.event_id,
                        "received_at": datetime.now().isoformat()
                    }) + "\n"
                    writer.write(response.encode('utf-8'))
                    await writer.drain()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = json.dumps({
                        "status": "error",
                        "message": f"Invalid JSON: {str(e)}"
                    }) + "\n"
                    writer.write(error_response.encode('utf-8'))
                    await writer.drain()
                    
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    error_response = json.dumps({
                        "status": "error",
                        "message": str(e)
                    }) + "\n"
                    writer.write(error_response.encode('utf-8'))
                    await writer.drain()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            logger.info(f"Connection closed from {addr}")
            writer.close()
            await writer.wait_closed()
    
    async def start(self) -> None:
        """Start the TCP server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )
        self._running = True
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"TCP Event Listener started on {addr[0]}:{addr[1]}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self) -> None:
        """Stop the TCP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False
            logger.info("TCP Event Listener stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running


async def run_listener(
    host: str = "0.0.0.0",
    port: int = 9000,
    on_event: Optional[Callable[[MotionEvent], None]] = None
) -> None:
    """
    Convenience function to run the TCP listener.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        on_event: Callback function for received events
    """
    listener = TCPEventListener(host=host, port=port, on_event=on_event)
    
    try:
        await listener.start()
    except KeyboardInterrupt:
        await listener.stop()


if __name__ == "__main__":
    # Run the listener when executed directly
    asyncio.run(run_listener())
