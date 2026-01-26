#!/usr/bin/env python3
"""
Test script for the TCP motion event listener.

Run this script to start the TCP listener on port 9000.
Then send test events using send_test_event.py or netcat/telnet.

Usage:
    python scripts/test_tcp_listener.py

Example event to send (via netcat):
    echo '{"camera_id": "cam_01", "event_type": "motion"}' | nc localhost 9000
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import TCPEventListener, MotionEvent


def handle_event(event: MotionEvent) -> None:
    """Custom event handler that prints events nicely."""
    print("\n" + "=" * 60)
    print("📢 MOTION EVENT RECEIVED")
    print("=" * 60)
    print(f"  Event ID:    {event.event_id}")
    print(f"  Camera ID:   {event.camera_id}")
    print(f"  Event Time:  {event.event_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"  Event Type:  {event.event_type}")
    if event.raw_data:
        print(f"  Raw Data:    {event.raw_data}")
    print("=" * 60 + "\n")


async def main():
    print("=" * 60)
    print("TCP Motion Event Listener - Test Script")
    print("=" * 60)
    print("Listening on port 9000...")
    print("Send JSON events like:")
    print('  {"camera_id": "cam_01", "event_type": "motion"}')
    print("\nPress Ctrl+C to stop.\n")
    
    listener = TCPEventListener(
        host="0.0.0.0",
        port=9000,
        on_event=handle_event
    )
    
    try:
        await listener.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await listener.stop()


if __name__ == "__main__":
    asyncio.run(main())
