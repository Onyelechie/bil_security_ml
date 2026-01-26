#!/usr/bin/env python3
"""
Send test motion events to the TCP listener.

Usage:
    python scripts/send_test_event.py [camera_id] [event_type]

Examples:
    python scripts/send_test_event.py
    python scripts/send_test_event.py cam_02
    python scripts/send_test_event.py cam_03 motion
"""

import socket
import json
import sys
from datetime import datetime


def send_event(
    camera_id: str = "cam_01",
    event_type: str = "motion",
    host: str = "localhost",
    port: int = 9000
) -> None:
    """Send a test event to the TCP listener."""
    
    event = {
        "camera_id": camera_id,
        "event_time": datetime.now().isoformat(),
        "event_type": event_type
    }
    
    message = json.dumps(event) + "\n"
    
    print(f"Sending event to {host}:{port}")
    print(f"  Camera ID:  {camera_id}")
    print(f"  Event Type: {event_type}")
    print(f"  Timestamp:  {event['event_time']}")
    print()
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.sendall(message.encode('utf-8'))
            
            # Wait for response
            response = sock.recv(1024).decode('utf-8')
            print(f"Response: {response}")
            
    except ConnectionRefusedError:
        print("ERROR: Could not connect to TCP listener.")
        print("Make sure test_tcp_listener.py is running first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    camera_id = sys.argv[1] if len(sys.argv) > 1 else "cam_01"
    event_type = sys.argv[2] if len(sys.argv) > 2 else "motion"
    
    send_event(camera_id=camera_id, event_type=event_type)
