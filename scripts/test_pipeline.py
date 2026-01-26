#!/usr/bin/env python3
"""
End-to-end pipeline test with fake camera and TCP events.

This script demonstrates the complete workflow:
1. Starts a fake camera generating frames
2. Feeds frames into a ring buffer
3. Listens for TCP events
4. When an event arrives, extracts frames and saves as clip

Usage:
    python scripts/test_pipeline.py

Then in another terminal:
    python scripts/send_test_event.py cam_01

The script will save an event clip to the event_clips/ folder.
"""

import sys
import asyncio
import threading
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import TCPEventListener, MotionEvent
from pipeline import (
    FakeCamera,
    MultiCameraBufferManager,
    EventFrameWindowSelector,
    EventClipSaver
)


# Global components
buffer_manager = MultiCameraBufferManager(max_duration_seconds=30.0)
clip_saver = EventClipSaver(output_dir="event_clips", save_format="video")  # Save as MP4
window_selector = EventFrameWindowSelector(
    buffer_manager=buffer_manager,
    clip_saver=clip_saver,
    before_seconds=2.0,
    after_seconds=5.0
)


def handle_frame(camera_id: str, frame):
    """Callback for new frames from the fake camera."""
    buffer_manager.add_frame(camera_id, frame)


def handle_event(event: MotionEvent) -> None:
    """Process incoming motion events."""
    print("\n" + "=" * 60)
    print("📢 MOTION EVENT RECEIVED")
    print("=" * 60)
    print(f"  Event ID:    {event.event_id}")
    print(f"  Camera ID:   {event.camera_id}")
    print(f"  Event Time:  {event.event_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print("=" * 60)
    
    # Check buffer stats
    stats = buffer_manager.get_buffer_stats()
    if event.camera_id in stats:
        s = stats[event.camera_id]
        print(f"  Buffer has {s['frame_count']} frames ({s['buffer_duration']:.1f}s)")
    
    # Process event in background thread to not block TCP listener
    def process():
        print(f"\n⏳ Extracting frames for event (waiting {window_selector.after_seconds}s for future frames)...")
        clip_path = window_selector.process_event(
            camera_id=event.camera_id,
            event_time=event.event_time,
            event_id=event.event_id,
            wait_for_future_frames=True
        )
        
        if clip_path:
            print(f"\n✅ Event clip saved to: {clip_path}")
        else:
            print(f"\n❌ Failed to save event clip")
    
    thread = threading.Thread(target=process, daemon=True)
    thread.start()


async def run_tcp_listener():
    """Run the TCP event listener."""
    listener = TCPEventListener(
        host="0.0.0.0",
        port=9000,
        on_event=handle_event
    )
    await listener.start()


def main():
    print("=" * 60)
    print("End-to-End Pipeline Test")
    print("=" * 60)
    print()
    
    # Create and start fake camera
    camera_id = "cam_01"
    print(f"Starting fake camera '{camera_id}' at 15 FPS...")
    camera = FakeCamera(camera_id=camera_id, fps=15, resolution=(640, 480))
    camera.start(lambda f: handle_frame(camera_id, f))
    
    print("Waiting for buffer to fill...")
    time.sleep(3)  # Let buffer accumulate some frames
    
    stats = buffer_manager.get_buffer_stats()
    print(f"Buffer stats: {stats}")
    print()
    
    print("TCP Event Listener starting on port 9000...")
    print()
    print("To test, run in another terminal:")
    print("  python scripts/send_test_event.py cam_01")
    print()
    print("Press Ctrl+C to stop.")
    print()
    
    try:
        asyncio.run(run_tcp_listener())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
