#!/usr/bin/env python3
"""
Quick end-to-end test that doesn't need TCP.
Simulates receiving an event and extracting frames.

Usage:
    python scripts/quick_test.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import MotionEvent
from pipeline import (
    FakeCamera,
    MultiCameraBufferManager,
    EventFrameWindowSelector,
    EventClipSaver
)


def main():
    print("=" * 60)
    print("Quick End-to-End Test (No TCP)")
    print("=" * 60)
    print()
    
    # Setup components
    buffer_manager = MultiCameraBufferManager(max_duration_seconds=30.0)
    clip_saver = EventClipSaver(output_dir="event_clips", save_format="video")  # Save as MP4
    window_selector = EventFrameWindowSelector(
        buffer_manager=buffer_manager,
        clip_saver=clip_saver,
        before_seconds=2.0,
        after_seconds=3.0  # Shorter for quick test
    )
    
    camera_id = "cam_01"
    
    # Start fake camera
    print(f"1. Starting fake camera '{camera_id}' at 15 FPS...")
    camera = FakeCamera(camera_id=camera_id, fps=15, resolution=(640, 480))
    camera.start(lambda f: buffer_manager.add_frame(camera_id, f))
    
    # Let buffer fill for 3 seconds
    print("2. Filling buffer with 3 seconds of frames...")
    time.sleep(3)
    
    stats = buffer_manager.get_buffer_stats()
    print(f"   Buffer stats: {stats[camera_id]['frame_count']} frames, "
          f"{stats[camera_id]['buffer_duration']:.1f}s duration")
    
    # Simulate a motion event NOW
    print("\n3. Simulating motion event...")
    event = MotionEvent(
        camera_id=camera_id,
        event_time=datetime.now(),
        event_type="motion"
    )
    print(f"   Event: {event.event_id[:8]}... at {event.event_time.strftime('%H:%M:%S')}")
    
    # Wait for future frames (3 seconds worth)
    print(f"\n4. Waiting 3.5s for future frames to accumulate...")
    time.sleep(3.5)
    
    # Extract and save clip
    print("\n5. Extracting frames and saving clip...")
    clip_path = window_selector.process_event(
        camera_id=event.camera_id,
        event_time=event.event_time,
        event_id=event.event_id,
        wait_for_future_frames=False  # Already waited
    )
    
    # Stop camera
    camera.stop()
    
    if clip_path:
        print(f"\n✅ SUCCESS! Event clip saved to: {clip_path}")
        
        # Show video file info
        import os
        file_size = os.path.getsize(clip_path)
        print(f"   Video file size: {file_size / 1024:.1f} KB")
    else:
        print("\n❌ FAILED to save event clip")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
