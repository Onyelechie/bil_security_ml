#!/usr/bin/env python3
"""
Test script to verify the fake camera is generating frames correctly.

Usage:
    python scripts/test_fake_camera.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import FakeCamera, TimestampedFrame


def main():
    print("=" * 60)
    print("Fake Camera Test")
    print("=" * 60)
    print()
    
    camera = FakeCamera(camera_id="test_cam", fps=15, resolution=(640, 480))
    
    frame_count = 0
    start_time = time.time()
    
    print("Generating 30 frames at 15 FPS...")
    print()
    
    for frame in camera.frames():
        frame_count += 1
        print(f"Frame {frame_count:3d} | "
              f"Time: {frame.timestamp.strftime('%H:%M:%S.%f')[:-3]} | "
              f"Shape: {frame.frame.shape}")
        
        if frame_count >= 30:
            break
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed
    
    print()
    print(f"Generated {frame_count} frames in {elapsed:.2f}s")
    print(f"Actual FPS: {actual_fps:.1f}")
    
    # Try saving a frame
    try:
        import cv2
        output_path = "test_frame.jpg"
        cv2.imwrite(output_path, frame.frame)
        print(f"\nSaved last frame to: {output_path}")
    except ImportError:
        print("\nOpenCV not available, skipping frame save test")


if __name__ == "__main__":
    main()
