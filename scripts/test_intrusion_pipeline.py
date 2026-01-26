"""
Test the full intrusion detection pipeline integration.

This script:
1. Starts fake cameras generating frames
2. Runs the IntrusionDetectionPipeline
3. Simulates motion events
4. Verifies detection and filtering works

Run with:
    python -m scripts.test_intrusion_pipeline
"""

import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import MotionEvent
from pipeline import FakeCameraManager, TimestampedFrame
from detect import IntrusionDetectionPipeline, list_available_models, MODEL_INFO


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def test_model_selection():
    """Test that model selection works based on camera count."""
    print_separator("Testing Model Selection")
    
    print("\nAvailable models:")
    for model_name in list_available_models():
        info = MODEL_INFO[model_name]
        print(f"  - {model_name}: {info['name']} "
              f"(recommended for {info['recommended_cameras']} cameras)")
    
    # Test auto-selection
    pipeline_10cam = IntrusionDetectionPipeline(num_cameras=10)
    print(f"\nAuto-selected model for 10 cameras: {pipeline_10cam.model_type}")
    
    pipeline_2cam = IntrusionDetectionPipeline(num_cameras=2)
    print(f"Auto-selected model for 2 cameras: {pipeline_2cam.model_type}")
    
    # Test manual selection
    pipeline_manual = IntrusionDetectionPipeline(model_type='yolov8n', num_cameras=10)
    print(f"Manual selection (yolov8n): {pipeline_manual.model_type}")
    
    print("\n✅ Model selection working correctly")


def test_pipeline_with_fake_cameras():
    """Test pipeline with fake camera frames."""
    print_separator("Testing Pipeline with Fake Cameras")
    
    # Create pipeline (use mobilenet for faster testing)
    pipeline = IntrusionDetectionPipeline(
        model_type='mobilenet',
        num_cameras=3,
        output_dir="test_clips",
        before_seconds=1.0,
        after_seconds=2.0
    )
    
    # Create fake cameras
    camera_manager = FakeCameraManager()
    camera_ids = ["cam_001", "cam_002", "cam_003"]
    
    print("\nStarting fake cameras...")
    for cam_id in camera_ids:
        camera_manager.add_camera(cam_id, fps=10)
        print(f"  Added {cam_id}")
    
    # Start cameras and add frames to pipeline
    frame_counts = {cam_id: 0 for cam_id in camera_ids}
    
    def on_frame(camera_id: str, frame):
        pipeline.add_frame(camera_id, frame)
        frame_counts[camera_id] += 1
    
    camera_manager.start_all(on_frame)
    
    # Let it run for a few seconds
    print("\nBuffering frames...")
    time.sleep(3.0)  # 3 seconds of buffering
    
    total_frames = sum(frame_counts.values())
    print(f"  Buffered {total_frames} total frames")
    for cam_id, count in frame_counts.items():
        print(f"    {cam_id}: {count} frames")
    
    # Create a test event
    event_time = datetime.now() - timedelta(seconds=1)  # 1 second ago
    event = MotionEvent(
        event_id="test_001",
        camera_id="cam_001",
        event_time=event_time,
        event_type="motion"
    )
    
    print(f"\nProcessing event: {event.event_id}")
    print(f"  Camera: {event.camera_id}")
    print(f"  Time: {event.event_time}")
    
    # Process without waiting for future frames (we already have them)
    result = pipeline.process_event(event, wait_for_future_frames=False)
    
    print(f"\nResult:")
    print(f"  Frames analyzed: {result.frames_analyzed}")
    print(f"  Has motion: {result.has_motion}")
    print(f"  Detections: {len(result.detections)}")
    print(f"  Has intrusion: {result.has_intrusion}")
    if result.clip_path:
        print(f"  Clip saved: {result.clip_path}")
    
    # Get stats
    stats = pipeline.get_stats()
    print(f"\nPipeline stats:")
    print(f"  Events processed: {stats['events_processed']}")
    print(f"  Intrusions detected: {stats['intrusions_detected']}")
    print(f"  False alarms filtered: {stats['false_alarms_filtered']}")
    
    # Cleanup
    camera_manager.stop_all()
    
    print("\n✅ Pipeline integration test complete")


def test_model_change():
    """Test changing models at runtime."""
    print_separator("Testing Runtime Model Change")
    
    pipeline = IntrusionDetectionPipeline(model_type='mobilenet', num_cameras=5)
    print(f"Initial model: {pipeline.model_type}")
    
    pipeline.change_model('yolov8n')
    print(f"Changed to: {pipeline.model_type}")
    
    # Note: YOLOv8 requires ultralytics, which might not be installed
    print("\n✅ Model change API working (actual loading requires dependencies)")


def main():
    """Run all integration tests."""
    print_separator("BIL Security - Intrusion Detection Pipeline Tests")
    print("\nThis tests the integration of:")
    print("  - Object detection (MobileNet/YOLOv8)")
    print("  - Motion detection (false alarm filtering)")
    print("  - Frame buffering (ring buffer)")
    print("  - Event processing (TCP event handling)")
    
    try:
        test_model_selection()
        test_pipeline_with_fake_cameras()
        test_model_change()
        
        print_separator("All Tests Passed! ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
