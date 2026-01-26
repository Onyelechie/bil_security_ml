"""
Test the intrusion detection pipeline with real video files.

This script:
1. Loads video files from data/videos/
2. Runs them through the IntrusionDetectionPipeline
3. Simulates motion events at specific timestamps
4. Outputs detection results

Usage:
    python -m scripts.test_video_detection
    python -m scripts.test_video_detection --video path/to/video.mp4
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import MotionEvent
from pipeline import VideoFileSource, VideoSourceManager, find_videos
from detect import (
    IntrusionDetectionPipeline,
    list_available_models,
    MODEL_INFO,
    get_recommended_model
)


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def test_single_video(video_path: str, model_type: str = None):
    """Test detection on a single video file."""
    print_separator(f"Testing: {Path(video_path).name}")
    
    # Auto-select model if not specified
    if model_type is None:
        model_type = 'mobilenet'  # Fast for testing
    
    # Create pipeline
    pipeline = IntrusionDetectionPipeline(
        model_type=model_type,
        num_cameras=1,
        output_dir="test_clips",
        before_seconds=2.0,
        after_seconds=3.0
    )
    
    # Create video source
    video_source = VideoFileSource(
        video_path=video_path,
        camera_id="cam_001",
        loop=False,  # Don't loop for testing
        playback_speed=1.0
    )
    
    print(f"\nVideo info:")
    print(f"  Resolution: {video_source.width}x{video_source.height}")
    print(f"  FPS: {video_source.fps:.1f}")
    print(f"  Duration: {video_source.duration:.1f}s")
    print(f"  Frames: {video_source.frame_count}")
    print(f"\nModel: {model_type}")
    
    # Track frames
    frame_count = [0]
    event_times = []
    
    def on_frame(frame):
        frame_count[0] += 1
        pipeline.add_frame("cam_001", frame)
        
        # Create events at specific intervals (every 5 seconds)
        if frame_count[0] % int(video_source.fps * 5) == 0:
            event_times.append(frame.timestamp)
    
    # Start video
    print("\nProcessing video...")
    video_source.start(on_frame)
    
    # Wait for video to finish (with timeout)
    timeout = video_source.duration + 10  # Extra buffer
    start_time = time.time()
    
    while video_source.is_running and (time.time() - start_time) < timeout:
        progress = video_source.progress * 100
        print(f"\r  Progress: {progress:.1f}% ({frame_count[0]} frames)", end="", flush=True)
        time.sleep(0.5)
    
    video_source.stop()
    print(f"\n  Completed: {frame_count[0]} frames processed")
    
    # Process accumulated events
    print(f"\nProcessing {len(event_times)} motion events...")
    
    for i, event_time in enumerate(event_times):
        event = MotionEvent(
            event_id=f"event_{i+1:03d}",
            camera_id="cam_001",
            event_time=event_time,
            event_type="motion"
        )
        
        print(f"\n  Event {i+1}: t={event_time.strftime('%H:%M:%S')}")
        result = pipeline.process_event(event, wait_for_future_frames=False)
        
        print(f"    Frames analyzed: {result.frames_analyzed}")
        print(f"    Has motion: {result.has_motion}")
        print(f"    Detections: {len(result.detections)}")
        if result.has_intrusion:
            print(f"    🚨 INTRUSION DETECTED!")
            for det in result.alert_detections[:3]:  # Show top 3
                print(f"       - {det.class_name}: {det.confidence:.1%}")
            if result.clip_path:
                print(f"    Clip: {result.clip_path}")
        else:
            print(f"    ✅ No intrusion")
    
    # Summary
    stats = pipeline.get_stats()
    print_separator("Summary")
    print(f"  Events processed: {stats['events_processed']}")
    print(f"  Intrusions detected: {stats['intrusions_detected']}")
    print(f"  False alarms filtered: {stats['false_alarms_filtered']}")
    if stats['events_processed'] > 0:
        intrusion_rate = stats['intrusions_detected'] / stats['events_processed'] * 100
        print(f"  Intrusion rate: {intrusion_rate:.1f}%")


def test_multiple_videos(video_dir: str, model_type: str = None):
    """Test detection on multiple video files (simulating multiple cameras)."""
    print_separator("Multi-Video Test")
    
    videos = find_videos(video_dir)
    if not videos:
        print(f"No videos found in {video_dir}")
        print("Please add video files (mp4, avi, mov, mkv) to test.")
        return
    
    print(f"\nFound {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.name}")
    
    # Use only first 3 for testing
    videos = videos[:3]
    
    # Auto-select model based on video count
    if model_type is None:
        model_type = get_recommended_model(len(videos))
    
    # Create pipeline
    pipeline = IntrusionDetectionPipeline(
        model_type=model_type,
        num_cameras=len(videos),
        output_dir="test_clips"
    )
    
    # Create video source manager
    manager = VideoSourceManager()
    for i, video in enumerate(videos):
        camera_id = f"cam_{i+1:03d}"
        manager.add_source(camera_id, str(video), loop=False)
    
    print(f"\nModel: {model_type} (auto-selected for {len(videos)} cameras)")
    
    # Track frames per camera
    frame_counts = {cid: 0 for cid in manager.list_sources()}
    
    def on_frame(camera_id, frame):
        frame_counts[camera_id] += 1
        pipeline.add_frame(camera_id, frame)
    
    # Start all videos
    print("\nProcessing videos...")
    manager.start_all(on_frame)
    
    # Run for the duration of shortest video or max 30 seconds
    run_time = min(30, min(s.duration for s in manager.sources.values()))
    
    start_time = time.time()
    while (time.time() - start_time) < run_time:
        stats = manager.get_stats()
        total_frames = sum(frame_counts.values())
        print(f"\r  Frames: {total_frames}", end="", flush=True)
        time.sleep(0.5)
    
    manager.stop_all()
    
    print(f"\n\nFrames per camera:")
    for camera_id, count in frame_counts.items():
        print(f"  {camera_id}: {count}")
    
    print("\n✅ Multi-video test complete")


def main():
    parser = argparse.ArgumentParser(description="Test intrusion detection with video files")
    parser.add_argument("--video", "-v", help="Path to a specific video file")
    parser.add_argument("--dir", "-d", default="data/videos", help="Directory with video files")
    parser.add_argument("--model", "-m", choices=list(MODEL_INFO.keys()), help="Model to use")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print_separator("Available Models")
        for name, info in MODEL_INFO.items():
            print(f"\n{name}:")
            print(f"  Name: {info['name']}")
            print(f"  Classes: {info['classes']}")
            print(f"  Speed: {info['speed']}")
            print(f"  Recommended cameras: {info['recommended_cameras']}")
            print(f"  Requires: {info['requires']}")
        return 0
    
    print_separator("BIL Security - Video Detection Test")
    
    # Check for video directory
    video_dir = Path(args.dir)
    if not video_dir.exists():
        video_dir.mkdir(parents=True)
        print(f"\nCreated video directory: {video_dir}")
        print("Please add video files to this directory for testing.")
    
    if args.video:
        # Test specific video
        if not Path(args.video).exists():
            print(f"Error: Video not found: {args.video}")
            return 1
        test_single_video(args.video, args.model)
    else:
        # Check for videos in directory
        videos = find_videos(str(video_dir))
        if videos:
            test_single_video(str(videos[0]), args.model)
        else:
            print(f"\nNo videos found in {video_dir}")
            print("\nTo test, either:")
            print(f"  1. Add video files to {video_dir}/")
            print("  2. Run with --video path/to/your/video.mp4")
            print("\nSupported formats: mp4, avi, mov, mkv, wmv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
