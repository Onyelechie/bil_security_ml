"""
Model Speed Benchmark

Tests inference speed for each available detection model.
Results are used to update the research documentation.

Usage:
    python -m scripts.benchmark_models
    python -m scripts.benchmark_models --video data/videos/your_video.mp4
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from detect import (
    create_detector,
    list_available_models,
    MODEL_INFO
)
from pipeline import find_videos


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def benchmark_model(model_type: str, frames: list, warmup_frames: int = 10) -> dict:
    """
    Benchmark a single model's inference speed.
    
    Args:
        model_type: Model to test
        frames: List of frames to process
        warmup_frames: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n  Testing {model_type}...", end=" ", flush=True)
    
    try:
        # Create detector
        start_load = time.time()
        detector = create_detector(model_type)
        load_time = time.time() - start_load
        
        # Warmup
        for frame in frames[:warmup_frames]:
            detector.detect(frame, use_tiled=False)
        
        # Benchmark
        times = []
        detections_count = []
        
        for frame in frames:
            start = time.time()
            detections = detector.detect(frame, use_tiled=False)
            elapsed = time.time() - start
            times.append(elapsed)
            detections_count.append(len(detections))
        
        # Calculate stats
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_detections = sum(detections_count) / len(detections_count)
        
        print(f"✓ {fps:.1f} FPS")
        
        return {
            "model": model_type,
            "success": True,
            "load_time_ms": load_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "fps": fps,
            "frames_tested": len(frames),
            "avg_detections": avg_detections,
            "error": None
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "model": model_type,
            "success": False,
            "error": str(e),
            "fps": 0
        }


def load_test_frames(video_path: str = None, num_frames: int = 100, resize: tuple = (640, 480)) -> list:
    """
    Load test frames from a video file or generate synthetic frames.
    
    Args:
        video_path: Path to video file (optional)
        num_frames: Number of frames to load
        resize: Target frame size
        
    Returns:
        List of numpy arrays (frames)
    """
    frames = []
    
    if video_path and Path(video_path).exists():
        print(f"  Loading frames from: {Path(video_path).name}")
        cap = cv2.VideoCapture(video_path)
        
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
                continue
            
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
        
        cap.release()
    else:
        print("  Using synthetic test frames")
        for _ in range(num_frames):
            # Create realistic-looking test frame with noise
            frame = np.random.randint(50, 200, (*resize[::-1], 3), dtype=np.uint8)
            frames.append(frame)
    
    print(f"  Loaded {len(frames)} frames at {resize[0]}x{resize[1]}")
    return frames


def run_benchmarks(video_path: str = None, models: list = None) -> list:
    """
    Run benchmarks on all specified models.
    
    Args:
        video_path: Path to test video
        models: List of models to test (default: all)
        
    Returns:
        List of benchmark results
    """
    if models is None:
        models = list_available_models()
    
    # Load test frames
    print_separator("Loading Test Frames")
    frames = load_test_frames(video_path, num_frames=100)
    
    # Run benchmarks
    print_separator("Running Benchmarks")
    print(f"  Testing {len(models)} models with {len(frames)} frames each")
    
    results = []
    for model in models:
        result = benchmark_model(model, frames)
        results.append(result)
    
    return results


def print_results(results: list):
    """Print benchmark results as a table."""
    print_separator("Benchmark Results")
    
    # Header
    print(f"\n{'Model':<12} {'FPS':>8} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Load (ms)':>10} {'Status':<10}")
    print("-" * 75)
    
    # Results sorted by FPS
    for r in sorted(results, key=lambda x: x.get('fps', 0), reverse=True):
        if r['success']:
            print(f"{r['model']:<12} {r['fps']:>8.1f} {r['avg_time_ms']:>10.1f} {r['min_time_ms']:>10.1f} {r['max_time_ms']:>10.1f} {r['load_time_ms']:>10.0f} {'✓ OK':<10}")
        else:
            print(f"{r['model']:<12} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'✗ Failed':<10}")
            print(f"             Error: {r['error'][:50]}...")


def generate_markdown_table(results: list, video_info: dict = None) -> str:
    """Generate markdown table for research log."""
    
    lines = []
    lines.append("## Model Speed Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**System:** Windows PC")
    
    if video_info:
        lines.append(f"**Test Video:** {video_info.get('name', 'N/A')} ({video_info.get('resolution', 'N/A')})")
    
    lines.append(f"**Test Frames:** 100 frames at 640x480")
    lines.append("")
    lines.append("| Model | FPS | Avg (ms) | Min (ms) | Max (ms) | Load (ms) | Status |")
    lines.append("|-------|-----|----------|----------|----------|-----------|--------|")
    
    for r in sorted(results, key=lambda x: x.get('fps', 0), reverse=True):
        if r['success']:
            status = "✓"
            lines.append(f"| {r['model']} | {r['fps']:.1f} | {r['avg_time_ms']:.1f} | {r['min_time_ms']:.1f} | {r['max_time_ms']:.1f} | {r['load_time_ms']:.0f} | {status} |")
        else:
            error_short = r['error'][:30] if r['error'] else "Unknown"
            lines.append(f"| {r['model']} | N/A | N/A | N/A | N/A | N/A | ✗ {error_short} |")
    
    lines.append("")
    lines.append("### Recommendations")
    lines.append("")
    
    # Find best performing models
    successful = [r for r in results if r['success']]
    if successful:
        fastest = max(successful, key=lambda x: x['fps'])
        lines.append(f"- **Fastest Model:** {fastest['model']} ({fastest['fps']:.1f} FPS)")
        
        # Recommendations by camera count
        lines.append("")
        lines.append("**Recommended models by camera count:**")
        lines.append("")
        lines.append("| Cameras | Recommended Model | Expected FPS/Camera |")
        lines.append("|---------|-------------------|---------------------|")
        
        for r in sorted(successful, key=lambda x: x['fps'], reverse=True):
            max_cameras = int(r['fps'] / 15)  # Assuming 15 FPS target per camera
            if max_cameras >= 1:
                lines.append(f"| {max_cameras}+ | {r['model']} | {r['fps']/max_cameras:.1f} |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection model speeds")
    parser.add_argument("--video", "-v", help="Path to test video")
    parser.add_argument("--models", "-m", nargs="+", help="Models to test (default: all)")
    parser.add_argument("--output", "-o", help="Output markdown file")
    
    args = parser.parse_args()
    
    print_separator("BIL Security - Model Speed Benchmark")
    
    # Find video
    video_path = args.video
    video_info = None
    
    if not video_path:
        videos = find_videos("data/videos")
        if videos:
            video_path = str(videos[0])
    
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        video_info = {
            "name": Path(video_path).name,
            "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        }
        cap.release()
    
    # Run benchmarks
    results = run_benchmarks(video_path, args.models)
    
    # Print results
    print_results(results)
    
    # Generate markdown
    markdown = generate_markdown_table(results, video_info)
    
    print_separator("Markdown Output")
    print(markdown)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(markdown)
        print(f"\nSaved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
