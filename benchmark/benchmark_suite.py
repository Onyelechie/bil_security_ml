import argparse
import gc
import glob
import os
import sys
import time

import cv2
import pandas as pd
import psutil
import torch

# Constants (Defaults)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.edge_agent.models import YOLOWrapper
from src.edge_agent.models.efficientdet import EfficientDetWrapper
from src.edge_agent.models.ssd import TorchvisionSSDWrapper

VIDEO_EXTENSIONS = ["cctv_samples/*.mp4"]
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "benchmark_results.csv")
OUTPUT_SUMMARY = os.path.join(SCRIPT_DIR, "benchmark_summary.txt")
DEFAULT_WARMUP = 10
DEFAULT_MAX_FRAMES = 100
DEFAULT_THREADS = 4
DEFAULT_CONF = 0.25


def run_benchmark(args):
    """
    Main benchmark execution.
    """
    # 0. Hardware / Reproduction consistency
    print(f"Setting torch threads to {args.threads}")
    torch.set_num_threads(args.threads)

    # 1. Setup Models
    available_models = {
        "YOLOv8-Nano": YOLOWrapper(
            "YOLOv8-Nano", os.path.join(SCRIPT_DIR, "yolov8n.pt"), args.input_size
        ),
        "YOLOv8-Small": YOLOWrapper(
            "YOLOv8-Small", os.path.join(SCRIPT_DIR, "yolov8s.pt"), args.input_size
        ),
        "YOLOv5-Nano": YOLOWrapper(
            "YOLOv5-Nano", os.path.join(SCRIPT_DIR, "yolov5n.pt"), args.input_size
        ),
        "EfficientDet-D0": EfficientDetWrapper("efficientdet_d0", args.input_size),
        "SSD-MobileNet": TorchvisionSSDWrapper("SSD-MobileNet", args.input_size),
    }

    selected_model_names = (
        args.models.split(",")
        if args.models != "all"
        else list(available_models.keys())
    )
    models_to_run = []
    for m_name in selected_model_names:
        if m_name in available_models:
            models_to_run.append(available_models[m_name])
        else:
            print(f"Warning: Model '{m_name}' not recognized. Skipping.")

    if not models_to_run:
        print("Error: No valid models selected.")
        return

    # 2. Find Videos
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(SCRIPT_DIR, ext)))

    if not videos:
        print(f"No videos found in {os.path.join(SCRIPT_DIR, 'cctv_samples')}!")
        return

    print(f"Benchmarking {len(models_to_run)} models on {len(videos)} videos.")

    all_results = []
    process = psutil.Process(os.getpid())
    num_cpus = psutil.cpu_count() or 1

    # 3. Main Loop
    for model_wrapper in models_to_run:
        print(f"\n{'=' * 30}\nBenchmarking: {model_wrapper.name}\n{'=' * 30}")

        try:
            model_wrapper.load()
        except Exception as e:
            print(f"Failed to load {model_wrapper.name}: {e}")
            continue

        for video_path in videos:
            print(f"Processing {video_path}...")
            cap = cv2.VideoCapture(video_path)

            latencies, fps_list, cpu_usages, ram_usages = [], [], [], []
            detection_counts = {"person": 0, "vehicle": 0, "other": 0}
            frame_count = 0

            # WARMUP
            print(f"Warming up ({args.warmup} frames)...")
            for _ in range(args.warmup):
                ret, frame = cap.read()
                if not ret:
                    break
                model_wrapper.predict(frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while cap.isOpened() and frame_count < args.max_frames:
                start_frame_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                detections = model_wrapper.predict(frame)
                end_frame_time = time.time()
                latency_ms = (end_frame_time - start_frame_time) * 1000
                fps = (
                    1.0 / (end_frame_time - start_frame_time)
                    if (end_frame_time - start_frame_time) > 0
                    else 0
                )

                ram_mb = process.memory_info().rss / (1024 * 1024)
                cpu_pct = process.cpu_percent() / num_cpus

                latencies.append(latency_ms)
                fps_list.append(fps)
                ram_usages.append(ram_mb)
                cpu_usages.append(cpu_pct)

                for _, _, _, _, conf, label in detections:
                    if conf < args.confidence:
                        continue
                    label_lower = label.lower()
                    if label_lower == "person":
                        detection_counts["person"] += 1
                    elif any(
                        v in label_lower
                        for v in ["car", "truck", "bus", "motorcycle", "vehicle"]
                    ):
                        detection_counts["vehicle"] += 1
                    else:
                        detection_counts["other"] += 1

                frame_count += 1
                if frame_count % 20 == 0:
                    print(f"Frame {frame_count}/{args.max_frames}...")

            cap.release()

            if latencies:
                all_results.append(
                    {
                        "Model": model_wrapper.name,
                        "Video": os.path.basename(video_path),
                        "Avg_FPS": round(sum(fps_list) / len(fps_list), 2),
                        "Avg_Latency_ms": round(sum(latencies) / len(latencies), 2),
                        "Peak_RAM_MB": round(max(ram_usages), 2),
                        "Avg_CPU_Util": round(sum(cpu_usages) / len(cpu_usages), 2),
                        "Person_Detections": detection_counts["person"],
                        "Vehicle_Detections": detection_counts["vehicle"],
                        "Resolution": (
                            "High"
                            if "HighRes" in video_path
                            else "Low"
                            if "LowRes" in video_path
                            else "Unknown"
                        ),
                    }
                )

        model_wrapper.unload()
        gc.collect()

    # 4. Save Results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV}")

        with open(OUTPUT_SUMMARY, "w") as f:
            f.write("Multi-Model Benchmark Summary\n")
            f.write("=============================\n")
            f.write(
                f"Threads: {args.threads} | Input Size: {args.input_size} | Conf: {args.confidence}\n\n"
            )

            for res in ["High", "Low"]:
                f.write(f"{res.upper()} RESOLUTION SUMMARY\n")
                f.write("-" * 25 + "\n")
                res_df = df[df["Resolution"] == res]
                if not res_df.empty:
                    f.write(res_df.groupby("Model").mean(numeric_only=True).to_string())
                else:
                    f.write(f"No {res}Res videos processed.")
                f.write("\n\n")

            f.write("OVERALL SUMMARY\n")
            f.write("-" * 15 + "\n")
            f.write(df.groupby("Model").mean(numeric_only=True).to_string())

        print(f"Summary saved to {OUTPUT_SUMMARY}")


def create_dummy_video(filename, width=640, height=480, fps=30, duration=5):
    print(f"Generating dummy video {filename}...")
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for _ in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Benchmark Suite")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated models (e.g. YOLOv8-Nano,YOLOv8-Small).",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS, help="Torch threads."
    )
    parser.add_argument(
        "--input-size", type=int, default=640, help="Input resolution (imgsz)."
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup.")
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Max frames."
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONF, help="Conf threshold."
    )

    args = parser.parse_args()
    run_benchmark(args)
