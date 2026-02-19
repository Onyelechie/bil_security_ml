import time
import os
import psutil
import gc
import cv2
import glob
import torch
import pandas as pd
import argparse
from abc import ABC, abstractmethod

# Constants (Defaults)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXTENSIONS = ["cctv_samples/*.mp4"]
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "benchmark_results.csv")
OUTPUT_SUMMARY = os.path.join(SCRIPT_DIR, "benchmark_summary.txt")
DEFAULT_WARMUP = 10
DEFAULT_MAX_FRAMES = 100
DEFAULT_THREADS = 4
DEFAULT_CONF = 0.25

# COCO Class Mapping (Standard IDs)
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


class ModelWrapper(ABC):
    """
    Abstract Base Class for all object detection models.
    """

    def __init__(self, name, input_size=None):
        self.name = name
        self.model = None
        self.input_size = input_size

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        gc.collect()


class YOLOWrapper(ModelWrapper):
    def __init__(self, model_name, weights_path, input_size=640):
        super().__init__(model_name, input_size)
        self.weights_path = weights_path

    def load(self):
        print(f"Loading {self.name} ({self.weights_path})...")
        from ultralytics import YOLO

        if not os.path.exists(self.weights_path):
            weights_name = os.path.basename(self.weights_path)
            print(f"Warning: {self.weights_path} not found.")
            print(f"Attempting to download {weights_name} automatically...")
            self.model = YOLO(weights_name)
            if os.path.exists(weights_name) and not os.path.exists(self.weights_path):
                try:
                    import shutil

                    shutil.move(weights_name, self.weights_path)
                    print(f"Moved downloaded weights to {self.weights_path}")
                except Exception as e:
                    print(f"Note: Could not move weights to {self.weights_path}: {e}")
        else:
            self.model = YOLO(self.weights_path)

    def predict(self, frame):
        # Ultralytics YOLO supports imgsz parameter directly
        results = self.model(frame, verbose=False, imgsz=self.input_size)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                detections.append((label, conf))
        return detections


class EfficientDetWrapper(ModelWrapper):
    def __init__(self, model_name="efficientdet_d0", input_size=512):
        # EfficientDet-D0 has strict architecture constraints (512x512 recommended)
        # We force 512 for D0 to avoid 'stack expects each tensor to be equal size' errors
        if model_name == "efficientdet_d0" and input_size != 512:
            print(
                f"Note: EfficientDet-D0 requires 512x512. Ignoring --input-size {input_size}."
            )
            input_size = 512
        super().__init__("EfficientDet-D0", input_size)
        self.model_name = model_name

    def load(self):
        print(f"Loading {self.name}...")
        from effdet import create_model

        self.model = create_model(
            self.model_name, bench_task="predict", pretrained=True
        )
        self.model.eval()

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = (
            torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        )
        img_tensor = img_tensor / 255.0
        with torch.no_grad():
            output = self.model(img_tensor)
        detections = []
        if output is not None and len(output) > 0:
            for detection in output[0]:
                score = float(detection[4])
                # Note: confidence filter is applied in the main loop
                cls_id = int(detection[5])
                label = COCO_CLASSES.get(cls_id, f"Class_{cls_id}")
                detections.append((label, score))
        return detections


class TorchvisionSSDWrapper(ModelWrapper):
    def __init__(self, name="SSD-MobileNet", input_size=320):
        # ssdlite320_mobilenet_v3_large is hardcoded to 320 in torchvision's default weights
        if input_size != 320:
            print(
                f"Note: SSD-MobileNet (SSDLite320) using native 320x320. Ignoring --input-size {input_size}."
            )
            input_size = 320
        super().__init__(name, input_size)

    def load(self):
        print(f"Loading {self.name}...")
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def predict(self, frame):
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        batch = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(batch)[0]
        detections = []
        for label, score in zip(prediction["labels"], prediction["scores"]):
            cls_id = label.item()
            label_str = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            detections.append((label_str, float(score)))
        return detections


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

                for label, conf in detections:
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
