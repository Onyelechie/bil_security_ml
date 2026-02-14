import time
import os
import psutil
import gc
import cv2
import glob
import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXTENSIONS = ["cctv_samples/*.mp4"]
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "benchmark_results.csv")
OUTPUT_SUMMARY = os.path.join(SCRIPT_DIR, "benchmark_summary.txt")
WARMUP_FRAMES = 10
MAX_FRAMES_PER_VIDEO = 100


class ModelWrapper(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, frame):
        """Returns list of detections (class_name, confidence, box)"""
        pass

    @abstractmethod
    def unload(self):
        if self.model:
            del self.model
            self.model = None
        gc.collect()


class YOLOWrapper(ModelWrapper):
    def __init__(self, model_name, weights_path):
        super().__init__(model_name)
        self.weights_path = weights_path

    def load(self):
        print(f"Loading {self.name} ({self.weights_path})...")
        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)

    def predict(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                detections.append((label, conf))
        return detections

    def unload(self):
        super().unload()
        # Ultralytics sometimes leaves things behind, force more cleanup if possible
        gc.collect()


class EfficientDetWrapper(ModelWrapper):
    def __init__(self, model_name="efficientdet_d0"):
        super().__init__("EfficientDet-D0")
        self.model_name = model_name
        self.config = None

    def load(self):
        print(f"Loading {self.name}...")
        try:
            from effdet import create_model  # type: ignore

            # Create model with pretrained weights and 'predict' bench_task for NMS output
            self.model = create_model(
                self.model_name, bench_task="predict", pretrained=True
            )
            self.model.eval()
        except ImportError:
            print(
                "Error: 'effdet' library not found. Please install it with 'pip install effdet'"
            )
            raise

    def predict(self, frame):
        # EfficientDet expects specific preprocessing - resize to 512x512
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))  # EfficientDet-D0 expects 512x512
        # Convert to tensor, [C, H, W], float32, batch dim [1, C, H, W]
        img_tensor = (
            torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        )
        img_tensor = img_tensor / 255.0

        with torch.no_grad():
            output = self.model(img_tensor)

        # Output format for DetBenchPredict is [1, N, 6] (x1, y1, x2, y2, score, class)
        detections = []
        if output is not None and len(output) > 0:
            for detection in output[0]:
                score = float(detection[4])
                if score < 0.25:  # Confidence threshold
                    continue
                cls_id = int(detection[5])
                # COCO classes rough mapping could be added here preferably
                detections.append((f"Class_{cls_id}", score))
        return detections

    def unload(self):
        super().unload()
        gc.collect()


class TorchvisionSSDWrapper(ModelWrapper):
    def __init__(self, name="SSD-MobileNet"):
        super().__init__(name)

    def load(self):
        print(f"Loading {self.name}...")
        from torchvision.models.detection import (  # type: ignore
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
            if score > 0.25:
                # Basic mapping for COCO: torchvision labels are 1-based index
                detections.append((f"class_{label.item()}", float(score)))
        return detections

    def unload(self):
        super().unload()
        gc.collect()


def run_benchmark():
    # 1. Setup Models
    models = [
        YOLOWrapper("YOLOv8-Nano", os.path.join(SCRIPT_DIR, "yolov8n.pt")),
        YOLOWrapper("YOLOv8-Small", os.path.join(SCRIPT_DIR, "yolov8s.pt")),
        YOLOWrapper("YOLOv5-Nano", os.path.join(SCRIPT_DIR, "yolov5n.pt")),
        EfficientDetWrapper("efficientdet_d0"),
        TorchvisionSSDWrapper("SSD-MobileNet"),
    ]

    # 2. Find Videos
    videos = []
    for ext in VIDEO_EXTENSIONS:
        search_path = os.path.join(SCRIPT_DIR, ext)
        videos.extend(glob.glob(search_path))

    if not videos:
        print(f"No videos found in {os.path.join(SCRIPT_DIR, 'cctv_samples')}!")
        # Fallback
        fallback_path = os.path.join(SCRIPT_DIR, "../sample_video.mp4")
        if os.path.exists(fallback_path):
            videos.append(fallback_path)
            print(f"Using fallback: {fallback_path}")

    if not videos:
        print("Creating dummy video for testing...")
        dummy_name = os.path.join(SCRIPT_DIR, "test_video.mp4")
        create_dummy_video(dummy_name)
        videos.append(dummy_name)

    print(f"Found {len(videos)} videos: {videos}")

    all_results = []
    process = psutil.Process(os.getpid())

    # 3. Main Loop
    for model_wrapper in models:
        print(f"\n{'=' * 30}\nBenchmarking: {model_wrapper.name}\n{'=' * 30}")

        try:
            model_wrapper.load()
        except Exception as e:
            print(f"Failed to load {model_wrapper.name}: {e}")
            continue

        for video_path in videos:
            print(f"Processing {video_path}...")
            cap = cv2.VideoCapture(video_path)

            # Metrics Storage
            latencies = []
            fps_list = []
            cpu_usages = []
            ram_usages = []
            detection_counts = {"person": 0, "vehicle": 0, "other": 0}

            frame_count = 0

            # WARMUP
            print("Warming up (10 frames)...")
            for _ in range(WARMUP_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
                model_wrapper.predict(frame)

            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while cap.isOpened():
                if frame_count >= MAX_FRAMES_PER_VIDEO:
                    break

                start_frame_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                detections = model_wrapper.predict(frame)

                end_frame_time = time.time()
                latency_ms = (end_frame_time - start_frame_time) * 1000
                fps = (
                    1.0 / (end_frame_time - start_frame_time)
                    if (end_frame_time - start_frame_time) > 0
                    else 0
                )

                # System Metrics
                ram_mb = process.memory_info().rss / (1024 * 1024)
                cpu_pct = psutil.cpu_percent()

                # Log Data
                latencies.append(latency_ms)
                fps_list.append(fps)
                ram_usages.append(ram_mb)
                cpu_usages.append(cpu_pct)

                # Count classes
                for label, conf in detections:
                    label_lower = label.lower()
                    if (
                        "person" in label_lower or "class_1" == label_lower
                    ):  # Class 1 is usually person in COCO
                        detection_counts["person"] += 1
                    elif any(
                        v in label_lower
                        for v in [
                            "car",
                            "truck",
                            "bus",
                            "motorcycle",
                            "class_3",
                            "class_6",
                            "class_8",
                        ]
                    ):
                        detection_counts["vehicle"] += 1
                    else:
                        detection_counts["other"] += 1

                frame_count += 1
                if frame_count % 10 == 0:
                    print(
                        f"Frame {frame_count}: FPS={fps:.1f}, Lat={latency_ms:.1f}ms, RAM={ram_mb:.1f}MB"
                    )

            cap.release()

            # Aggregate Results for this Video/Model
            if latencies:
                avg_fps = sum(fps_list) / len(fps_list)
                avg_lat = sum(latencies) / len(latencies)
                peak_ram = max(ram_usages)
                avg_cpu = sum(cpu_usages) / len(cpu_usages)

                all_results.append(
                    {
                        "Model": model_wrapper.name,
                        "Video": os.path.basename(video_path),
                        "Avg_FPS": round(avg_fps, 2),
                        "Avg_Latency_ms": round(avg_lat, 2),
                        "Peak_RAM_MB": round(peak_ram, 2),
                        "Avg_CPU_Util": round(avg_cpu, 2),
                        "Person_Count": detection_counts["person"],
                        "Vehicle_Count": detection_counts["vehicle"],
                    }
                )

        # Explicitly unload
        print(f"Unloading {model_wrapper.name}...")
        model_wrapper.unload()
        torch.cuda.empty_cache()  # No-op on CPU but safety habit
        gc.collect()
        time.sleep(2)

    # 4. Save Results
    if all_results:
        df = pd.DataFrame(all_results)

        # Categorize by resolution
        df["Resolution"] = df["Video"].apply(
            lambda x: (
                "High" if "HighRes" in x else "Low" if "LowRes" in x else "Unknown"
            )
        )

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV}")

        # Generate Summaries
        with open(OUTPUT_SUMMARY, "w") as f:
            f.write("Multi-Model Benchmark Summary\n")
            f.write("=============================\n\n")

            f.write("1. HIGH RESOLUTION SUMMARY (Model-wise Averages)\n")
            f.write("----------------------------------------------\n")
            high_df = df[df["Resolution"] == "High"]
            if not high_df.empty:
                high_summary = high_df.groupby("Model").mean(numeric_only=True)
                f.write(high_summary.to_string())
            else:
                f.write("No HighRes videos processed.")
            f.write("\n\n")

            f.write("2. LOW RESOLUTION SUMMARY (Model-wise Averages)\n")
            f.write("---------------------------------------------\n")
            low_df = df[df["Resolution"] == "Low"]
            if not low_df.empty:
                low_summary = low_df.groupby("Model").mean(numeric_only=True)
                f.write(low_summary.to_string())
            else:
                f.write("No LowRes videos processed.")
            f.write("\n\n")

            f.write("3. OVERALL SUMMARY (Model-wise Averages)\n")
            f.write("--------------------------------------\n")
            overall_summary = df.groupby("Model").mean(numeric_only=True)
            f.write(overall_summary.to_string())
            f.write("\n\n")

            f.write("4. FULL RAW RESULTS\n")
            f.write("------------------\n")
            f.write(df.to_string(index=False))

        print(f"Summary saved to {OUTPUT_SUMMARY}")
    else:
        print("No results generated.")


def create_dummy_video(filename, width=640, height=480, fps=30, duration=5):
    print(f"Generating dummy video {filename}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for _ in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    import numpy as np  # Ensure numpy is available for dummy video

    run_benchmark()
