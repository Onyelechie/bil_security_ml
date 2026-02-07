import time
import os
import psutil
import csv
import gc
import cv2
import glob
import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Configuration
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov']
OUTPUT_CSV = "benchmark_results.csv"
OUTPUT_SUMMARY = "benchmark_summary.txt"
WARMUP_FRAMES = 10

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
    def __init__(self, model_name='efficientdet_d0'):
        super().__init__(f"EfficientDet-D0")
        self.model_name = model_name
        self.config = None

    def load(self):
        print(f"Loading {self.name}...")
        try:
            from effdet import create_model
            # Create model with pretrained weights and 'predict' bench_task for NMS output
            self.model = create_model(self.model_name, bench_task='predict', pretrained=True)
            self.model.eval()
        except ImportError:
            print("Error: 'effdet' library not found. Please install it with 'pip install effdet'")
            raise

    def predict(self, frame):
        # EfficientDet expects specific preprocessing - resize to 512x512
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))  # EfficientDet-D0 expects 512x512
        # Convert to tensor, [C, H, W], float32, batch dim [1, C, H, W]
        img_tensor = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0  

        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Output format for DetBenchPredict is [1, N, 6] (x1, y1, x2, y2, score, class)
        detections = []
        if output is not None and len(output) > 0:
            for detection in output[0]:
                score = float(detection[4])
                if score < 0.25: # Confidence threshold
                    continue
                cls_id = int(detection[5])
                # COCO classes rough mapping could be added here preferably
                detections.append((f"Class_{cls_id}", score))
        return detections

    def unload(self):
        super().unload()
        gc.collect()

def run_benchmark():
    # 1. Setup Models
    models = [
        YOLOWrapper("YOLOv8-Nano", "yolov8n.pt"),
        YOLOWrapper("YOLOv8-Small", "yolov8s.pt"),
        EfficientDetWrapper("efficientdet_d0") 
    ]

    # 2. Find Videos
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(ext))
    
    if not videos:
        print("No videos found! Please place .mp4 files in this directory.")
        # Fallback
        if os.path.exists("sample_video.mp4"):
            videos.append("sample_video.mp4")
            print("Using fallback: sample_video.mp4")

    if not videos:
        print("Creating dummy video for testing...")
        dummy_name = "test_video.mp4"
        create_dummy_video(dummy_name)
        videos.append(dummy_name)

    print(f"Found {len(videos)} videos: {videos}")
    
    all_results = []
    process = psutil.Process(os.getpid())

    # 3. Main Loop
    for model_wrapper in models:
        print(f"\n{'='*30}\nBenchmarking: {model_wrapper.name}\n{'='*30}")
        
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
                if not ret: break
                model_wrapper.predict(frame)
            
            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while cap.isOpened():
                start_frame_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                detections = model_wrapper.predict(frame)
                
                end_frame_time = time.time()
                latency_ms = (end_frame_time - start_frame_time) * 1000
                fps = 1.0 / (end_frame_time - start_frame_time) if (end_frame_time - start_frame_time) > 0 else 0
                
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
                    l = label.lower()
                    if 'person' in l or 'class_1' == l: # Class 1 is usually person in COCO
                        detection_counts['person'] += 1
                    elif any(v in l for v in ['car', 'truck', 'bus', 'motorcycle', 'class_3', 'class_6', 'class_8']):
                        detection_counts['vehicle'] += 1
                    else:
                        detection_counts['other'] += 1
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Frame {frame_count}: FPS={fps:.1f}, Lat={latency_ms:.1f}ms, RAM={ram_mb:.1f}MB")

            cap.release()
            
            # Aggregate Results for this Video/Model
            if latencies:
                avg_fps = sum(fps_list) / len(fps_list)
                avg_lat = sum(latencies) / len(latencies)
                peak_ram = max(ram_usages)
                avg_cpu = sum(cpu_usages) / len(cpu_usages)
                
                all_results.append({
                    "Model": model_wrapper.name,
                    "Video": video_path,
                    "Avg_FPS": round(avg_fps, 2),
                    "Avg_Latency_ms": round(avg_lat, 2),
                    "Peak_RAM_MB": round(peak_ram, 2),
                    "Avg_CPU_Util": round(avg_cpu, 2),
                    "Person_Count": detection_counts['person'],
                    "Vehicle_Count": detection_counts['vehicle']
                })

        # Explicitly unload
        print(f"Unloading {model_wrapper.name}...")
        model_wrapper.unload()
        torch.cuda.empty_cache() # No-op on CPU but safety habit
        gc.collect()
        time.sleep(2) 

    # 4. Save Results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV}")

        # Summary Text
        with open(OUTPUT_SUMMARY, "w") as f:
            f.write("Multi-Model Benchmark Summary\n")
            f.write("=============================\n\n")
            f.write(df.to_string(index=False))
        print(f"Summary saved to {OUTPUT_SUMMARY}")
    else:
        print("No results generated.")

def create_dummy_video(filename, width=640, height=480, fps=30, duration=5):
    print(f"Generating dummy video {filename}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for _ in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8) 
        out.write(frame)
    out.release()

if __name__ == "__main__":
    import numpy as np # Ensure numpy is available for dummy video
    run_benchmark()
