# Multi-Model Benchmark Report (CPU-Only Evaluation)

## 1. Methodology & Reproduction

- **Run Count:** Single run per model/video combination.
- **Frames Evaluated:** 100 frames per video (after 10 warmup frames).
- **Averaging:** Results are averaged across all frames for that specific execution. No multiple-pass variance is included in this run.
- **Reproduction Command:** See the [How to Reproduce Exactly](README.md#how-to-reproduce-exactly) section in the README for the exact CLI command and parameter documentation.
- **Dataset:** 12 CCTV clips (6 models x 2 resolutions). Download links available in [README.md](README.md#dataset-cctv-samples).

## 2. Environment Details

- **Operating System:** Windows 11  
- **Python Version:** 3.13.12  
- **CPU:** Intel64 Family 6 Model 140 Stepping 1 (GenuineIntel)  
- **RAM:** 15.79 GB  
- **Device:** CPU (CUDA Unavailable)  
- **torch:** 2.10.0+cpu  
- **torchvision:** 0.25.0+cpu  
- **ultralytics:** 8.4.12  
- **OpenCV (cv2):** 4.13.0  

---

## 2. Benchmark Configuration

- **Warmup Frames:** 10  
- **Max Frames Per Video:** 100  
- **Confidence Threshold:** 0.25  
- **Torch Threads:** 4  
- **Input Sizes:**
  - YOLO models → 640
  - EfficientDet → 512
  - SSD-MobileNet → 320  

---

## 3. High Resolution Summary

| Model | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU (%) | Person Detections | Vehicle Detections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientDet-D0 | 3.07 | 350.13 | 666.20 | 81.18 | 41.50 | 13.00 |
| SSD-MobileNet | 4.83 | 226.20 | 805.01 | 83.79 | 31.17 | 20.00 |
| YOLOv5-Nano | 13.14 | 91.74 | 545.05 | 82.46 | 10.33 | 57.00 |
| YOLOv8-Nano | 16.02 | 67.67 | 517.59 | 84.91 | 9.17 | 89.50 |
| YOLOv8-Small | 7.20 | 140.53 | 592.10 | 84.09 | 11.00 | 138.17 |

---

## 4. Low Resolution Summary

| Model | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU (%) | Person Detections | Vehicle Detections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientDet-D0 | 3.45 | 307.86 | 554.39 | 82.02 | 31.83 | 2.00 |
| SSD-MobileNet | 8.41 | 133.97 | 680.34 | 82.74 | 11.50 | 21.00 |
| YOLOv5-Nano | 16.07 | 79.28 | 424.02 | 81.99 | 12.83 | 70.17 |
| YOLOv8-Nano | 16.77 | 69.51 | 400.47 | 81.88 | 12.00 | 64.50 |
| YOLOv8-Small | 6.82 | 152.99 | 472.91 | 82.98 | 15.17 | 116.50 |

---

## 5. Overall Summary

| Model | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU (%) | Person Detections | Vehicle Detections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientDet-D0 | 3.26 | 328.99 | 610.29 | 81.60 | 36.67 | 7.50 |
| SSD-MobileNet | 6.62 | 180.09 | 742.67 | 83.26 | 21.33 | 20.50 |
| YOLOv5-Nano | 14.60 | 85.51 | 484.53 | 82.22 | 11.58 | 63.58 |
| YOLOv8-Nano | 16.40 | 68.59 | 459.03 | 83.40 | 10.58 | 77.00 |
| YOLOv8-Small | 7.01 | 146.76 | 532.50 | 83.53 | 13.08 | 127.33 |

---

## 6. Performance Analysis

### 🔥 Fastest Model

**YOLOv8-Nano** achieved the highest overall FPS (~16.4 FPS) with the lowest latency (~68.6 ms).

### ⚡ Lowest Latency

YOLOv8-Nano delivered the lowest latency across resolutions.

### 💾 Most Memory Efficient

YOLOv8-Nano consumed the least RAM overall (~459 MB average).

### 🐢 Slowest Model

EfficientDet-D0 was the slowest model (~3.26 FPS) with very high latency (~329 ms).

### 🎯 Highest Vehicle Detection

YOLOv8-Small detected the highest number of vehicles (~127 average), indicating stronger detection capacity but at lower speed.

### 👤 Person Detection Performance

EfficientDet-D0 produced the highest person detection counts, though at significant performance cost.

---

## 7. Key Observations

1. CPU utilization remained consistently high (~81–85%) across all models.
2. Lower resolution significantly improves FPS for most models.
3. YOLOv8-Nano provides the best trade-off between speed, latency, and memory.
4. YOLOv8-Small improves detection quality but halves inference speed.
5. EfficientDet-D0 is not suitable for real-time CPU inference.
6. SSD-MobileNet offers moderate performance but is less efficient than YOLO models.

---

## 8. Deployment Recommendation (CPU-Based System)

For CPU-only real-time deployment:

- **✅ Best Overall Choice: YOLOv8-Nano**
  - Highest FPS
  - Lowest latency
  - Lowest RAM usage
  - Balanced detection performance

- **✅ Budget / Lightweight Option: YOLOv5-Nano**
  - Slightly slower but still efficient

- **❌ Not Recommended for Real-Time CPU**
  - EfficientDet-D0

---

## 9. Conclusion

On a CPU-only Windows 11 system (no CUDA), YOLO-based architectures significantly outperform EfficientDet and SSD-MobileNet in both speed and efficiency.

For real-time surveillance or edge inference deployment on CPU systems, YOLOv8-Nano provides the optimal balance between detection accuracy and performance.

---

End of Report
