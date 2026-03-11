# Multi-Model Benchmark Report (CPU-Only Evaluation)

## 1. Methodology & Reproduction

- **Run Count:** Single run per model/video combination.
- **Frames Evaluated:** Full video duration (processed every frame after 10 warmup frames).
- **Averaging:** Results are averaged across all frames for that specific execution. No multiple-pass variance is included in this run.
- **Reproduction Command:** See the [How to Reproduce Exactly](README.md#how-to-reproduce-exactly) section in the README for the exact CLI command and parameter documentation.
- **Dataset:** 12 CCTV clips (6 models x 2 resolutions). Download links available in [README.md](README.md#dataset-cctv-samples).

### Understanding Detection Metrics

- **Accumulated Detections**: The values for "Person Detections" and "Vehicle Detections" are a **sum of all detections across every frame** of the video.
- **Is Higher Better?**: Generally, **YES**.
  - A **higher** count across the same video indicates the model is more **sensitive** (it sees the object) and more **consistent** (it doesn't "flicker" or lose the object frame-by-frame).
  - A **lower** count on the same video means the model is "blind" in many frames or is failing to detect objects that other models are seeing.

## 2. Environment Details

- **Operating System:** Windows 11 Pro (Build 26200)
- **CPU:** Intel Core i7-7700K @ 4.20GHz (Intel64 Family 6 Model 158 Stepping 13)
- **RAM:** 16.22 GB
- **Device:** CPU (CUDA Unavailable)  
- **torch:** 2.10.0+cpu  
- **torchvision:** 0.25.0+cpu  
- **ultralytics:** 8.4.12  
- **OpenCV (cv2):** 4.13.0  

---

## 2. Benchmark Configuration

- **Warmup Frames:** 10  
- **Max Frames Per Video:** 0 (All frames)
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
| EfficientDet-D0 | 4.48 | 226.04 | 726.89 | 81.51 | 170.17 | 33.00 |
| SSD-MobileNet | 7.24 | 140.10 | 794.50 | 79.88 | 160.83 | 66.33 |
| YOLOv5-Nano | 19.91 | 50.95 | 580.66 | 80.34 | 69.50 | 253.00 |
| YOLOv8-Nano | 19.67 | 51.20 | 540.84 | 81.32 | 47.33 | 309.33 |
| YOLOv8-Small | 10.31 | 98.03 | 615.49 | 80.00 | 53.83 | 429.17 |

---

## 4. Low Resolution Summary

| Model | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU (%) | Person Detections | Vehicle Detections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientDet-D0 | 4.75 | 213.84 | 611.51 | 81.05 | 170.50 | 3.50 |
| SSD-MobileNet | 13.28 | 76.44 | 676.42 | 80.15 | 42.50 | 58.83 |
| YOLOv5-Nano | 23.01 | 44.53 | 463.16 | 81.09 | 64.67 | 247.33 |
| YOLOv8-Nano | 21.80 | 46.78 | 422.32 | 80.27 | 46.17 | 232.17 |
| YOLOv8-Small | 10.44 | 97.72 | 498.95 | 80.21 | 59.67 | 389.83 |

---

## 5. Overall Summary

| Model | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU (%) | Person Detections | Vehicle Detections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientDet-D0 | 4.62 | 219.94 | 669.20 | 81.28 | 170.33 | 18.25 |
| SSD-MobileNet | 10.26 | 108.27 | 735.46 | 80.02 | 101.67 | 62.58 |
| YOLOv5-Nano | 21.46 | 47.74 | 521.91 | 80.72 | 67.08 | 250.17 |
| YOLOv8-Nano | 20.74 | 48.99 | 481.58 | 80.79 | 46.75 | 270.75 |
| YOLOv8-Small | 10.37 | 97.88 | 557.22 | 80.10 | 56.75 | 409.50 |

---

## 6. Performance Analysis

###  Fastest Model

**YOLOv5-Nano** achieved the highest overall FPS (~21.5 FPS) with the lowest latency (~47.7 ms) on this Windows 11 PC.

###  Lowest Latency

YOLOv5-Nano and YOLOv8-Nano delivered the lowest latency (~47-49 ms).

###  Most Memory Efficient

YOLOv8-Nano consumed the least RAM overall (~481.6 MB average).

###  Slowest Model

EfficientDet-D0 remained the slowest model (~4.6 FPS) with high latency (~220 ms).

###  Highest Vehicle Detection

YOLOv8-Small detected the highest number of vehicles (~409.5 average across the whole video), showing stronger detection density but at roughly half the speed of the Nano models.

### Person Detection Performance

EfficientDet-D0 produced the highest person detection counts (170.3), though at significant performance cost.

---

## 7. Key Observations

1. CPU utilization remained consistently high (~80–82%) across all models with 4 threads.
2. Low-resolution processing significantly improves FPS for models like SSD-MobileNet (from 7 to 13 FPS) and YOLO models.
3. YOLOv5-Nano provides the best overall speed and latency on this system.
4. YOLOv8-Small offers significantly higher vehicle detection density (nearly double YOLOv5-Nano) but with an FPS drop from 21 to 10.
5. EfficientDet-D0 is not suitable for real-time CPU-only inference.
6. SSD-MobileNet offers moderate performance (10 FPS overall) but lacks the detection density of YOLO models.

---

## 8. Deployment Recommendation (CPU-Based System)

For CPU-only real-time deployment:

- ** Best Overall Choice: YOLOv5-Nano**
  - Highest FPS (21.5)
  - Lowest latency (47.7 ms)
  - Low RAM usage (522 MB)
  - Excellent speed/accuracy balance

- ** Budget / Lightweight Option: YOLOv8-Nano**
  - Slightly lower FPS but lowest RAM footprint
  - Strong alternative to YOLOv5-Nano

- ** Not Recommended for Real-Time CPU**
  - EfficientDet-D0

---

## 9. Conclusion

On a CPU-only Windows 11 system (no CUDA), YOLO-based architectures significantly outperform EfficientDet and SSD-MobileNet in both speed and efficiency.

For real-time surveillance or edge inference deployment on CPU systems, YOLOv8-Nano provides the optimal balance between detection accuracy and performance.

---

End of Report
