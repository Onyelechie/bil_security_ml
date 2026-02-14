# Multi-Model Benchmark Report

## Overview

This report presents benchmarking results for five object detection models evaluated on High Resolution and Low Resolution video datasets.

Models evaluated:

- EfficientDet-D0
- SSD-MobileNet
- YOLOv5-Nano
- YOLOv8-Nano
- YOLOv8-Small

Metrics collected:

- Average FPS
- Average Latency (ms)
- Peak RAM Usage (MB)
- Average CPU Utilization (%)
- Person Count
- Vehicle Count

---

# 1. High Resolution Summary (Model-wise Averages)

| Model            | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU Util (%) | Person Count | Vehicle Count |
|------------------|---------|------------------|---------------|------------------|--------------|---------------|
| EfficientDet-D0  | 3.70    | 270.47           | 805.53        | 31.03            | 40.00       | 21.67        |
| SSD-MobileNet    | 8.33    | 120.80           | 844.30        | 20.65            | 48.33       | 0.33         |
| YOLOv5-Nano      | 30.99   | 32.68            | 692.20        | 8.33             | 16.67       | 99.67        |
| YOLOv8-Nano      | 29.99   | 33.99            | 579.62        | 11.33            | 18.67       | 108.67       |
| YOLOv8-Small     | 17.08   | 59.23            | 677.80        | 15.05            | 20.67       | 136.00       |

---

# 2. Low Resolution Summary (Model-wise Averages)

| Model            | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU Util (%) | Person Count | Vehicle Count |
|------------------|---------|------------------|---------------|------------------|--------------|---------------|
| EfficientDet-D0  | 3.74    | 267.73           | 777.91        | 31.33            | 34.33       | 1.33         |
| SSD-MobileNet    | 9.57    | 104.58           | 824.47        | 20.21            | 21.67       | 6.00         |
| YOLOv5-Nano      | 29.61   | 34.37            | 690.88        | 7.12             | 17.00       | 96.00        |
| YOLOv8-Nano      | 28.80   | 35.45            | 590.78        | 9.83             | 15.00       | 57.00        |
| YOLOv8-Small     | 15.72   | 65.24            | 676.84        | 16.14            | 16.67       | 136.33       |

---

# 3. Overall Summary (Model-wise Averages)

| Model            | Avg FPS | Avg Latency (ms) | Peak RAM (MB) | Avg CPU Util (%) | Person Count | Vehicle Count |
|------------------|---------|------------------|---------------|------------------|--------------|---------------|
| EfficientDet-D0  | 3.72    | 269.10           | 791.72        | 31.18            | 37.17       | 11.50        |
| SSD-MobileNet    | 8.95    | 112.69           | 834.39        | 20.43            | 35.00       | 3.17         |
| YOLOv5-Nano      | 30.30   | 33.52            | 691.54        | 7.73             | 16.83       | 97.83        |
| YOLOv8-Nano      | 29.40   | 34.72            | 585.20        | 10.58            | 16.83       | 82.83        |
| YOLOv8-Small     | 16.40   | 62.24            | 677.32        | 15.59            | 18.67       | 136.17       |

---

# 4. Performance Analysis

## 🚀 Fastest Model
YOLOv5-Nano achieved the highest overall FPS (~30 FPS) with very low latency (~33 ms).

## ⚡ Lowest Latency
YOLOv5-Nano and YOLOv8-Nano both maintained ~33–35 ms latency.

## 🧠 Lowest RAM Usage
YOLOv8-Nano used the least RAM (~585 MB average).

## 🔥 Highest CPU Usage
EfficientDet-D0 consumed the most CPU (~31%).

## 🎯 Highest Vehicle Detection
YOLOv8-Small detected the highest number of vehicles overall (~136).

## 🐢 Slowest Model
EfficientDet-D0 performed the slowest (~3.7 FPS) with very high latency (~269 ms).

---

# 5. Key Observations

- YOLOv5-Nano offers the best speed-to-resource ratio.
- YOLOv8-Nano provides slightly better memory efficiency.
- YOLOv8-Small improves detection counts but sacrifices speed.
- EfficientDet-D0 is not suitable for real-time CPU-based inference.
- SSD-MobileNet provides moderate performance but weaker vehicle detection.

---

# 6. Recommendation

For real-time CPU-based on-premise deployment:

**Best Overall Choice:** YOLOv5-Nano  
**Best Memory Efficient Choice:** YOLOv8-Nano  
**Best Detection Quality (Vehicles):** YOLOv8-Small  

---

# End of Report
