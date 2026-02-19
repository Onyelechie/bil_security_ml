# Benchmark Suite Documentation

This suite evaluates the performance and detection accuracy of various computer vision models on surveillance video data.

## Purpose

The tool is used to compare object detection models (YOLO, EfficientDet, SSD) in terms of:

- **Inference Speed**: Accuracy vs. Latency trade-offs.
- **Resource Usage**: CPU and RAM footprints on constrained hardware.
- **Detection Density**: How frequently people and vehicles are detected across video samples.

## Dataset: CCTV Samples

The benchmark uses a curated set of real-world CCTV clips. These are not included in the repository to keep the size small.

| Filename | Type | Resolution | Download Link |
| :--- | :--- | :--- | :--- |
| C1HighRes - Human.mp4 | Human | High | [Download](https://drive.google.com/file/d/1rUlnJr5g4Tj6WsLfKStRAJq2_P96b6Uj/view?usp=drive_link) |
| C1LowRes - Human.mp4 | Human | Low | [Download](https://drive.google.com/file/d/1UjZv3yxt-28pmyIy2W9N4TAfLYpRZqqQ/view?usp=drive_link) |
| C2HighRes - Car.mp4 | Vehicle | High | [Download](https://drive.google.com/file/d/1EGluAF3Y6q_H4Kg7ZXsOBxAG6YJoio_z/view?usp=drive_link) |
| C2LowRes - Car.mp4 | Vehicle | Low | [Download](https://drive.google.com/file/d/1IKayZA9K4UqCOqkSqvjJW8TONXUY0Qig/view?usp=drive_link) |
| C3HighRes - Car.mp4 | Vehicle | High | [Download](https://drive.google.com/file/d/1XPW5fHKhgmTo2H_ScXC1RYGqePXw7QhT/view?usp=drive_link) |
| C3LowRes - Car.mp4 | Vehicle | Low | [Download](https://drive.google.com/file/d/1AmEw-l0qa6HBisYxm6YfN5D-a-JsvAfm/view?usp=drive_link) |
| C4HighRes - Human.mp4 | Human | High | [Download](https://drive.google.com/file/d/1jtIhRHc8no55JAmf2wltdE65lArCHB3t/view?usp=drive_link) |
| C4LowRes - Human.mp4 | Human | Low | [Download](https://drive.google.com/file/d/1Z01n9zQ7BMXG_FOG-wQmG4QTXctkS9vD/view?usp=drive_link) |
| C5HighResPTZ - Car.mp4 | Vehicle (PTZ) | High | [Download](https://drive.google.com/file/d/1ev5oXKDYHac0FiUDpICQcAWGXUhXPWpg/view?usp=drive_link) |
| C5LowResPTZ - Car.mp4 | Vehicle (PTZ) | Low | [Download](https://drive.google.com/file/d/11pgoDhvAXxmXlA2qOkPfjPJPk7bwmvX0/view?usp=drive_link) |
| C6HighResPTZ - Human.mp4 | Human (PTZ) | High | [Download](https://drive.google.com/file/d/1hCTfCLiUpW2sN48vBWb6QmmtlAJ94t_I/view?usp=drive_link) |
| C6LowResPTZ - Human.mp4 | Human (PTZ) | Low | [Download](https://drive.google.com/file/d/11kZOHq86kuju9qr9h18vvJBWGPMcNeOu/view?usp=drive_link) |

## Supported Models

| Model Identifier | Source | Weights File | Input Size |
| :--- | :--- | :--- | :--- |
| YOLOv8-Nano | Ultralytics | `yolov8n.pt` | 640 |
| YOLOv8-Small | Ultralytics | `yolov8s.pt` | 640 |
| YOLOv5-Nano | Ultralytics | `yolov5n.pt` | 640 |
| EfficientDet-D0 | [effdet](https://github.com/rwightman/efficientdet-pytorch) | Pretrained | 512 |
| SSD-MobileNet | [torchvision](https://pytorch.org/vision/stable/models.html) | `SSDLite320_MobileNet_V3_Large` | 320 |

## How to Run

### 1. Prerequisites

Ensure you have the virtual environment activated and dependencies installed.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Samples

If you don't have video clips, run the setup script to download real CCTV clips:

```bash
python3 benchmark/setup_samples.py
```

### 3. Run Benchmark

Execute the main suite:

```bash
python3 benchmark/benchmark_suite.py
```

### 4. Run Smoke Test (Fast Check)

To verify the benchmark pipeline without downloading heavy models or video files:

```bash
export PYTHONPATH="$PWD/src:$PWD" && .venv/bin/python3 -m pytest tests/test_benchmark_smoke.py
```

## Output Files

The suite generates two main files in the `benchmark/` directory:

1. **`benchmark_results.csv`**: Raw data with metrics for every model/video combination.
2. **`benchmark_summary.txt`**: A human-readable summary averaging results by model and video resolution (HighRes vs. LowRes).

## Metrics Explained

- **Avg_FPS**: Average frames per second processed by the model.
- **Avg_Latency_ms**: Time taken to process a single frame in milliseconds.
- **Peak_RAM_MB**: Maximum memory used by the process during model execution.
- **Avg_CPU_Util**: Average CPU utilization percentage (normalized across all cores).
- **Person_Detections**: Total count of all person detections across all frames.
- **Vehicle_Detections**: Total count of all vehicle (car, truck, bus, bike) detections across all frames.

## CSV Schema & Units

The `benchmark_results.csv` file contains the following columns:

| Column | Description | Units / Notes |
| :--- | :--- | :--- |
| `Model` | The name of the object detection model. | String |
| `Video` | The name of the video file processed. | String |
| `Avg_FPS` | Average frames per second. | Higher is better |
| `Avg_Latency_ms` | Average inference time per frame. | Milliseconds (Lower is better) |
| `Peak_RAM_MB` | Maximum resident set size (RSS) memory. | Megabytes (MB) |
| `Avg_CPU_Util` | Average CPU utilization. | Percentage (Normalized per core) |
| `Person_Detections` | Cumulative person bounding boxes. | Density measure |
| `Vehicle_Detections` | Cumulative vehicle bounding boxes. | Density measure |
| `Resolution` | Category based on video filename. | High/Low |

### Summary Derivation

The tables in `benchmark_summary.txt` and `benchmark_report.md` are derived by grouping the raw CSV data by `Model` and `Resolution`, then calculating the mean for all numeric metrics.

## Artifact Policy

To balance version control with performance variability, we follow this policy:

1. **Baseline Snapshots**: `benchmark_report.md` is a tracked file representing a "gold standard" for a specific environment (e.g., `benchmark_report_win11_cpu.md`). If adding results for a new hardware tier, create a new named file.
2. **Generated Outputs**: `benchmark_results.csv` and `benchmark_summary.txt` are ephemeral. They are ignored by Git and should be stored as CI/CD artifacts or shared manually.

> [!TIP]
> Always rename your `benchmark_report.md` if you intend to commit it as a new baseline for a different machine.
>
> [!NOTE]
> Detection counts represent the total number of bounding boxes found by the model and are NOT a count of unique individuals or vehicles.
