# Benchmark Suite Documentation

This suite evaluates the performance and detection accuracy of various computer vision models on surveillance video data.

## Purpose

The tool is used to compare object detection models (YOLO, EfficientDet, SSD) in terms of:

- **Inference Speed**: Accuracy vs. Latency trade-offs.
- **Resource Usage**: CPU and RAM footprints on constrained hardware.
- **Detection Density**: How frequently people and vehicles are detected across video samples.

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

> [!NOTE]
> Detection counts represent the total number of bounding boxes found by the model and are NOT a count of unique individuals or vehicles.
