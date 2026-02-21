import os
import pandas as pd
from benchmark.benchmark_suite import run_benchmark, create_dummy_video


def test_benchmark_smoke(tmp_path, monkeypatch):
    """
    Smoke test to ensure the benchmark pipeline runs and produces output.
    Uses a dummy video and mocks models to avoid heavy downloads/inference.
    """
    # 1. Setup mock environment
    dummy_video = tmp_path / "test_video.mp4"
    create_dummy_video(str(dummy_video), duration=1)

    output_csv = tmp_path / "test_results.csv"
    output_sum = tmp_path / "test_summary.txt"
    monkeypatch.setattr("benchmark.benchmark_suite.OUTPUT_CSV", str(output_csv))
    monkeypatch.setattr("benchmark.benchmark_suite.OUTPUT_SUMMARY", str(output_sum))

    # 2. Mock model list and arguments
    from benchmark.benchmark_suite import ModelWrapper

    class MockWrapper(ModelWrapper):
        def load(self):
            pass

        def predict(self, frame):
            return [("person", 0.9)]

    class MockArgs:
        def __init__(self):
            self.models = "all"
            self.threads = 1
            self.input_size = 640
            self.warmup = 1
            self.max_frames = 5
            self.confidence = 0.25

    # We need to monkeypatch the search logic
    def mock_glob(path):
        return [str(dummy_video)]

    monkeypatch.setattr("glob.glob", mock_glob)
    monkeypatch.setattr("benchmark.benchmark_suite.VIDEO_EXTENSIONS", ["*.mp4"])

    # Use args[0] to capture the actual model name (e.g., YOLOv8-Nano)
    monkeypatch.setattr(
        "benchmark.benchmark_suite.YOLOWrapper", lambda *args: MockWrapper(args[0])
    )
    monkeypatch.setattr(
        "benchmark.benchmark_suite.EfficientDetWrapper",
        lambda *args: MockWrapper("MockEffDet"),
    )
    monkeypatch.setattr(
        "benchmark.benchmark_suite.TorchvisionSSDWrapper",
        lambda *args: MockWrapper("MockSSD"),
    )

    # Run it
    run_benchmark(MockArgs())

    # 3. Assertions
    assert os.path.exists(output_csv), "CSV output file was not created"
    assert os.path.exists(output_sum), "Summary text file was not created"

    df = pd.read_csv(output_csv)
    assert not df.empty, "CSV output is empty"

    # Check specific columns and data integrity
    expected_cols = ["Model", "Video", "Person_Detections", "Vehicle_Detections"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    # We expect 5 rows (3 YOLO variants + EffDet + SSD)
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
    assert (df["Person_Detections"] > 0).all(), "Expected person detections in all rows"

    # Check summary file content
    with open(output_sum, "r") as f:
        content = f.read()
        # Verify specific YOLO versions are present
        assert "YOLOv8-Nano" in content
        assert "YOLOv8-Small" in content
        assert "YOLOv5-Nano" in content
        assert "MockEffDet" in content
        assert "MockSSD" in content


def test_benchmark_no_videos(monkeypatch, capsys):
    """
    Ensure the benchmark handles the case where no videos are found gracefully.
    """

    class MockArgs:
        models = "all"
        threads = 1
        input_size = 640
        warmup = 0
        max_frames = 0
        confidence = 0.25

    # Mock glob to return empty list (no videos found)
    monkeypatch.setattr("glob.glob", lambda p: [])

    run_benchmark(MockArgs())

    captured = capsys.readouterr()
    assert "No videos found" in captured.out
