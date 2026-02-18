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

    # Override constants via monkeypatch
    monkeypatch.setattr("benchmark.benchmark_suite.VIDEO_EXTENSIONS", [])
    monkeypatch.setattr("benchmark.benchmark_suite.MAX_FRAMES_PER_VIDEO", 5)
    monkeypatch.setattr("benchmark.benchmark_suite.WARMUP_FRAMES", 1)

    output_csv = tmp_path / "test_results.csv"
    output_sum = tmp_path / "test_summary.txt"
    monkeypatch.setattr("benchmark.benchmark_suite.OUTPUT_CSV", str(output_csv))
    monkeypatch.setattr("benchmark.benchmark_suite.OUTPUT_SUMMARY", str(output_sum))

    # 2. Mock model list
    from benchmark.benchmark_suite import ModelWrapper

    class MockWrapper(ModelWrapper):
        def load(self):
            pass

        def predict(self, frame):
            return [("person", 0.9)]

        def unload(self):
            pass

    # We need to monkeypatch the search logic
    def mock_glob(path):
        return [str(dummy_video)]

    monkeypatch.setattr("glob.glob", mock_glob)

    monkeypatch.setattr(
        "benchmark.benchmark_suite.YOLOWrapper", lambda *args: MockWrapper("MockYOLO")
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
    run_benchmark()

    # 3. Assertions
    assert os.path.exists(output_csv), "CSV output file was not created"
    assert os.path.exists(output_sum), "Summary text file was not created"

    df = pd.read_csv(output_csv)
    assert not df.empty
    assert "Person_Detections" in df.columns
    assert "Vehicle_Detections" in df.columns
    assert df.iloc[0]["Person_Detections"] > 0
