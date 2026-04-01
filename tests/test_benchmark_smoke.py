import os

import pandas as pd

from benchmark.benchmark_suite import create_dummy_video, run_benchmark
from tests.temp_dirs import repo_temp_dir


def test_benchmark_smoke(monkeypatch):
    """
    Smoke test to ensure the benchmark pipeline runs and produces output.
    Uses a dummy video and mocks models to avoid heavy downloads/inference.
    """
    # 1. Setup mock environment
    with repo_temp_dir("benchmark-") as temp_path:
        dummy_video = temp_path / "test_video.mp4"
        create_dummy_video(str(dummy_video), duration=1)

        output_csv = temp_path / "test_results.csv"
        output_sum = temp_path / "test_summary.txt"
        monkeypatch.setattr("benchmark.benchmark_suite.OUTPUT_CSV", str(output_csv))
        monkeypatch.setattr("benchmark.benchmark_suite.OUTPUT_SUMMARY", str(output_sum))

        from src.edge_agent.models import ModelWrapper

        class MockWrapper(ModelWrapper):
            def load(self):
                pass

            def predict(self, frame):
                return [(0, 0, 0, 0, 0.9, "person")]

        class MockArgs:
            def __init__(self):
                self.models = "all"
                self.threads = 1
                self.input_size = 640
                self.warmup = 1
                self.max_frames = 5
                self.confidence = 0.25
                self.production = False

        def mock_glob(path):
            return [str(dummy_video)]

        monkeypatch.setattr("glob.glob", mock_glob)
        monkeypatch.setattr("benchmark.benchmark_suite.VIDEO_EXTENSIONS", ["*.mp4"])
        monkeypatch.setattr(
            "src.edge_agent.models.YOLOWrapper", lambda *args, **kwargs: MockWrapper(args[0])
        )
        monkeypatch.setattr(
            "src.edge_agent.models.efficientdet.EfficientDetWrapper",
            lambda *args, **kwargs: MockWrapper("MockEffDet"),
        )
        monkeypatch.setattr(
            "src.edge_agent.models.ssd.TorchvisionSSDWrapper",
            lambda *args, **kwargs: MockWrapper("MockSSD"),
        )

        run_benchmark(MockArgs())

        assert os.path.exists(output_csv), "CSV output file was not created"
        assert os.path.exists(output_sum), "Summary text file was not created"

        df = pd.read_csv(output_csv)
        assert not df.empty, "CSV output is empty"

        expected_cols = ["Model", "Video", "Person_Detections", "Vehicle_Detections"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
        assert (
            df["Person_Detections"] > 0
        ).all(), "Expected person detections in all rows"

        with open(output_sum, "r") as f:
            content = f.read()
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
        production = False

    # Mock glob to return empty list (no videos found)
    monkeypatch.setattr("glob.glob", lambda p: [])

    run_benchmark(MockArgs())

    captured = capsys.readouterr()
    assert "No videos found" in captured.out
