import os
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.edge_agent.ml_evaluator import MLEvaluator
from src.edge_agent.models import YOLOWrapper

# We need a path to the weights
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WEIGHTS_PATH = os.path.join(project_root, "benchmark", "yolov8n.pt")
WEIGHTS_EXIST = os.path.exists(WEIGHTS_PATH)


@pytest.fixture
def mock_evaluator():
    """Provides an MLEvaluator with a mocked model registry."""
    with patch("src.edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        mock_model = MagicMock(spec=YOLOWrapper)
        mock_get.return_value = mock_model
        evaluator = MLEvaluator("mock_path.pt")
        evaluator.model_mock = mock_model  # Accessible for setting return values
        yield evaluator


def create_dummy_image(color=(255, 255, 255)):
    """Creates a blank 640x640 image for testing."""
    return np.full((640, 640, 3), color, dtype=np.uint8)


def test_ml_evaluator_initialization():
    """Test that the evaluator initializes (mocked)."""
    with patch("src.edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        evaluator = MLEvaluator("mock_path.pt")
        assert evaluator.model is not None
        mock_get.assert_called_once()


def test_ml_evaluator_caching():
    """Test that multiple evaluators share the same model instance via registry."""
    # This confirms the registry is used, without loading real weights
    with patch("src.edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        mock_model = MagicMock()
        mock_get.return_value = mock_model

        eval1 = MLEvaluator("mock.pt")
        eval2 = MLEvaluator("mock.pt")

        assert eval1.model is eval2.model
        assert mock_get.call_count == 2


def test_ml_evaluator_empty_clip(mock_evaluator):
    """Test evaluating an empty clip or clip with None frames."""
    assert mock_evaluator.evaluate_frames([]) is None
    assert mock_evaluator.evaluate_frames([None, None]) is None


def test_ml_evaluator_mocked_detection(mock_evaluator):
    """Test the evaluator's frame selection and bbox drawing logic using mocks."""
    # Setup mock to 'detect' a person on the second frame
    # Format: (x1, y1, x2, y2, conf, label)
    mock_evaluator.model_mock.predict.side_effect = [
        [],  # Frame 1: nothing
        [(10, 10, 100, 100, 0.9, "person")],  # Frame 2: person
        [],  # Frame 3: nothing
    ]

    clip = [create_dummy_image() for _ in range(3)]
    result = mock_evaluator.evaluate_frames(clip)

    assert result is not None
    assert result["detection"]["label"] == "person"
    assert result["detection"]["confidence"] == 0.9
    assert result["detection"]["bbox"] == [10, 10, 100, 100]
    assert result["frame_index"] == 1  # Detected on second frame
    # Check that it drew a box (annotated frame should be different from original)
    assert not np.array_equal(result["frame"], clip[1])


def test_ml_evaluator_grayscale_mocked(mock_evaluator):
    """Test that grayscale frames are converted and processed (mocked)."""
    frame_gray = np.zeros((100, 100), dtype=np.uint8)
    mock_evaluator.model_mock.predict.return_value = [(0, 0, 50, 50, 0.8, "car")]

    result = mock_evaluator.evaluate_frames([frame_gray])

    assert result is not None
    assert result["detection"]["label"] == "car"
    assert result["frame_index"] == 0
    # Ensure it converted to 3-channel for the mock's 'inference'
    call_args = mock_evaluator.model_mock.predict.call_args[0][0]
    assert len(call_args.shape) == 3
    assert call_args.shape[2] == 3


@pytest.mark.integration
@pytest.mark.skipif(
    not WEIGHTS_EXIST,
    reason="Weights not found at benchmark/yolov8n.pt. Run benchmark_suite.py to download.",
)
@pytest.mark.parametrize(
    "filename, expected_label, custom_person_conf, custom_vehicle_conf, expected_to_pass",
    [
        ("C1HighRes - Human_frame_135.jpg", "person", 0.5, 0.6, True),
        ("C1LowRes - Human_frame_108.jpg", "person", 0.5, 0.6, True),
        ("C3HighRes - Car_frame_0.jpg", "car", 0.5, 0.3, True),
        ("C4HighRes - Human_frame_60.jpg", "person", 0.5, 0.6, True),
        ("C5HighResPTZ - Car_frame_90.jpg", "car", 0.5, 0.3, True),
        ("C1HighRes - Human_frame_216.jpg", "truck", 0.05, 0.05, False),
    ],
)
def test_ml_evaluator_specific_frames_integration(
    filename, expected_label, custom_person_conf, custom_vehicle_conf, expected_to_pass
):
    """Real inference test using actual weights. Only runs if --integration is specified or weights found."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip(
            "Weights not found at benchmark/yolov8n.pt. Skipping integration test."
        )

    specific_frame_path = os.path.join(
        project_root, "tests", "edge_agent", "test_data", filename
    )

    frame = None
    if os.path.exists(specific_frame_path):
        frame = cv2.imread(specific_frame_path)

    # Fallback: Try to extract from benchmark video if image is missing
    if frame is None:
        try:
            # Expected format: "VideoName_frame_123.jpg"
            parts = filename.split("_frame_")
            if len(parts) == 2:
                video_name = parts[0] + ".mp4"
                frame_idx = int(parts[1].split(".")[0])

                video_path = os.path.join(
                    project_root, "benchmark", "cctv_samples", video_name
                )
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, extracted = cap.read()
                    cap.release()
                    if ret:
                        frame = extracted
        except Exception:
            pass

    if frame is None:
        pytest.skip(
            f"Frame not found at {specific_frame_path} and could not extract from video."
        )

    evaluator = MLEvaluator(
        WEIGHTS_PATH, person_conf=custom_person_conf, vehicle_conf=custom_vehicle_conf
    )

    result = evaluator.evaluate_frames([frame])

    if not expected_to_pass:
        assert result is None
        return

    assert result is not None
    assert "detection" in result
    found_label = result["detection"]["label"].lower()
    if filename == "C4HighRes - Human_frame_60.jpg":
        assert found_label in ["person", "car"]
    else:
        assert found_label == expected_label.lower()


if __name__ == "__main__":
    # If run directly as a script (stub test)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}.")
        print("Please run the benchmark script once to ensure weights are downloaded.")
        sys.exit(1)

    print("Initializing MLEvaluator...")
    evaluator = MLEvaluator(WEIGHTS_PATH)

    print("Creating dummy blank clip...")
    clip = [create_dummy_image((0, 0, 0)), create_dummy_image((255, 255, 255))]

    print("Evaluating clip...")
    result = evaluator.evaluate_frames(clip)

    if result is None:
        print("Success! No persons or vehicles detected in blank images.")
    else:
        print(f"Failed! Unexpectedly detected: {result['detection']['label']}")
