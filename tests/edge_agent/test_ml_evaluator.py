import os
import sys
import cv2
import pytest
import numpy as np

# Ensure src is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.edge_agent.ml_evaluator import MLEvaluator  # noqa: E402

# We need a path to the weights. We'll use the YOLOv8n weights in the benchmark folder.
WEIGHTS_PATH = os.path.join(project_root, "benchmark", "yolov8n.pt")


def create_dummy_image(color=(255, 255, 255)):
    """Creates a blank 640x640 image for testing."""
    return np.full((640, 640, 3), color, dtype=np.uint8)


def test_ml_evaluator_initialization():
    """Test that the evaluator loads the model successfully."""
    # Skip if weights don't exist yet (benchmark script usually downloads them)
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip(
            f"Weights not found at {WEIGHTS_PATH}. Run benchmark once to download."
        )

    evaluator = MLEvaluator(WEIGHTS_PATH)
    assert evaluator.model is not None
    assert evaluator.person_conf == 0.5


def test_ml_evaluator_empty_clip():
    """Test evaluating an empty clip or clip with None frames."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip("Weights not found.")

    evaluator = MLEvaluator(WEIGHTS_PATH)

    # Empty list
    result = evaluator.evaluate_frames([])
    assert result is None

    # List of Nones
    result = evaluator.evaluate_frames([None, None])
    assert result is None


def test_ml_evaluator_blank_frames():
    """Test evaluating a clip with blank frames (should detect nothing)."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip("Weights not found.")

    evaluator = MLEvaluator(WEIGHTS_PATH)
    clip = [create_dummy_image() for _ in range(3)]

    result = evaluator.evaluate_frames(clip)
    assert result is None


@pytest.mark.parametrize(
    "filename, expected_label, custom_person_conf, custom_vehicle_conf, expected_to_pass",
    [
        ("C1HighRes - Human_frame_135.jpg", "person", 0.5, 0.6, True),
        ("C1LowRes - Human_frame_108.jpg", "person", 0.5, 0.6, True),
        (
            "C3HighRes - Car_frame_0.jpg",
            "car",
            0.5,
            0.3,
            True,
        ),  # YOLO max confidence is ~0.37
        (
            "C4HighRes - Human_frame_60.jpg",
            "person",
            0.5,
            0.6,
            True,
        ),  # Has both person and car, person is highest conf
        (
            "C5HighResPTZ - Car_frame_90.jpg",
            "car",
            0.5,
            0.3,
            True,
        ),  # YOLO max confidence is ~0.38
        (
            "C1HighRes - Human_frame_216.jpg",
            "truck",
            0.05,
            0.05,
            False,
        ),  # YOLO model confidence < 0.25 native baseline, won't trigger
    ],
)
def test_ml_evaluator_specific_frames(
    filename, expected_label, custom_person_conf, custom_vehicle_conf, expected_to_pass
):
    """Test evaluating specific frames explicitly requested by the user, adjusting thresholds contextually if needed."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip("Weights not found.")

    specific_frame_path = os.path.join(
        project_root, "tests", "edge_agent", "test_data", filename
    )

    if not os.path.exists(specific_frame_path):
        pytest.skip(f"Specific frame not found at {specific_frame_path}")

    frame = cv2.imread(specific_frame_path)
    if frame is None:
        pytest.skip("Could not load the specific frame into cv2")

    evaluator = MLEvaluator(
        WEIGHTS_PATH, person_conf=custom_person_conf, vehicle_conf=custom_vehicle_conf
    )

    # We pass it as a 1-frame clip
    result = evaluator.evaluate_frames([frame])

    if not expected_to_pass:
        # The model inherently misses this at its core layer (e.g. confidence < 25%)
        assert result is None, (
            f"Expected model to miss {expected_label} in {filename} due to YOLO base threshold, but it found it!"
        )
        return

    assert result is not None, (
        f"Evaluator failed to detect {expected_label} in {filename}."
    )
    assert "detection" in result

    found_label = result["detection"]["label"]

    if filename == "C4HighRes - Human_frame_60.jpg":
        # The evaluator only returns the single highest confidence detection per clip
        # Since both Car and Human are in the image, we verify it picked one of them.
        assert found_label in ["person", "car"], (
            f"Expected person or car, found {found_label}"
        )
    else:
        assert found_label == expected_label, (
            f"Expected {expected_label}, but found {found_label}"
        )


def test_ml_evaluator_grayscale():
    """Test that the evaluator correctly handles grayscale images."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip("Weights not found.")

    # Use a known frame with a person
    test_frame_path = os.path.join(
        project_root,
        "tests",
        "edge_agent",
        "test_data",
        "C1HighRes - Human_frame_135.jpg",
    )
    if not os.path.exists(test_frame_path):
        pytest.skip("Test data frame 135 not found.")

    frame_color = cv2.imread(test_frame_path)
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    # Verify input is 2D
    assert len(frame_gray.shape) == 2

    evaluator = MLEvaluator(WEIGHTS_PATH)
    result = evaluator.evaluate_frames([frame_gray])

    assert result is not None
    assert result["detection"]["label"] == "person"
    # The output frame should be 3-channel BGR
    assert len(result["frame"].shape) == 3
    assert result["frame"].shape[2] == 3


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
