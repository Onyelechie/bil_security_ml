import os
import sys
import glob
import cv2
import pytest
import numpy as np

# Ensure src is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ruff: noqa: E402
from src.edge_agent.ml_evaluator import MLEvaluator

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
    result = evaluator.evaluate_clip([])
    assert result is None

    # List of Nones
    result = evaluator.evaluate_clip([None, None])
    assert result is None


def test_ml_evaluator_blank_frames():
    """Test evaluating a clip with blank frames (should detect nothing)."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip("Weights not found.")

    evaluator = MLEvaluator(WEIGHTS_PATH)
    clip = [create_dummy_image() for _ in range(3)]

    result = evaluator.evaluate_clip(clip)
    assert result is None


def test_ml_evaluator_real_frames():
    """Test evaluating a clip using actual images from dataset_frames."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip("Weights not found.")

    dataset_dir = os.path.join(project_root, "accuracy", "dataset_frames")
    if not os.path.exists(dataset_dir):
        pytest.skip(f"Dataset directory not found at {dataset_dir}")

    # Find all jpg frames in the directory
    frame_paths = glob.glob(os.path.join(dataset_dir, "*.jpg"))

    if not frame_paths:
        pytest.skip(f"No .jpg frames found in {dataset_dir}")

    # We'll test with just the first 3 frames to form a "clip"
    clip_paths = frame_paths[:3]
    clip_frames = []

    for path in clip_paths:
        frame = cv2.imread(path)
        if frame is not None:
            clip_frames.append(frame)

    if not clip_frames:
        pytest.skip("Could not load any frames into cv2")

    evaluator = MLEvaluator(WEIGHTS_PATH)
    result = evaluator.evaluate_clip(clip_frames)

    # Since these are real dataset frames from CCTV, there is a very high likelihood
    # that at least one person or car exists in these 3 random frames.
    assert result is not None, (
        "Evaluator failed to detect any person/vehicle in the real dataset frames."
    )
    assert "detection" in result
    assert "bbox" in result["detection"]
    assert result["detection"]["label"] in [
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "vehicle",
    ]


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
    result = evaluator.evaluate_clip(clip)

    if result is None:
        print("Success! No persons or vehicles detected in blank images.")
    else:
        print(f"Failed! Unexpectedly detected: {result['detection']['label']}")
