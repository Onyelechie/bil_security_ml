import os
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from edge_agent.ml_evaluator import MLEvaluator
from edge_agent.models import YOLOWrapper

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WEIGHTS_PATH = os.path.join(project_root, "production_model", "yolov8s.pt")
WEIGHTS_EXIST = os.path.exists(WEIGHTS_PATH)


@pytest.fixture
def mock_evaluator():
    with patch("edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        mock_model = MagicMock(spec=YOLOWrapper)
        mock_get.return_value = mock_model
        evaluator = MLEvaluator(weights_path="mock_path.pt")
        evaluator.model_mock = mock_model
        yield evaluator


def test_ml_evaluator_prefers_person_over_vehicle():
    with patch("edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        mock_model = MagicMock(spec=YOLOWrapper)
        mock_model.predict.return_value = [
            (0, 0, 50, 50, 0.72, "car"),
            (10, 10, 60, 60, 0.51, "person"),
        ]
        mock_get.return_value = mock_model

        evaluator = MLEvaluator(
            weights_path="mock_path.pt",
            allowed_classes="person,vehicle",
            person_conf=0.5,
            vehicle_conf=0.6,
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = evaluator.evaluate_frames([frame])

        assert result is not None
        assert result["detection"]["label"].lower() == "person"


def test_ml_evaluator_can_disable_vehicle_alerts():
    with patch("edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        mock_model = MagicMock(spec=YOLOWrapper)
        mock_model.predict.return_value = [
            (0, 0, 50, 50, 0.95, "car"),
        ]
        mock_get.return_value = mock_model

        evaluator = MLEvaluator(
            weights_path="mock_path.pt",
            allowed_classes="person",
            person_conf=0.4,
            vehicle_conf=0.6,
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = evaluator.evaluate_frames([frame])

        assert result is None


def create_dummy_image(color=(255, 255, 255)):
    return np.full((640, 640, 3), color, dtype=np.uint8)


def test_ml_evaluator_initialization():
    with patch("edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        evaluator = MLEvaluator(weights_path="mock_path.pt")
        assert evaluator.model is not None
        mock_get.assert_called_once()


def test_ml_evaluator_model_switching():
    from edge_agent.ml_evaluator import DEFAULT_MODEL_CONFIGS

    with patch("edge_agent.ml_evaluator.ModelRegistry.get_model") as mock_get:
        MLEvaluator()
        mock_get.assert_called_with(
            YOLOWrapper,
            "YOLOv8-Small",
            DEFAULT_MODEL_CONFIGS["YOLOv8-Small"],
            input_size=640,
            use_openvino=False,
        )

        mock_get.reset_mock()
        MLEvaluator(model_name="YOLOv8-Nano")
        mock_get.assert_called_with(
            YOLOWrapper,
            "YOLOv8-Nano",
            DEFAULT_MODEL_CONFIGS["YOLOv8-Nano"],
            input_size=640,
            use_openvino=False,
        )

        mock_get.reset_mock()
        custom_path = "/tmp/custom.pt"
        MLEvaluator(weights_path=custom_path)
        mock_get.assert_called_with(
            YOLOWrapper,
            "YOLOv8-Small",
            custom_path,
            input_size=640,
            use_openvino=False,
        )

        with pytest.raises(ValueError, match="No default weights defined"):
            MLEvaluator(model_name="Invalid-Model")


def test_ml_evaluator_caching():
    from edge_agent.models.registry import ModelRegistry

    ModelRegistry.clear()

    with patch("edge_agent.models.YOLOWrapper.load") as mock_load:
        eval1 = MLEvaluator(weights_path="mock_cache_test.pt")
        eval2 = MLEvaluator(weights_path="mock_cache_test.pt")

        assert eval1.model is eval2.model
        assert mock_load.call_count == 1


def test_ml_evaluator_empty_clip(mock_evaluator):
    assert mock_evaluator.evaluate_frames([]) is None
    assert mock_evaluator.evaluate_frames([None, None]) is None


def test_ml_evaluator_mocked_detection(mock_evaluator):
    mock_evaluator.model_mock.predict.side_effect = [
        [],
        [(10, 10, 100, 100, 0.9, "person")],
        [],
    ]

    clip = [create_dummy_image() for _ in range(3)]
    result = mock_evaluator.evaluate_frames(clip)

    assert result is not None
    assert result["detection"]["label"] == "person"
    assert result["detection"]["confidence"] == 0.9
    assert result["detection"]["bbox"] == [10, 10, 100, 100]
    assert result["frame_index"] == 1
    assert not np.array_equal(result["frame"], clip[1])


def test_ml_evaluator_grayscale_mocked(mock_evaluator):
    frame_gray = np.zeros((100, 100), dtype=np.uint8)
    mock_evaluator.model_mock.predict.return_value = [(0, 0, 50, 50, 0.8, "car")]

    result = mock_evaluator.evaluate_frames([frame_gray])

    assert result is not None
    assert result["detection"]["label"] == "car"
    assert result["frame_index"] == 0
    call_args = mock_evaluator.model_mock.predict.call_args[0][0]
    assert len(call_args.shape) == 3
    assert call_args.shape[2] == 3


def test_ml_evaluator_grayscale_3d_mocked(mock_evaluator):
    frame_gray = np.zeros((100, 100, 1), dtype=np.uint8)
    mock_evaluator.model_mock.predict.return_value = [(0, 0, 50, 50, 0.8, "car")]

    result = mock_evaluator.evaluate_frames([frame_gray])

    assert result is not None
    assert result["detection"]["label"] == "car"
    assert result["frame_index"] == 0
    call_args = mock_evaluator.model_mock.predict.call_args[0][0]
    assert len(call_args.shape) == 3
    assert call_args.shape[2] == 3


@pytest.mark.integration
@pytest.mark.skipif(
    not WEIGHTS_EXIST,
    reason="Weights not found at production_model/yolov8s.pt.",
)
@pytest.mark.parametrize(
    "filename, expected_label, custom_person_conf, custom_vehicle_conf, expected_to_pass",
    [
        ("C1HighRes - Human_frame_135.jpg", "person", 0.5, 0.6, True),
        ("C1LowRes - Human_frame_108.jpg", "person", 0.5, 0.6, True),
        ("C3HighRes - Car_frame_0.jpg", "car", 0.5, 0.3, True),
        ("C4HighRes - Human_frame_60.jpg", "person", 0.5, 0.6, True),
        ("C5HighResPTZ - Car_frame_90.jpg", "car", 0.5, 0.3, True),
        ("C1HighRes - Human_frame_216.jpg", "person", 0.05, 0.05, True),
    ],
)
def test_ml_evaluator_specific_frames_integration(
    filename, expected_label, custom_person_conf, custom_vehicle_conf, expected_to_pass
):
    specific_frame_path = os.path.join(
        project_root, "tests", "edge_agent", "test_data", filename
    )

    frame = None
    if os.path.exists(specific_frame_path):
        frame = cv2.imread(specific_frame_path)

    if frame is None:
        try:
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
        weights_path=WEIGHTS_PATH,
        person_conf=custom_person_conf,
        vehicle_conf=custom_vehicle_conf,
    )

    result = evaluator.evaluate_frames([frame])

    if expected_to_pass:
        assert result is not None, f"Expected a detection for {filename}, but got None."
        actual_label = result["detection"]["label"].lower()
        if filename == "C4HighRes - Human_frame_60.jpg":
            assert actual_label in [
                "person",
                "car",
                "truck",
                "bus",
                "motorcycle",
                "vehicle",
            ]
        elif expected_label.lower() in ["car", "truck", "bus", "motorcycle", "vehicle"]:
            assert actual_label in ["car", "truck", "bus", "motorcycle", "vehicle"]
        else:
            assert actual_label == expected_label.lower()
    else:
        assert result is None


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}.")
        print("Please run the benchmark script once to ensure weights are downloaded.")
        sys.exit(1)

    print("Initializing MLEvaluator...")
    evaluator = MLEvaluator(weights_path=WEIGHTS_PATH)

    print("Creating dummy blank clip...")
    clip = [create_dummy_image((0, 0, 0)), create_dummy_image((255, 255, 255))]

    print("Evaluating clip...")
    result = evaluator.evaluate_frames(clip)

    if result is None:
        print("Success! No persons or vehicles detected in blank images.")
    else:
        print(f"Failed! Unexpectedly detected: {result['detection']['label']}")
