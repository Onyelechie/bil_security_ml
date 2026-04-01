import os
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .models import ModelRegistry, YOLOWrapper

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import EdgeSettings
# Bounding box colors (BGR)
COLOR_PERSON = (0, 255, 0)  # Green
COLOR_VEHICLE = (255, 165, 0)  # Orange

# Default production model paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRODUCTION_DIR = os.path.join(PROJECT_ROOT, "production_model")

DEFAULT_MODEL_CONFIGS = {
    "YOLOv8-Small": os.path.join(PRODUCTION_DIR, "yolov8s.pt"),
    "YOLOv8-Nano": os.path.join(PRODUCTION_DIR, "yolov8n.pt"),
}


class MLEvaluator:
    """
    Evaluates a clip of frames (max 40, BGR format from RingBuffer)
    using YOLOv8-Small to determine if a person or vehicle is present
    with high enough confidence to trigger an alert.

    Returns the annotated frame with bounding box drawn on it.
    """

    VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "vehicle"}

    def __init__(
        self,
        model_name: str = "YOLOv8-Small",
        weights_path: str | None = None,
        person_conf: float = 0.5,
        vehicle_conf: float = 0.6,
    ):
        self.person_conf = person_conf
        self.vehicle_conf = vehicle_conf

        if weights_path is None:
            weights_path = DEFAULT_MODEL_CONFIGS.get(model_name)
            if not weights_path:
                raise ValueError(f"No default weights defined for model type: {model_name}")

        # Use the registry to get a cached instance of YOLO
        self.model = ModelRegistry.get_model(
            YOLOWrapper, model_name, weights_path, input_size=640
        )
        logger.info(f"MLEvaluator initialized with model from {weights_path}")

    @classmethod
    def from_settings(cls, cfg: "EdgeSettings") -> "MLEvaluator":
        """
        Build an evaluator from EdgeSettings detector configuration.
        """
        return cls(
            model_name=cfg.detector_model,
            weights_path=cfg.detector_weights,
        )

    def evaluate_frames(self, frames: list[np.ndarray]) -> dict | None:
        """
        Runs the configured YOLO model on a list of frames.
        If frames are grayscale, they are converted to BGR for YOLO compatibility.
        Returns the best detection with an annotated frame (bounding box drawn),
        or None if no person/vehicle found.
        """
        best_detection = None
        best_conf = 0.0
        best_frame_index = -1
        best_frame = None

        for idx, frame in enumerate(frames):
            if frame is None:
                continue

            # Handle grayscale to BGR conversion at inference time
            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            detections = self.model.predict(frame_bgr)

            for det in detections:
                x1, y1, x2, y2, conf, label = det

                label_lower = label.lower()

                is_person = label_lower == "person" and conf >= self.person_conf
                is_vehicle = (
                    label_lower in self.VEHICLE_LABELS and conf >= self.vehicle_conf
                )

                if is_person or is_vehicle:
                    if conf > best_conf:
                        best_conf = conf
                        best_detection = {
                            "label": label,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                        best_frame = frame_bgr  # Use the BGR version for annotation
                        best_frame_index = idx

        if best_detection and best_frame is not None:
            annotated = self._draw_bbox(
                best_frame,
                best_detection["bbox"],
                best_detection["label"],
                best_detection["confidence"],
            )
            return {
                "detection": best_detection,
                "frame": annotated,
                "frame_index": best_frame_index,
            }

        return None

    @staticmethod
    def _draw_bbox(
        frame: np.ndarray,
        bbox: list,
        label: str,
        confidence: float,
    ) -> np.ndarray:
        """
        Draws a bounding box with label and confidence on a copy of the frame.
        Returns the annotated image (does not modify the original).
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Pick color based on object type
        color = COLOR_PERSON if label.lower() == "person" else COLOR_VEHICLE

        # Draw bounding box (thickness=2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background + text
        text = f"{label} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Ensure label is visible (handle top edge)
        if y1 - th - 8 < 0:
            # Draw inside/below top edge
            cv2.rectangle(annotated, (x1, y1), (x1 + tw + 4, y1 + th + 8), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 2, y1 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
