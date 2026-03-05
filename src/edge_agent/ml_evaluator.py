import logging

import cv2
import numpy as np
from .models.yolo import YOLOWrapper

logger = logging.getLogger(__name__)

# Bounding box colors (BGR)
COLOR_PERSON = (0, 255, 0)  # Green
COLOR_VEHICLE = (255, 165, 0)  # Orange


class MLEvaluator:
    """
    Evaluates a clip of frames (max 40, BGR format from RingBuffer)
    using YOLOv8-Nano to determine if a person or vehicle is present
    with high enough confidence to trigger an alert.

    Returns the annotated frame with bounding box drawn on it.
    """

    VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "vehicle"}

    def __init__(
        self, weights_path: str, person_conf: float = 0.5, vehicle_conf: float = 0.6
    ):
        self.person_conf = person_conf
        self.vehicle_conf = vehicle_conf

        # We use the YOLO wrapper you built in the benchmark suite
        self.model = YOLOWrapper("YOLOv8-Nano", weights_path, input_size=640)

        try:
            self.model.load()
            logger.info(f"MLEvaluator loaded {weights_path} successfully.")
        except Exception as e:
            logger.error(f"MLEvaluator failed to load {weights_path}: {e}")
            raise

    def evaluate_frames(self, frames: list) -> dict | None:
        """
        Runs YOLOv8-Nano on a list of frames (BGR or Grayscale, up to 40 from RingBuffer).
        If frames are grayscale, they are converted to BGR for YOLO compatibility.
        Returns the best detection with an annotated frame (bounding box drawn),
        or None if no person/vehicle found.
        """
        best_detection = None
        best_conf = 0.0
        best_frame = None

        for frame in frames:
            if frame is None:
                continue

            # Handle grayscale to BGR conversion at inference time
            if len(frame.shape) == 2:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            detections = self.model.predict(frame_bgr)

            for det in detections:
                x1, y1, x2, y2, conf, label = det

                label_lower = label.lower()

                is_person = label_lower == "person" and conf >= self.person_conf
                is_vehicle = (
                    any(v in label_lower for v in self.VEHICLE_LABELS)
                    and conf >= self.vehicle_conf
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

        if best_detection and best_frame is not None:
            annotated = self._draw_bbox(
                best_frame,
                best_detection["bbox"],
                best_detection["label"],
                best_detection["confidence"],
            )
            return {"detection": best_detection, "frame": annotated}

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
