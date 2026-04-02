import logging
import os

import cv2
import numpy as np

from .models import ModelRegistry, YOLOWrapper

logger = logging.getLogger(__name__)

COLOR_PERSON = (0, 255, 0)
COLOR_VEHICLE = (255, 165, 0)

best_person_any = -1.0
best_vehicle_any = -1.0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRODUCTION_DIR = os.path.join(PROJECT_ROOT, "production_model")

DEFAULT_MODEL_CONFIGS = {
    "YOLOv8-Small": os.path.join(PRODUCTION_DIR, "yolov8s.pt"),
    "YOLOv8-Nano": os.path.join(PRODUCTION_DIR, "yolov8n.pt"),
}


class MLEvaluator:
    """
    Evaluates a clip of frames using YOLO to determine if a person or vehicle is
    present with high enough confidence to trigger an alert.

    Improvements:
    - per-camera / per-run allowed classes
    - higher default vehicle threshold
    - prefer person over vehicle when both are valid
    """

    VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "vehicle"}
    SUPPORTED_ALERT_CLASSES = {"person", "vehicle"}

    def __init__(
        self,
        model_name: str = "YOLOv8-Small",
        weights_path: str | None = None,
        person_conf: float = 0.5,
        vehicle_conf: float = 0.6,
        allowed_classes: str | list[str] | set[str] | None = "person,vehicle",
    ):
        self.person_conf = float(person_conf)
        self.vehicle_conf = float(vehicle_conf)
        self.allowed_classes = self._normalize_allowed_classes(allowed_classes)

        if not weights_path:
            weights_path = DEFAULT_MODEL_CONFIGS.get(model_name)
            if not weights_path:
                raise ValueError(
                    f"No default weights defined for model type: {model_name}"
                )

        self.model = ModelRegistry.get_model(
            YOLOWrapper,
            model_name,
            weights_path,
            input_size=640,
            use_openvino=False,
        )
        logger.info(
            "MLEvaluator initialized with model=%s allowed_classes=%s person_conf=%.2f vehicle_conf=%.2f",
            weights_path,
            sorted(self.allowed_classes),
            self.person_conf,
            self.vehicle_conf,
        )

    @classmethod
    def _normalize_allowed_classes(
        cls, allowed_classes: str | list[str] | set[str] | None
    ) -> set[str]:
        if allowed_classes is None:
            return {"person", "vehicle"}

        if isinstance(allowed_classes, str):
            parts = [p.strip().lower() for p in allowed_classes.split(",")]
            out = {p for p in parts if p in cls.SUPPORTED_ALERT_CLASSES}
            return out or {"person", "vehicle"}

        out = {str(p).strip().lower() for p in allowed_classes}
        out = {p for p in out if p in cls.SUPPORTED_ALERT_CLASSES}
        return out or {"person", "vehicle"}

    def evaluate_frames(self, frames: list[np.ndarray]) -> dict | None:
        """
        Runs YOLO on a list of frames.
        If frames are grayscale, they are converted to BGR for compatibility.

        Selection logic:
        - choose the best valid PERSON if any valid person exists
        - otherwise choose the best valid VEHICLE
        """
        best_person = None
        best_person_conf = -1.0

        best_vehicle = None
        best_vehicle_conf = -1.0

        best_person_any = -1.0
        best_vehicle_any = -1.0

        for idx, frame in enumerate(frames):
            if frame is None:
                continue

            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            detections = self.model.predict(frame_bgr)

            for det in detections:
                x1, y1, x2, y2, conf, label = det
                label_lower = label.lower()
                conf = float(conf)

                if label_lower == "person":
                    best_person_any = max(best_person_any, conf)
                elif label_lower in self.VEHICLE_LABELS:
                    best_vehicle_any = max(best_vehicle_any, conf)

                is_person = (
                    "person" in self.allowed_classes
                    and label_lower == "person"
                    and conf >= self.person_conf
                )

                is_vehicle = (
                    "vehicle" in self.allowed_classes
                    and label_lower in self.VEHICLE_LABELS
                    and conf >= self.vehicle_conf
                )

                if is_person and conf > best_person_conf:
                    best_person_conf = conf
                    best_person = {
                        "detection": {
                            "label": label,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        },
                        "frame": frame_bgr,
                        "frame_index": idx,
                    }

                elif is_vehicle and conf > best_vehicle_conf:
                    best_vehicle_conf = conf
                    best_vehicle = {
                        "detection": {
                            "label": label,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        },
                        "frame": frame_bgr,
                        "frame_index": idx,
                    }

        chosen = best_person if best_person is not None else best_vehicle

        if chosen is None:
            logger.info(
                "No valid detection in chunk: best_person_any=%.2f best_vehicle_any=%.2f total_frames=%d",
                best_person_any,
                best_vehicle_any,
                len(frames),
            )
            return None

        logger.info(
            "Chosen detection: class=%s conf=%.2f frame_index=%d "
            "best_person_any=%.2f best_vehicle_any=%.2f total_frames=%d",
            chosen["detection"]["label"],
            chosen["detection"]["confidence"],
            chosen["frame_index"],
            best_person_any,
            best_vehicle_any,
            len(frames),
        )

        annotated = self._draw_bbox(
            chosen["frame"],
            chosen["detection"]["bbox"],
            chosen["detection"]["label"],
            chosen["detection"]["confidence"],
        )
        return {
            "detection": chosen["detection"],
            "frame": annotated,
            "frame_index": chosen["frame_index"],
        }

    @staticmethod
    def _draw_bbox(
        frame: np.ndarray,
        bbox: list,
        label: str,
        confidence: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        x1, y1, x2, y2 = [int(c) for c in bbox]

        color = COLOR_PERSON if label.lower() == "person" else COLOR_VEHICLE

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        if y1 - th - 8 < 0:
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

    def evaluate_frame_all(self, frame: np.ndarray) -> dict | None:
        """
        Run YOLO on a single frame and return ALL valid detections for that frame.
        """
        if frame is None:
            return None

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        raw_detections = self.model.predict(frame_bgr)
        valid_detections = []

        for det in raw_detections:
            x1, y1, x2, y2, conf, label = det
            label_lower = label.lower()

            is_person = (
                "person" in self.allowed_classes
                and label_lower == "person"
                and conf >= self.person_conf
            )

            is_vehicle = (
                "vehicle" in self.allowed_classes
                and label_lower in self.VEHICLE_LABELS
                and conf >= self.vehicle_conf
            )

            if not (is_person or is_vehicle):
                continue

            valid_detections.append(
                {
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        if not valid_detections:
            return None

        annotated = frame_bgr.copy()
        for det in valid_detections:
            annotated = self._draw_bbox(
                annotated,
                det["bbox"],
                det["label"],
                det["confidence"],
            )

        return {
            "detections": valid_detections,
            "frame": annotated,
        }
