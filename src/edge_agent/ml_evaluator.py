import logging
import os
import sys

# Ensure benchmark is importable (using the same sys.path hack as eval_accuracy.py for now)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.benchmark_suite import YOLOWrapper  # noqa: E402

logger = logging.getLogger(__name__)


class MLEvaluator:
    """
    Evaluates a clip of frames using YOLOv8-Nano to determine if a person
    or vehicle is present with high enough confidence to trigger an alert.
    """

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

    def evaluate_clip(self, frames: list) -> dict | None:
        """
        Runs YOLOv8-Nano on a list of frames.
        Returns the best detection details if a person or vehicle is found.
        Otherwise, returns None.
        """
        best_detection = None
        best_conf = 0.0
        best_frame = None

        for frame in frames:
            if frame is None:
                continue

            # Run inference using your wrapper
            detections = self.model.predict(frame)

            for det in detections:
                if len(det) == 6:
                    x1, y1, x2, y2, conf, label = det
                else:
                    # Fallback in case wrapper wasn't updated
                    label, conf = det
                    x1, y1, x2, y2 = 0, 0, 0, 0

                label_lower = label.lower()

                is_person = label_lower == "person" and conf >= self.person_conf
                is_vehicle = (
                    any(
                        v in label_lower
                        for v in ["car", "truck", "bus", "motorcycle", "vehicle"]
                    )
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
                        best_frame = frame

        if best_detection:
            return {"detection": best_detection, "frame": best_frame}

        return None
