# Detection module
from .config import (
    DEFAULT_MODEL,
    ALERT_CLASSES,
    PRIORITY_CLASSES,
    MODEL_INFO,
    get_model_info,
    list_available_models,
    get_recommended_model
)
from .detector import (
    Detection,
    ObjectDetector,
    YOLOv8Detector,
    create_detector,
    get_recommended_detector,
    calculate_iou,
    non_max_suppression
)
from .motion import MotionDetector, MotionRegion
from .intrusion_pipeline import IntrusionDetectionPipeline, IntrusionResult

__all__ = [
    # Config
    "DEFAULT_MODEL",
    "ALERT_CLASSES",
    "PRIORITY_CLASSES",
    "MODEL_INFO",
    "get_model_info",
    "list_available_models",
    "get_recommended_model",
    # Detector
    "Detection",
    "ObjectDetector",
    "YOLOv8Detector",
    "create_detector",
    "get_recommended_detector",
    "calculate_iou",
    "non_max_suppression",
    # Motion
    "MotionDetector",
    "MotionRegion",
    # Pipeline
    "IntrusionDetectionPipeline",
    "IntrusionResult",
]