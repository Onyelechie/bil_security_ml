from .base import ModelWrapper, COCO_CLASSES
from .yolo import YOLOWrapper
from .registry import ModelRegistry

__all__ = ["ModelWrapper", "COCO_CLASSES", "YOLOWrapper", "ModelRegistry"]
# from .efficientdet import EfficientDetWrapper  # Optional: only if dependencies installed
# from .ssd import TorchvisionSSDWrapper      # Optional: only if dependencies installed
