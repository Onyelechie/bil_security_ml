from .base import COCO_CLASSES, ModelWrapper
from .registry import ModelRegistry
from .yolo import YOLOWrapper

__all__ = ["ModelWrapper", "COCO_CLASSES", "YOLOWrapper", "ModelRegistry"]
# from .efficientdet import EfficientDetWrapper  # Optional: only if dependencies installed
# from .ssd import TorchvisionSSDWrapper      # Optional: only if dependencies installed
