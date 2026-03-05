import gc
from abc import ABC, abstractmethod

# COCO Class Mapping (Standard IDs)
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


class ModelWrapper(ABC):
    """
    Abstract Base Class for all object detection models.
    """

    def __init__(self, name, input_size=None, weights_path=None):
        self.name = name
        self.model = None
        self.input_size = input_size
        self.weights_path = weights_path

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        gc.collect()
