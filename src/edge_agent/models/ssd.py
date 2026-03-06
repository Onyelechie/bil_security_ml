import cv2
import torch
from .base import ModelWrapper, COCO_CLASSES


class TorchvisionSSDWrapper(ModelWrapper):
    def __init__(
        self, name="SSD-MobileNet", weights_path=None, input_size=320, **kwargs
    ):
        # ssdlite320_mobilenet_v3_large is hardcoded to 320 in torchvision's default weights
        if input_size != 320:
            print(
                f"Note: SSD-MobileNet (SSDLite320) using native 320x320. Ignoring --input-size {input_size}."
            )
            input_size = 320
        super().__init__(
            name, input_size=input_size, weights_path=weights_path or "SSDLite320"
        )

    def load(self):
        print(f"Loading {self.name}...")
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def predict(self, frame):
        from PIL import Image

        h, w = frame.shape[:2]
        rgb_frame = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if len(frame.shape) == 2
            else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        img = Image.fromarray(rgb_frame)
        batch = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(batch)[0]
        detections = []
        for label, score, box in zip(
            prediction["labels"], prediction["scores"], prediction["boxes"]
        ):
            cls_id = label.item()
            label_str = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = box.tolist()

            # Rescale if needed (torchvision transforms might have resized)
            # Default SSDLite320 uses 320x320
            x1 = x1 * (w / 320.0)
            y1 = y1 * (h / 320.0)
            x2 = x2 * (w / 320.0)
            y2 = y2 * (h / 320.0)

            detections.append((x1, y1, x2, y2, float(score), label_str))
        return detections
