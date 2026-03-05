import cv2
import torch
from .base import ModelWrapper, COCO_CLASSES


class EfficientDetWrapper(ModelWrapper):
    def __init__(
        self, name="EfficientDet-D0", weights_path=None, input_size=512, **kwargs
    ):
        # EfficientDet-D0 has strict architecture constraints (512x512 recommended)
        if name == "efficientdet_d0" and input_size != 512:
            print(
                f"Note: EfficientDet-D0 requires 512x512. Ignoring --input-size {input_size}."
            )
            input_size = 512
        super().__init__(
            name, input_size=input_size, weights_path=weights_path or "efficientdet_d0"
        )
        self.model_name = name if not weights_path else weights_path

    def load(self):
        print(f"Loading {self.name}...")
        from effdet import create_model

        self.model = create_model(
            self.model_name, bench_task="predict", pretrained=True
        )
        self.model.eval()

    def predict(self, frame):
        h, w = frame.shape[:2]
        img = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if len(frame.shape) == 2
            else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = (
            torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        )
        img_tensor = img_tensor / 255.0
        with torch.no_grad():
            output = self.model(img_tensor)
        detections = []
        if output is not None and len(output) > 0:
            for detection in output[0]:
                score = float(detection[4])
                cls_id = int(detection[5])
                label = COCO_CLASSES.get(cls_id, f"Class_{cls_id}")

                # Extract and rescale bbox coordinates
                # effdet returns [x1, y1, x2, y2] relative to input_size
                x1, y1, x2, y2 = detection[:4].tolist()
                x1 = x1 * (w / self.input_size)
                y1 = y1 * (h / self.input_size)
                x2 = x2 * (w / self.input_size)
                y2 = y2 * (h / self.input_size)

                detections.append((x1, y1, x2, y2, score, label))
        return detections
