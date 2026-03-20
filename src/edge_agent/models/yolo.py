import os
from .base import ModelWrapper


class YOLOWrapper(ModelWrapper):
    def __init__(self, name="YOLOv8-Nano", weights_path=None, input_size=640):
        super().__init__(name, input_size=input_size, weights_path=weights_path)

    def load(self):
        print(f"Loading {self.name} ({self.weights_path})...")
        from ultralytics import YOLO

        if not os.path.exists(self.weights_path):
            weights_name = os.path.basename(self.weights_path)
            print(f"Warning: {self.weights_path} not found.")
            print(f"Attempting to download {weights_name} automatically...")
            self.model = YOLO(weights_name)
            if os.path.exists(weights_name) and not os.path.exists(self.weights_path):
                try:
                    import shutil

                    shutil.move(weights_name, self.weights_path)
                    print(f"Moved downloaded weights to {self.weights_path}")
                except Exception as e:
                    print(f"Note: Could not move weights to {self.weights_path}: {e}")
        else:
            self.model = YOLO(self.weights_path)

    def predict(self, frame):
        # Ultralytics YOLO supports imgsz parameter directly
        results = self.model(frame, verbose=False, imgsz=self.input_size)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((x1, y1, x2, y2, conf, label))
        return detections
