import os
from pathlib import Path
from .base import ModelWrapper


class YOLOWrapper(ModelWrapper):
    def __init__(self, name="YOLOv8-Nano", weights_path=None, input_size=640):
        super().__init__(name, input_size=input_size, weights_path=weights_path)

    def _openvino_dir(self) -> Path:
        """Returns the path to the compiled OpenVINO model directory."""
        pt_path = Path(self.weights_path)
        return pt_path.parent / (pt_path.stem + "_openvino_model")

    def load(self):
        from ultralytics import YOLO

        ov_dir = self._openvino_dir()

        # Download weights first if they don't exist
        if not os.path.exists(self.weights_path):
            weights_name = os.path.basename(self.weights_path)
            print(f"Warning: {self.weights_path} not found.")
            print(f"Attempting to download {weights_name} automatically...")
            YOLO(weights_name)
            if os.path.exists(weights_name) and not os.path.exists(self.weights_path):
                import shutil
                shutil.move(weights_name, self.weights_path)
                print(f"Moved downloaded weights to {self.weights_path}")

        # Export to OpenVINO IR format if not already done (YOLOv8 only)
        weights_name_lower = os.path.basename(self.weights_path).lower()
        if "yolov8" in weights_name_lower:
            if not ov_dir.exists():
                print(f"Compiling {self.name} to OpenVINO format (one-time export)...")
                pt_model = YOLO(self.weights_path)
                pt_model.export(format="openvino", imgsz=self.input_size)
                print(f"Export complete → {ov_dir}")

            # Load the compiled OpenVINO model
            print(f"Loading {self.name} from OpenVINO: {ov_dir}")
            self.model = YOLO(str(ov_dir))
        else:
            # Non-YOLOv8 models (e.g. YOLOv5) use standard PyTorch runtime
            print(f"Loading {self.name} from PyTorch: {self.weights_path}")
            self.model = YOLO(self.weights_path)

    def predict(self, frame):
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
