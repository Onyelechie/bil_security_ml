import os
from pathlib import Path
from .base import ModelWrapper


class YOLOWrapper(ModelWrapper):
    def __init__(
        self,
        name="YOLOv8-Nano",
        weights_path=None,
        input_size=640,
        use_openvino=True,
    ):
        super().__init__(name, input_size=input_size, weights_path=weights_path)
        self.use_openvino = use_openvino

    def _openvino_enabled(self) -> bool:
        """
        Returns False when OpenVINO is explicitly disabled via env var.
        Useful for testing or troubleshooting.
        """
        flag = os.getenv("EDGE_AGENT_DISABLE_OPENVINO", "").strip().lower()
        return flag not in {"1", "true", "yes", "on"}

    def _openvino_dir(self) -> Path:
        """Returns the path to the compiled OpenVINO model directory."""
        pt_path = Path(self.weights_path)
        return pt_path.parent / (pt_path.stem + "_openvino_model")

    def _openvino_needs_export(self, ov_dir: Path) -> bool:
        """
        Export if OpenVINO artifacts are missing or older than the .pt weights.
        """
        if not ov_dir.exists():
            return True

        try:
            pt_mtime = Path(self.weights_path).stat().st_mtime
            ov_mtime = max(
                (p.stat().st_mtime for p in ov_dir.rglob("*") if p.is_file()),
                default=0,
            )
            return ov_mtime < pt_mtime
        except Exception:
            # If we can't stat, be safe and re-export.
            return True

    def load(self):
        from ultralytics import YOLO

        ov_dir = self._openvino_dir()

        # Download weights first if they don't exist
        if not os.path.exists(self.weights_path):
            weights_name = os.path.basename(self.weights_path)
            # If it's yolov5n.pt, use the updated 'yolov5nu.pt' for better compatibility with current Ultralytics
            download_name = "yolov5nu.pt" if weights_name == "yolov5n.pt" else weights_name

            print(f"Warning: {self.weights_path} not found.")
            print(f"Attempting to download {download_name} automatically...")

            # This downloads to the current working directory
            YOLO(download_name, task="detect")

            # Move it to the expected benchmark folder if it exists locally
            if os.path.exists(download_name) and not os.path.exists(self.weights_path):
                import shutil

                os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
                shutil.move(download_name, self.weights_path)
                print(f"Moved downloaded weights to {self.weights_path}")

        # Export to OpenVINO IR format if not already done (YOLOv8 only)
        # We only do this if use_openvino is enabled.
        weights_name_lower = os.path.basename(self.weights_path).lower()
        use_openvino = (
            self.use_openvino
            and self._openvino_enabled()
            and "yolov8" in weights_name_lower
        )
        if use_openvino:
            try:
                if self._openvino_needs_export(ov_dir):
                    print(
                        f"Compiling {self.name} to OpenVINO format (one-time export)..."
                    )
                    pt_model = YOLO(self.weights_path, task="detect")
                    pt_model.export(format="openvino", imgsz=self.input_size)
                    print(f"Export complete -> {ov_dir}")

                # Load the compiled OpenVINO model
                print(f"Loading {self.name} from OpenVINO: {ov_dir}")
                self.model = YOLO(str(ov_dir), task="detect")
                return
            except Exception as e:
                print(f"OpenVINO load failed: {e}. Falling back to PyTorch.")

        # Non-YOLOv8 models or explicit PyTorch requested
        print(f"Loading {self.name} from PyTorch: {self.weights_path}")
        self.model = YOLO(self.weights_path, task="detect")

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
