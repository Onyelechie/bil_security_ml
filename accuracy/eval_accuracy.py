import os
import cv2
import torch
import warnings
import argparse
import pandas as pd
from pathlib import Path
from typing import List

# Suppress specific pytorch warnings that might clutter the CLI output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    print(
        "Error: torchmetrics is not installed. Please run: pip install torchmetrics torchvision"
    )
    exit(1)

# Ensure benchmark models are importable
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# We extend the ModelWrapper slightly to return bounding boxes
from benchmark.benchmark_suite import (
    ModelWrapper,
    YOLOWrapper,
    EfficientDetWrapper,
    TorchvisionSSDWrapper,
    COCO_CLASSES,
)

OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "accuracy_results.csv"
)


# --- Monkeypatching the Wrappers to return raw boxes [x1, y1, x2, y2, conf, cls_name] ---
# For accuracy benchmarking, we need exact coordinates, not just counts.


def yolo_predict_with_bbox(self, frame):
    results = self.model(frame, verbose=False, imgsz=self.input_size)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self.model.names[cls_id]
            detections.append([x1, y1, x2, y2, conf, label])
    return detections


def effdet_predict_with_bbox(self, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
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
            x1, y1, x2, y2, score, cls_id = detection.tolist()

            # EfficientDet outputs at the resized scale. Map back to original image scale
            scale_x = img_w / self.input_size
            scale_y = img_h / self.input_size
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y

            label = COCO_CLASSES.get(int(cls_id), f"Class_{int(cls_id)}")
            detections.append([x1, y1, x2, y2, float(score), label])
    return detections


def ssd_predict_with_bbox(self, frame):
    from PIL import Image

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    batch = self.preprocess(img).unsqueeze(0)
    with torch.no_grad():
        prediction = self.model(batch)[0]

    detections = []
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        cls_id = label.item()
        x1, y1, x2, y2 = box.tolist()
        label_str = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
        detections.append([x1, y1, x2, y2, float(score), label_str])
    return detections


YOLOWrapper.predict = yolo_predict_with_bbox
EfficientDetWrapper.predict = effdet_predict_with_bbox
TorchvisionSSDWrapper.predict = ssd_predict_with_bbox
# --------------------------------------------------------------------------


def load_yolo_labels(txt_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    """
    Parses a YOLO format ground-truth text file.
    YOLO format: class_id x_center y_center width height (normalized 0.0 to 1.0)
    Returns: list of [x1, y1, x2, y2, cls_id]
    """
    if not txt_path.exists():
        return []

    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                # Note: This assigns arbitrary integer mapping since COCO/YOLO diff.
                # In a massive dataset, you should map your YOLO class IDs to standard names.
                # For this benchmark, we align them internally.
                x_c, y_c, w, h = map(float, parts[1:5])

                # Un-normalize
                x_c *= img_w
                y_c *= img_h
                w *= img_w
                h *= img_h

                # Convert to x1, y1, x2, y2
                x1 = x_c - (w / 2)
                y1 = y_c - (h / 2)
                x2 = x_c + (w / 2)
                y2 = y_c + (h / 2)

                boxes.append([x1, y1, x2, y2, cls_id])

    return boxes


def run_accuracy_evaluation(args):
    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(
            f"Error: Could not find 'images' or 'labels' directories inside {dataset_dir}"
        )
        print("Ensure you exported the dataset in 'YOLO' format.")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Loaded dataset: {len(image_files)} test images found.")

    # 1. Setup Models
    script_dir = os.path.join(project_root, "benchmark")
    available_models = {
        "YOLOv8-Nano": YOLOWrapper(
            "YOLOv8-Nano", os.path.join(script_dir, "yolov8n.pt"), args.input_size
        ),
        "YOLOv8-Small": YOLOWrapper(
            "YOLOv8-Small", os.path.join(script_dir, "yolov8s.pt"), args.input_size
        ),
        "YOLOv5-Nano": YOLOWrapper(
            "YOLOv5-Nano", os.path.join(script_dir, "yolov5n.pt"), args.input_size
        ),
        "EfficientDet-D0": EfficientDetWrapper("efficientdet_d0", args.input_size),
        "SSD-MobileNet": TorchvisionSSDWrapper("SSD-MobileNet", args.input_size),
    }

    selected_model_names = (
        args.models.split(",")
        if args.models != "all"
        else list(available_models.keys())
    )
    models_to_run = [
        available_models[m] for m in selected_model_names if m in available_models
    ]

    all_results = []

    # We maintain a crude class ID mapping for torchmetrics.
    # We will hash string labels (e.g. "person") into stable ints.
    def str_to_cls_id(label_str):
        name = label_str.lower()
        if name == "person" or name == "0":  # YOLO export usually makes person 0
            return 0
        if "car" in name or "vehicle" in name or name == "2":
            return 2
        return hash(name) % 100 + 10  # offset unknown classes

    for model_wrapper in models_to_run:
        print(f"\n{'=' * 30}\nEvaluating: {model_wrapper.name}\n{'=' * 30}")
        try:
            model_wrapper.load()
        except Exception as e:
            print(f"Failed to load {model_wrapper.name}: {e}")
            continue

        # torchmetrics MeanAveragePrecision evaluator
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

        preds_for_metric = []
        targets_for_metric = []

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            img_h, img_w = frame.shape[:2]

            # 1. Load Ground Truth
            label_filename = img_path.stem + ".txt"
            label_path = labels_dir / label_filename
            gt_boxes = load_yolo_labels(label_path, img_w, img_h)

            # Build target dict for torchmetrics
            if gt_boxes:
                target_boxes = torch.tensor(
                    [b[:4] for b in gt_boxes], dtype=torch.float32
                )
                target_labels = torch.tensor(
                    [b[4] for b in gt_boxes], dtype=torch.int64
                )
            else:
                target_boxes = torch.empty((0, 4), dtype=torch.float32)
                target_labels = torch.empty((0,), dtype=torch.int64)

            targets_for_metric.append(dict(boxes=target_boxes, labels=target_labels))

            # 2. Run Inference
            detections = model_wrapper.predict(frame)

            # Build prediction dict for torchmetrics
            if detections:
                pred_boxes = torch.tensor(
                    [d[:4] for d in detections], dtype=torch.float32
                )
                pred_scores = torch.tensor(
                    [d[4] for d in detections], dtype=torch.float32
                )
                pred_labels = torch.tensor(
                    [str_to_cls_id(d[5]) for d in detections], dtype=torch.int64
                )
            else:
                pred_boxes = torch.empty((0, 4), dtype=torch.float32)
                pred_scores = torch.empty((0,), dtype=torch.float32)
                pred_labels = torch.empty((0,), dtype=torch.int64)

            preds_for_metric.append(
                dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)
            )

        # 3. Calculate metrics for this model across entire dataset
        metric.update(preds_for_metric, targets_for_metric)
        results = metric.compute()

        # Unpack metrics
        map_50_95 = results["map"].item()
        map_50 = results["map_50"].item()
        map_75 = results["map_75"].item()

        print(f"Results for {model_wrapper.name}:")
        print(f"  mAP @ 50:      {map_50:.4f}")
        print(f"  mAP @ 50-95:   {map_50_95:.4f}")

        all_results.append(
            {
                "Model": model_wrapper.name,
                "mAP_50": round(map_50, 4),
                "mAP_50_95": round(map_50_95, 4),
                "mAP_75": round(map_75, 4),
            }
        )

        model_wrapper.unload()

    # Save outputs
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nFinal Accuracy Results saved to {OUTPUT_CSV}")
        print("-" * 30)
        print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Model Accuracy against YOLO-labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the YOLO-formatted dataset (must contain 'images' and 'labels' subfolders).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated models to evaluate.",
    )
    parser.add_argument(
        "--input-size", type=int, default=640, help="Input resolution (imgsz)."
    )

    args = parser.parse_args()

    # Needs torchmetrics to be installed
    run_accuracy_evaluation(args)
