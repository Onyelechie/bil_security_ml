"""
Object Detection module for BIL Security Intrusion Detection System

Supports multiple detection backends:
- MobileNet-SSD: Fast, lightweight (21 classes)
- YOLOv8: More accurate (80 classes), multiple size variants

Optimized for multi-camera scenarios (10+ cameras on i5/i7 with 4GB RAM)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from . import config


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def non_max_suppression(
    detections: List['Detection'],
    iou_threshold: float = 0.4
) -> List['Detection']:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.

    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for considering boxes as duplicates

    Returns:
        Filtered list of detections with duplicates removed
    """
    if not detections:
        return []

    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []

    while sorted_detections:
        best = sorted_detections.pop(0)
        keep.append(best)

        remaining = []
        for det in sorted_detections:
            if det.class_name != best.class_name:
                remaining.append(det)
            elif calculate_iou(best.bbox, det.bbox) < iou_threshold:
                remaining.append(det)

        sorted_detections = remaining

    return keep


@dataclass
class Detection:
    """Represents a single object detection"""
    class_name: str
    class_index: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def is_alert_class(self) -> bool:
        """Check if this detection is an alert-worthy class (person/vehicle)"""
        return self.class_name in config.ALERT_CLASSES
    
    def is_priority_class(self) -> bool:
        """Check if this is a priority class (e.g., person)"""
        return self.class_name in config.PRIORITY_CLASSES


class ObjectDetector:
    """
    Object detector using MobileNet-SSD model.
    
    Fast and lightweight - recommended for 10+ camera setups.
    """

    def __init__(
        self,
        prototxt_path: str = None,
        model_path: str = None,
        min_confidence: float = None
    ):
        """
        Initialize the MobileNet-SSD detector.

        Args:
            prototxt_path: Path to the prototxt file
            model_path: Path to the caffemodel file
            min_confidence: Minimum confidence threshold
        """
        self.prototxt_path = str(prototxt_path or config.MOBILENET_PROTOTXT)
        self.model_path = str(model_path or config.MOBILENET_CAFFEMODEL)
        self.min_confidence = min_confidence or config.MIN_CONFIDENCE

        self.net = None
        self.classes = config.MOBILENET_CLASSES
        
        # Generate consistent colors for each class
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        self.model_name = 'mobilenet'

    def load_model(self) -> bool:
        """Load the neural network model."""
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            return True
        except Exception as e:
            print(f"Error loading MobileNet model: {e}")
            return False

    def _detect_single(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame without tiling."""
        if self.net is None:
            if not self.load_model():
                return []

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, config.INPUT_SIZE),
            config.BLOB_SCALE,
            config.INPUT_SIZE,
            config.BLOB_MEAN
        )

        self.net.setInput(blob)
        detections_raw = self.net.forward()

        detections = []

        for i in range(detections_raw.shape[2]):
            confidence = detections_raw[0, 0, i, 2]

            if confidence > self.min_confidence:
                class_index = int(detections_raw[0, 0, i, 1])

                x1 = int(detections_raw[0, 0, i, 3] * width)
                y1 = int(detections_raw[0, 0, i, 4] * height)
                x2 = int(detections_raw[0, 0, i, 5] * width)
                y2 = int(detections_raw[0, 0, i, 6] * height)

                detection = Detection(
                    class_name=self.classes[class_index],
                    class_index=class_index,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2)
                )
                detections.append(detection)

        return detections

    def detect_tiled(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects using tiled/sliding window approach.
        Better for CCTV footage where objects may be small/distant.
        """
        height, width = frame.shape[:2]
        tile_size = config.TILE_SIZE
        overlap = config.TILE_OVERLAP
        max_tiles = config.MAX_TILES

        step = int(tile_size * (1 - overlap))
        all_detections = []

        # Full frame detection first
        full_frame_detections = self._detect_single(frame)
        all_detections.extend(full_frame_detections)

        if height < tile_size or width < tile_size:
            return all_detections

        # Generate tile positions
        y_positions = list(range(0, height - tile_size + 1, step))
        x_positions = list(range(0, width - tile_size + 1, step))

        if y_positions and y_positions[-1] + tile_size < height:
            y_positions.append(height - tile_size)
        if x_positions and x_positions[-1] + tile_size < width:
            x_positions.append(width - tile_size)

        tile_positions = [(y, x) for y in y_positions for x in x_positions]

        if len(tile_positions) > max_tiles:
            tile_positions = self._select_priority_tiles(
                tile_positions, max_tiles, height, width, tile_size
            )

        for y, x in tile_positions:
            tile = frame[y:y + tile_size, x:x + tile_size]
            tile_detections = self._detect_single(tile)

            for det in tile_detections:
                x1, y1, x2, y2 = det.bbox
                adjusted_detection = Detection(
                    class_name=det.class_name,
                    class_index=det.class_index,
                    confidence=det.confidence,
                    bbox=(x1 + x, y1 + y, x2 + x, y2 + y)
                )
                all_detections.append(adjusted_detection)

        return non_max_suppression(all_detections, iou_threshold=config.NMS_THRESHOLD)

    def _select_priority_tiles(
        self,
        tile_positions: List[Tuple[int, int]],
        max_tiles: int,
        height: int,
        width: int,
        tile_size: int
    ) -> List[Tuple[int, int]]:
        """Select priority tiles when there are too many."""
        if len(tile_positions) <= max_tiles:
            return tile_positions

        selected = []

        # Center tile
        center_y = (height - tile_size) // 2
        center_x = (width - tile_size) // 2
        center_tile = min(tile_positions, key=lambda t: abs(t[0] - center_y) + abs(t[1] - center_x))
        selected.append(center_tile)

        # Corner tiles
        corners = [
            (0, 0),
            (0, width - tile_size),
            (height - tile_size, 0),
            (height - tile_size, width - tile_size),
        ]
        for cy, cx in corners:
            nearest = min(tile_positions, key=lambda t: abs(t[0] - cy) + abs(t[1] - cx))
            if nearest not in selected:
                selected.append(nearest)
                if len(selected) >= max_tiles:
                    return selected

        # Fill remaining
        remaining = [t for t in tile_positions if t not in selected]
        step = max(1, len(remaining) // (max_tiles - len(selected)))
        for i in range(0, len(remaining), step):
            if len(selected) >= max_tiles:
                break
            selected.append(remaining[i])

        return selected

    def detect(self, frame: np.ndarray, use_tiled: bool = None) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input image/frame as numpy array
            use_tiled: Whether to use tiled detection. If None, uses config setting.

        Returns:
            List of Detection objects
        """
        if use_tiled is None:
            use_tiled = config.ENABLE_TILED

        if use_tiled:
            return self.detect_tiled(frame)
        else:
            return self._detect_single(frame)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        frame_copy = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.colors[detection.class_index % len(self.colors)]

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            if show_confidence:
                label = f"{detection.class_name}: {detection.confidence*100:.1f}%"
            else:
                label = detection.class_name

            label_y = y1 - 15 if y1 > 30 else y1 + 15
            cv2.putText(
                frame_copy, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return frame_copy

    def filter_by_class(
        self,
        detections: List[Detection],
        class_names: List[str]
    ) -> List[Detection]:
        """Filter detections by class names."""
        return [d for d in detections if d.class_name in class_names]

    def filter_by_confidence(
        self,
        detections: List[Detection],
        min_confidence: float
    ) -> List[Detection]:
        """Filter detections by minimum confidence."""
        return [d for d in detections if d.confidence >= min_confidence]
    
    def filter_alert_classes(self, detections: List[Detection]) -> List[Detection]:
        """Filter to only alert-worthy classes (person, vehicle, etc.)"""
        return [d for d in detections if d.is_alert_class()]


# =============================================================================
# YOLOv8 Detector
# =============================================================================

class YOLOv8Detector:
    """
    Object detector using YOLOv8 model.
    
    More accurate than MobileNet-SSD, especially for people.
    Requires: pip install ultralytics

    Model variants (speed vs accuracy):
        - yolov8n: Nano - fastest, recommended for 8+ cameras
        - yolov8s: Small - good balance, recommended for 5+ cameras
        - yolov8m: Medium - better accuracy, recommended for 3+ cameras
        - yolov8l: Large - high accuracy, recommended for 2 cameras
        - yolov8x: Extra Large - highest accuracy, recommended for 1 camera
    """

    MODELS = {
        'yolov8n': 'yolov8n.pt',
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8x': 'yolov8x.pt',
    }

    def __init__(
        self,
        model_name: str = "yolov8n",
        min_confidence: float = None,
        device: str = None
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model_name: Model variant - 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            min_confidence: Minimum confidence threshold
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        if model_name in self.MODELS:
            self.model_name = self.MODELS[model_name]
        elif model_name.endswith('.pt'):
            self.model_name = model_name
        else:
            self.model_name = model_name + '.pt'

        self.min_confidence = min_confidence or config.MIN_CONFIDENCE
        self.device = device
        self.model = None
        self.classes = config.YOLO_CLASSES

        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_model(self) -> bool:
        """Load the YOLOv8 model (downloads automatically if not present)."""
        try:
            from ultralytics import YOLO

            print(f"Loading YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)

            if self.device:
                self.model.to(self.device)

            print(f"YOLOv8 model loaded successfully!")
            return True

        except ImportError:
            print("=" * 50)
            print("ERROR: ultralytics package not installed!")
            print("Install with: pip install ultralytics")
            print("=" * 50)
            return False
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            return False

    def detect(self, frame: np.ndarray, use_tiled: bool = None) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input image/frame as numpy array
            use_tiled: Ignored for YOLOv8 (handles multi-scale internally)

        Returns:
            List of Detection objects
        """
        if self.model is None:
            if not self.load_model():
                return []

        results = self.model(frame, conf=self.min_confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"

                detection = Detection(
                    class_name=class_name,
                    class_index=class_id,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                )
                detections.append(detection)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        frame_copy = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.colors[detection.class_index % len(self.colors)]

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            if show_confidence:
                label = f"{detection.class_name}: {detection.confidence*100:.1f}%"
            else:
                label = detection.class_name

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 > 30 else y1 + 20

            cv2.rectangle(
                frame_copy,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color, -1
            )
            cv2.putText(
                frame_copy, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return frame_copy

    def filter_by_class(
        self,
        detections: List[Detection],
        class_names: List[str]
    ) -> List[Detection]:
        """Filter detections by class names."""
        return [d for d in detections if d.class_name in class_names]

    def filter_by_confidence(
        self,
        detections: List[Detection],
        min_confidence: float
    ) -> List[Detection]:
        """Filter detections by minimum confidence."""
        return [d for d in detections if d.confidence >= min_confidence]
    
    def filter_alert_classes(self, detections: List[Detection]) -> List[Detection]:
        """Filter to only alert-worthy classes (person, vehicle, etc.)"""
        return [d for d in detections if d.is_alert_class()]


# =============================================================================
# Factory Function
# =============================================================================

def create_detector(model_type: str = None, **kwargs):
    """
    Factory function to create the appropriate detector.

    Args:
        model_type: 'mobilenet' for MobileNet-SSD, or 'yolov8n/s/m/l/x' for YOLOv8
        **kwargs: Additional arguments passed to detector constructor

    Returns:
        Detector instance (ObjectDetector or YOLOv8Detector)

    Examples:
        detector = create_detector('mobilenet')  # Fast, for 10+ cameras
        detector = create_detector('yolov8n')    # Good balance
        detector = create_detector('yolov8s')    # Better accuracy
    """
    if model_type is None:
        model_type = config.DEFAULT_MODEL

    model_type = model_type.lower()

    if model_type in ('mobilenet', 'mobilenet-ssd', 'ssd'):
        return ObjectDetector(**kwargs)
    elif model_type.startswith('yolo'):
        if model_type in YOLOv8Detector.MODELS:
            return YOLOv8Detector(model_name=model_type, **kwargs)
        else:
            return YOLOv8Detector(model_name='yolov8n', **kwargs)
    else:
        print(f"Unknown model type: {model_type}, defaulting to MobileNet-SSD")
        return ObjectDetector(**kwargs)


def get_recommended_detector(num_cameras: int):
    """
    Get the recommended detector for a given number of cameras.
    
    Optimized for i5/i7 with 4GB RAM.
    
    Args:
        num_cameras: Number of simultaneous camera feeds
        
    Returns:
        Configured detector instance
    """
    recommended_model = config.get_recommended_model(num_cameras)
    print(f"Recommended model for {num_cameras} cameras: {recommended_model}")
    return create_detector(recommended_model)
