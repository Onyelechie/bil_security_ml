"""
Configuration settings for BIL Security Intrusion Detection System

This module contains all configurable parameters for the detection system.
Settings are organized by category for easy modification.
"""

import os
from pathlib import Path
from typing import List

# =============================================================================
# Base Paths
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent  # Project root
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONFIGS_DIR = BASE_DIR / "configs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model Selection
# =============================================================================
# Available models (user selectable via frontend):
#   - 'mobilenet': Fast, lightweight (21 classes) - no extra install needed
#   - 'yolov8n': YOLO Nano - fastest YOLO (80 classes) - needs: pip install ultralytics
#   - 'yolov8s': YOLO Small - RECOMMENDED for accuracy (80 classes)
#   - 'yolov8m': YOLO Medium - better accuracy (80 classes)
#   - 'yolov8l': YOLO Large - high accuracy (80 classes)
#   - 'yolov8x': YOLO Extra Large - highest accuracy (80 classes) - needs GPU for 10 cameras
#
# RECOMMENDATION: YOLOv8s for best accuracy. Manual testing showed it detects
# significantly better than MobileNet, especially for distant/small objects.
# For 10 cameras with YOLOv8x, need RTX 3080+ GPU.

DEFAULT_MODEL = 'yolov8s'

# Model paths (for MobileNet-SSD)
MOBILENET_PROTOTXT = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
MOBILENET_CAFFEMODEL = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"

# Alternative MobileNet model
MOBILENET_ALT_CAFFEMODEL = MODELS_DIR / "mobilenet_iter_73000.caffemodel"

# YOLOv8 model directory (models auto-download on first use)
YOLO_MODELS_DIR = MODELS_DIR

# =============================================================================
# Detection Settings
# =============================================================================
MIN_CONFIDENCE = 0.25        # Minimum confidence to display detections
ALERT_THRESHOLD = 0.5        # Confidence threshold for triggering alerts
INPUT_SIZE = (300, 300)      # MobileNet input size
BLOB_SCALE = 0.007843
BLOB_MEAN = 130.0

# =============================================================================
# Target Classes - What we're looking for
# =============================================================================
# For BIL Security: Focus on people and vehicles (intrusion detection)
ALERT_CLASSES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
PRIORITY_CLASSES = ["person"]  # Extra attention to people

# MobileNet-SSD Classes (21 total)
MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# YOLOv8/COCO Classes (80 total)
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# =============================================================================
# Motion Detection Settings (False Alarm Filtering)
# =============================================================================
MOTION_DETECTION_ENABLED = True   # Filter alerts to only moving objects
MOTION_MIN_AREA = 500             # Minimum pixel area to consider as motion
MOTION_OVERLAP_THRESHOLD = 0.05   # Min overlap between detection and motion region
MOTION_WARMUP_FRAMES = 30         # Frames to wait before detecting motion
SHOW_MOTION_REGIONS = False       # Draw motion regions (debugging)

# =============================================================================
# Tiled Detection Settings (for CCTV/small objects)
# =============================================================================
ENABLE_TILED = True           # Enable tiled detection
TILE_SIZE = 512               # Size of each tile
TILE_OVERLAP = 0.20           # Overlap between tiles
NMS_THRESHOLD = 0.3           # Non-max suppression threshold
TILED_FRAME_SKIP = 2          # Run tiled detection every N frames
MAX_TILES = 12                # Maximum tiles to process

# =============================================================================
# Multi-Camera Scaling (Target: 10 cameras on i5/i7, 4GB RAM)
# =============================================================================
MAX_CAMERAS = 10              # Maximum simultaneous camera feeds
DEFAULT_FPS = 15              # Target FPS per camera
FRAME_BUFFER_SECONDS = 30     # Seconds of frames to keep in buffer per camera

# Per-camera resource limits
MAX_FRAMES_PER_BUFFER = 500   # Maximum frames in ring buffer per camera
DETECTION_FRAME_SKIP = 3      # Run detection every N frames (saves CPU)
RESIZE_FOR_DETECTION = True   # Resize frames before detection
DETECTION_RESIZE = (640, 480) # Size for detection (smaller = faster)

# Processing strategies for multi-camera
PROCESSING_MODE = 'round-robin'  # 'round-robin', 'priority', 'event-driven'

# =============================================================================
# Event Frame Window
# =============================================================================
EVENT_WINDOW_BEFORE_SECONDS = 2.0   # Seconds before event to capture
EVENT_WINDOW_AFTER_SECONDS = 5.0    # Seconds after event to capture

# =============================================================================
# Alert Settings
# =============================================================================
ALERT_SOUND_ENABLED = True
ALERT_SOUND_FREQUENCY = 1000  # Hz
ALERT_SOUND_DURATION = 200    # ms
ALERT_LOG_FILE = LOGS_DIR / "alerts.log"
ALERT_COOLDOWN_SECONDS = 5    # Minimum time between alerts for same camera

# =============================================================================
# TCP Event Listener
# =============================================================================
TCP_HOST = "0.0.0.0"
TCP_PORT = 9000

# =============================================================================
# Output Settings
# =============================================================================
OUTPUT_DIR = BASE_DIR / "event_clips"
OUTPUT_FORMAT = "video"       # "video" (mp4) or "images" (folder)
VIDEO_CODEC = "mp4v"
IMAGE_FORMAT = "jpg"
JPEG_QUALITY = 95

# =============================================================================
# Model Performance Estimates (for UI display)
# =============================================================================
MODEL_INFO = {
    'mobilenet': {
        'name': 'MobileNet-SSD',
        'description': 'Fast, lightweight CNN',
        'classes': 21,
        'speed': 'Very Fast (~100+ FPS)',
        'accuracy': 'Good',
        'memory': 'Low (~50MB)',
        'recommended_cameras': 10,
        'requires': 'OpenCV (included)',
    },
    'yolov8n': {
        'name': 'YOLOv8 Nano',
        'description': 'Fastest YOLO variant',
        'classes': 80,
        'speed': 'Fast (~80-100 FPS)',
        'accuracy': 'Good',
        'memory': 'Low (~100MB)',
        'recommended_cameras': 8,
        'requires': 'pip install ultralytics',
    },
    'yolov8s': {
        'name': 'YOLOv8 Small',
        'description': 'Good speed/accuracy balance',
        'classes': 80,
        'speed': 'Medium (~50-60 FPS)',
        'accuracy': 'Better',
        'memory': 'Medium (~150MB)',
        'recommended_cameras': 5,
        'requires': 'pip install ultralytics',
    },
    'yolov8m': {
        'name': 'YOLOv8 Medium',
        'description': 'Better accuracy',
        'classes': 80,
        'speed': 'Slower (~30-40 FPS)',
        'accuracy': 'High',
        'memory': 'Higher (~300MB)',
        'recommended_cameras': 3,
        'requires': 'pip install ultralytics',
    },
    'yolov8l': {
        'name': 'YOLOv8 Large',
        'description': 'High accuracy',
        'classes': 80,
        'speed': 'Slow (~20-25 FPS)',
        'accuracy': 'Very High',
        'memory': 'High (~500MB)',
        'recommended_cameras': 2,
        'requires': 'pip install ultralytics',
    },
    'yolov8x': {
        'name': 'YOLOv8 Extra Large',
        'description': 'Highest accuracy',
        'classes': 80,
        'speed': 'Slowest (~10-15 FPS)',
        'accuracy': 'Highest',
        'memory': 'Very High (~700MB)',
        'recommended_cameras': 1,
        'requires': 'pip install ultralytics',
    },
}


def get_model_info(model_name: str) -> dict:
    """Get info about a specific model."""
    return MODEL_INFO.get(model_name.lower(), MODEL_INFO['mobilenet'])


def list_available_models() -> List[str]:
    """List all available model names."""
    return list(MODEL_INFO.keys())


def get_recommended_model(num_cameras: int) -> str:
    """
    Get recommended model based on number of cameras.
    
    For 10 cameras on i5/i7 with 4GB RAM:
    - mobilenet or yolov8n are the best choices
    """
    for model_name, info in MODEL_INFO.items():
        if info['recommended_cameras'] >= num_cameras:
            return model_name
    return 'mobilenet'  # Safest default for many cameras
