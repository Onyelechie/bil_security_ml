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
#   - 'mobilenet': Legacy, lightweight (21 classes) - SLOW at HD resolution!
#   - 'yolov8n': YOLO Nano - RECOMMENDED (80 classes) - needs: pip install ultralytics
#   - 'yolov8s': YOLO Small - Best accuracy (80 classes)
#   - 'yolov8m': YOLO Medium - better accuracy (80 classes)
#   - 'yolov8l': YOLO Large - high accuracy (80 classes)
#   - 'yolov8x': YOLO Extra Large - highest accuracy (80 classes) - needs GPU for 10 cameras
#
# RECOMMENDATION: YOLOv8n for best balance of speed and accuracy at HD resolution.
# 
# NOTE - MobileNet Resolution Issue (January 2026 Testing):
#   - At 640x480: MobileNet is 3x faster than YOLOv8n
#   - At 1280x720: MobileNet is 31% SLOWER than YOLOv8n!
#   - Surveillance cameras typically output 720p/1080p
#   - MobileNet should only be used for legacy/low-power systems or pre-resized 480p streams

DEFAULT_MODEL = 'yolov8n'  # Changed from yolov8s - faster at HD with good accuracy

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
# Updated January 2026 - Benchmarked at 1280x720 (native HD resolution)
# =============================================================================
MODEL_INFO = {
    'mobilenet': {
        'name': 'MobileNet-SSD',
        'description': 'Legacy CNN - Slow at HD',
        'classes': 21,
        'speed': '~16 FPS @ 720p',
        'accuracy': 'Lower',
        'memory': 'Low (~50MB)',
        'recommended_cameras': 10,
        'requires': 'OpenCV (included)',
        'use_case': 'Legacy only - use YOLOv8n instead',
    },
    'yolov8n': {
        'name': 'YOLOv8 Nano',
        'description': 'Best speed/accuracy at HD (RECOMMENDED)',
        'classes': 80,
        'speed': '~20 FPS @ 720p',
        'accuracy': 'Good',
        'memory': 'Low (~100MB)',
        'recommended_cameras': 10,
        'requires': 'pip install ultralytics',
        'use_case': 'RECOMMENDED - Best for most cameras',
    },
    'yolov8s': {
        'name': 'YOLOv8 Small',
        'description': 'Best accuracy',
        'classes': 80,
        'speed': '~14 FPS @ 720p',
        'accuracy': 'Better',
        'memory': 'Medium (~150MB)',
        'recommended_cameras': 8,
        'requires': 'pip install ultralytics',
        'use_case': 'Maximum accuracy needed',
    },
    'yolov8m': {
        'name': 'YOLOv8 Medium',
        'description': 'Higher accuracy',
        'classes': 80,
        'speed': '~6 FPS @ 720p',
        'accuracy': 'High',
        'memory': 'Higher (~300MB)',
        'recommended_cameras': 3,
        'requires': 'pip install ultralytics',
        'use_case': 'High-priority cameras, GPU helps',
    },
    'yolov8l': {
        'name': 'YOLOv8 Large',
        'description': 'High accuracy',
        'classes': 80,
        'speed': '~3 FPS @ 720p',
        'accuracy': 'Very High',
        'memory': 'High (~500MB)',
        'recommended_cameras': 2,
        'requires': 'pip install ultralytics',
        'use_case': 'GPU recommended',
    },
    'yolov8x': {
        'name': 'YOLOv8 Extra Large',
        'description': 'Highest accuracy',
        'classes': 80,
        'speed': '~2 FPS @ 720p',
        'accuracy': 'Highest',
        'memory': 'Very High (~700MB)',
        'recommended_cameras': 1,
        'requires': 'pip install ultralytics',
        'use_case': 'GPU only, maximum accuracy',
    },
}

# List of models available for selection
AVAILABLE_MODELS = list(MODEL_INFO.keys())

# Default confidence threshold
DEFAULT_CONFIDENCE = 0.5

# Default motion threshold for false alarm filtering
DEFAULT_MOTION_THRESHOLD = 0.01


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
