"""
Motion Detection module for filtering false alarms.

Uses background subtraction + frame differencing to detect actual movement,
filtering out stationary objects like parked cars, furniture, etc.

This is critical for reducing false alarms in security monitoring.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from . import config


@dataclass
class MotionRegion:
    """Represents a region with detected motion"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area: int
    intensity: float  # 0-1 scale of motion intensity


class MotionDetector:
    """
    Detects motion between consecutive frames using background subtraction.
    
    Key for filtering false alarms:
    - Parked cars appear in object detection but NOT in motion detection
    - Moving people/vehicles appear in BOTH
    - Weather/vegetation creates motion but not person/vehicle detections
    
    By requiring BOTH motion AND object detection, we filter:
    - Stationary objects (parked cars, furniture)
    - Random motion (weather, animals, vegetation)
    """

    def __init__(
        self,
        min_area: int = None,
        blur_size: int = 21,
        threshold: int = 20,
        dilate_iterations: int = 3,
        history: int = 500,
        var_threshold: int = 25,
        learning_rate: float = 0.005,
        warmup_frames: int = None
    ):
        """
        Initialize motion detector.

        Args:
            min_area: Minimum contour area to consider as motion
            blur_size: Gaussian blur kernel size (must be odd)
            threshold: Threshold for binary conversion
            dilate_iterations: Number of dilation iterations
            history: Number of frames for background model
            var_threshold: Variance threshold for background subtraction
            learning_rate: How fast background adapts (lower = slower, more stable)
            warmup_frames: Frames to skip before detecting motion
        """
        self.min_area = min_area or config.MOTION_MIN_AREA
        self.blur_size = blur_size
        self.threshold = threshold
        self.dilate_iterations = dilate_iterations
        self.learning_rate = learning_rate
        self.warmup_frames = warmup_frames or config.MOTION_WARMUP_FRAMES
        self.var_threshold = var_threshold

        self.frame_count = 0

        # Background subtractor - MOG2 works well for varying lighting
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )
        self.bg_subtractor.setShadowThreshold(0.5)

        # Previous frames for frame differencing
        self.prev_frame = None
        self.prev_prev_frame = None
        self.motion_mask = None

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[MotionRegion], np.ndarray]:
        """
        Detect motion in frame using dual method (background subtraction + frame differencing).

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of:
                - bool: Whether motion was detected
                - List of MotionRegion objects
                - Motion mask (for visualization)
        """
        self.frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Remove shadows (marked as 127 in MOG2)
        fg_mask[fg_mask == 127] = 0

        # During warmup, just build the background model
        if self.frame_count < self.warmup_frames:
            self.prev_prev_frame = self.prev_frame
            self.prev_frame = gray
            return False, [], np.zeros_like(fg_mask)

        # Frame differencing as secondary check
        frame_diff_mask = np.zeros_like(fg_mask)
        if self.prev_frame is not None and self.prev_prev_frame is not None:
            diff1 = cv2.absdiff(gray, self.prev_frame)
            diff2 = cv2.absdiff(self.prev_frame, self.prev_prev_frame)
            _, diff1_thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)
            _, diff2_thresh = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)
            frame_diff_mask = cv2.bitwise_and(diff1_thresh, diff2_thresh)
            frame_diff_mask = cv2.dilate(frame_diff_mask, np.ones((7, 7), np.uint8), iterations=2)

        # Update previous frames
        self.prev_prev_frame = self.prev_frame
        self.prev_frame = gray

        # Combine methods
        if self.prev_prev_frame is not None:
            combined_mask = cv2.bitwise_or(fg_mask, frame_diff_mask)
        else:
            combined_mask = fg_mask

        # Threshold to binary
        _, thresh = cv2.threshold(combined_mask, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=self.dilate_iterations)

        self.motion_mask = thresh.copy()

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        motion_regions = []
        frame_area = frame.shape[0] * frame.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                intensity = min(1.0, area / (frame_area * 0.1))

                motion_regions.append(MotionRegion(
                    bbox=(x, y, x + w, y + h),
                    area=area,
                    intensity=intensity
                ))

        motion_detected = len(motion_regions) > 0

        return motion_detected, motion_regions, thresh

    def check_overlap(
        self,
        detection_bbox: Tuple[int, int, int, int],
        motion_regions: List[MotionRegion],
        min_overlap: float = None
    ) -> bool:
        """
        Check if a detection overlaps with any motion region.
        
        This is the key filtering function:
        - If detection overlaps with motion → real intrusion (alert)
        - If detection has no motion overlap → stationary object (ignore)

        Args:
            detection_bbox: Bounding box of detection (x1, y1, x2, y2)
            motion_regions: List of motion regions
            min_overlap: Minimum overlap ratio to consider as moving

        Returns:
            True if detection overlaps with motion (is actually moving)
        """
        if not motion_regions:
            return False
        
        if min_overlap is None:
            min_overlap = config.MOTION_OVERLAP_THRESHOLD

        dx1, dy1, dx2, dy2 = detection_bbox
        det_area = (dx2 - dx1) * (dy2 - dy1)

        if det_area <= 0:
            return False

        for region in motion_regions:
            mx1, my1, mx2, my2 = region.bbox

            # Calculate intersection
            inter_x1 = max(dx1, mx1)
            inter_y1 = max(dy1, my1)
            inter_x2 = min(dx2, mx2)
            inter_y2 = min(dy2, my2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                overlap_ratio = inter_area / det_area

                if overlap_ratio >= min_overlap:
                    return True

        return False

    def filter_moving_detections(
        self,
        detections: List,
        motion_regions: List[MotionRegion]
    ) -> List:
        """
        Filter detections to only those that are moving.
        
        Args:
            detections: List of Detection objects from object detector
            motion_regions: List of motion regions from motion detection
            
        Returns:
            List of detections that overlap with motion (actually moving)
        """
        if not config.MOTION_DETECTION_ENABLED:
            return detections
            
        moving_detections = []
        for det in detections:
            if self.check_overlap(det.bbox, motion_regions):
                moving_detections.append(det)
        return moving_detections

    def draw_motion(
        self,
        frame: np.ndarray,
        motion_regions: List[MotionRegion],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw motion regions on frame."""
        frame_copy = frame.copy()

        for region in motion_regions:
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

            label = f"Motion: {region.intensity*100:.0f}%"
            cv2.putText(
                frame_copy, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return frame_copy

    def reset(self) -> None:
        """Reset the motion detector (clear background model)."""
        self.frame_count = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=self.var_threshold,
            detectShadows=True
        )
        self.bg_subtractor.setShadowThreshold(0.5)
        self.prev_frame = None
        self.prev_prev_frame = None
        self.motion_mask = None
