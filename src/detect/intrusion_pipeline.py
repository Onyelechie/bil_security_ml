"""
Intrusion Detection Pipeline

Integrates the event processing pipeline with object detection and motion filtering.
This is the core processing engine that:
1. Receives TCP events
2. Extracts frames from ring buffer
3. Runs motion detection (filters stationary objects)
4. Runs object detection (finds people/vehicles)
5. Filters to only moving detections
6. Saves clips of actual intrusions

Optimized for 10 simultaneous cameras on i5/i7 with 4GB RAM.

Note: This module uses late imports for pipeline components to avoid
circular import issues.
"""

import logging
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Import config directly (no circular import issue)
from . import config as detect_config
from .detector import Detection, create_detector
from .motion import MotionDetector, MotionRegion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_pipeline_components():
    """Late import of pipeline components to avoid circular imports."""
    from pipeline import (
        MultiCameraBufferManager,
        EventClipSaver,
        TimestampedFrame
    )
    return MultiCameraBufferManager, EventClipSaver, TimestampedFrame


def _get_events_components():
    """Late import of event components to avoid circular imports."""
    from events import MotionEvent
    return MotionEvent


@dataclass
class IntrusionResult:
    """Result of intrusion detection analysis on an event clip."""
    event_id: str
    camera_id: str
    event_time: datetime
    
    # Detection results
    has_intrusion: bool = False
    detections: List[Detection] = field(default_factory=list)
    moving_detections: List[Detection] = field(default_factory=list)
    alert_detections: List[Detection] = field(default_factory=list)
    
    # Motion results
    has_motion: bool = False
    motion_regions: List[MotionRegion] = field(default_factory=list)
    
    # Output
    clip_path: Optional[Path] = None
    frames_analyzed: int = 0
    
    def __str__(self) -> str:
        status = "🚨 INTRUSION" if self.has_intrusion else "✅ No intrusion"
        return (
            f"{status}\n"
            f"  Camera: {self.camera_id}\n"
            f"  Time: {self.event_time}\n"
            f"  Detections: {len(self.detections)} total, "
            f"{len(self.moving_detections)} moving, "
            f"{len(self.alert_detections)} alerts\n"
            f"  Motion: {self.has_motion} ({len(self.motion_regions)} regions)\n"
            f"  Clip: {self.clip_path}"
        )


class IntrusionDetectionPipeline:
    """
    Main intrusion detection pipeline.
    
    Combines:
    - Frame buffer management (per camera)
    - Motion detection (false alarm filtering)
    - Object detection (person/vehicle detection)
    - Event clip saving
    
    Architecture for 10 cameras:
    - Each camera has its own ring buffer
    - Detection runs on event trigger only (not continuous)
    - Motion filtering reduces unnecessary object detection calls
    - Model selection based on number of cameras
    """
    
    def __init__(
        self,
        model_type: str = None,
        num_cameras: int = 10,
        output_dir: str = "event_clips",
        save_format: str = "video",
        before_seconds: float = 2.0,
        after_seconds: float = 5.0
    ):
        """
        Initialize the intrusion detection pipeline.
        
        Args:
            model_type: Detection model ('mobilenet', 'yolov8n', etc.)
                       If None, auto-selects based on num_cameras.
            num_cameras: Number of cameras to support (affects model choice)
            output_dir: Directory for saving event clips
            save_format: 'video' (mp4) or 'images' (folder)
            before_seconds: Seconds before event to capture
            after_seconds: Seconds after event to capture
        """
        self.num_cameras = num_cameras
        self.before_seconds = before_seconds
        self.after_seconds = after_seconds
        
        # Select model based on camera count if not specified
        if model_type is None:
            model_type = detect_config.get_recommended_model(num_cameras)
            logger.info(f"Auto-selected model '{model_type}' for {num_cameras} cameras")
        
        self.model_type = model_type
        
        # Late import pipeline components
        MultiCameraBufferManager, EventClipSaver, _ = _get_pipeline_components()
        
        # Initialize components
        self.buffer_manager = MultiCameraBufferManager(
            max_duration_seconds=detect_config.FRAME_BUFFER_SECONDS,
            max_frames_per_camera=detect_config.MAX_FRAMES_PER_BUFFER
        )
        
        self.clip_saver = EventClipSaver(
            output_dir=output_dir,
            save_format=save_format
        )
        
        # Create detector (lazy loaded on first use)
        self._detector = None
        
        # Motion detectors per camera
        self._motion_detectors: dict[str, MotionDetector] = {}
        
        # Stats
        self.events_processed = 0
        self.intrusions_detected = 0
        self.false_alarms_filtered = 0
    
    @property
    def detector(self):
        """Lazy-load detector on first use."""
        if self._detector is None:
            logger.info(f"Loading detector: {self.model_type}")
            self._detector = create_detector(self.model_type)
        return self._detector
    
    def get_motion_detector(self, camera_id: str) -> MotionDetector:
        """Get or create motion detector for a camera."""
        if camera_id not in self._motion_detectors:
            self._motion_detectors[camera_id] = MotionDetector()
        return self._motion_detectors[camera_id]
    
    def add_frame(self, camera_id: str, frame: Any) -> None:
        """
        Add a frame to the buffer for a camera.
        
        Args:
            camera_id: Camera identifier
            frame: TimestampedFrame object
        """
        # Add to ring buffer only - motion detection happens during analysis
        self.buffer_manager.add_frame(camera_id, frame)
    
    def analyze_frames(
        self,
        frames: List[Any],
        camera_id: str
    ) -> Tuple[List[Detection], List[MotionRegion], bool]:
        """
        Analyze a list of frames for intrusion.
        
        Strategy:
        1. Run motion detection on all frames
        2. If motion detected, run object detection
        3. Filter detections to only moving objects
        4. Check for alert classes (person/vehicle)
        
        Args:
            frames: List of frames to analyze
            camera_id: Camera identifier
            
        Returns:
            Tuple of (all_detections, motion_regions, has_intrusion)
        """
        all_detections = []
        all_motion_regions = []
        has_motion = False
        
        # Create fresh motion detector for this analysis batch
        # This ensures consistent frame sizes within the analysis
        motion_detector = MotionDetector()
        
        # Analyze every Nth frame (configurable)
        frame_skip = detect_config.DETECTION_FRAME_SKIP
        
        for i, frame_obj in enumerate(frames):
            if i % frame_skip != 0:
                continue
            
            frame = frame_obj.frame
            
            # Resize for detection if configured
            if detect_config.RESIZE_FOR_DETECTION:
                h, w = frame.shape[:2]
                target_w, target_h = detect_config.DETECTION_RESIZE
                if w != target_w or h != target_h:
                    frame = cv2.resize(frame, (target_w, target_h))
            
            # Motion detection
            motion_detected, motion_regions, _ = motion_detector.detect(frame)
            if motion_detected:
                has_motion = True
                all_motion_regions.extend(motion_regions)
            
            # Object detection (only if motion detected to save CPU)
            if motion_detected:
                detections = self.detector.detect(frame, use_tiled=False)
                
                # Filter to alert classes
                alert_detections = [d for d in detections if d.is_alert_class()]
                
                # Filter to moving detections
                moving_detections = motion_detector.filter_moving_detections(
                    alert_detections, motion_regions
                )
                
                all_detections.extend(moving_detections)
        
        # Determine if this is an intrusion
        has_intrusion = len(all_detections) > 0
        
        return all_detections, all_motion_regions, has_intrusion
    
    def process_event(
        self,
        event: Any,
        wait_for_future_frames: bool = True
    ) -> IntrusionResult:
        """
        Process a motion event and determine if it's a real intrusion.
        
        Args:
            event: The MotionEvent from TCP
            wait_for_future_frames: Whether to wait for after_seconds of frames
            
        Returns:
            IntrusionResult with detection details
        """
        import time
        
        self.events_processed += 1
        
        result = IntrusionResult(
            event_id=event.event_id,
            camera_id=event.camera_id,
            event_time=event.event_time
        )
        
        # Wait for future frames if needed
        if wait_for_future_frames:
            logger.info(f"Waiting {self.after_seconds}s for future frames...")
            time.sleep(self.after_seconds + 0.5)
        
        # Extract frames from buffer
        frame_window = self.buffer_manager.get_frames_for_event(
            camera_id=event.camera_id,
            event_time=event.event_time,
            before_seconds=self.before_seconds,
            after_seconds=self.after_seconds
        )
        
        if frame_window is None or not frame_window.frames:
            logger.warning(f"No frames found for event {event.event_id}")
            return result
        
        result.frames_analyzed = len(frame_window.frames)
        
        # Analyze frames
        detections, motion_regions, has_intrusion = self.analyze_frames(
            frame_window.frames, event.camera_id
        )
        
        result.detections = detections
        result.moving_detections = detections  # Already filtered
        result.alert_detections = [d for d in detections if d.is_alert_class()]
        result.motion_regions = motion_regions
        result.has_motion = len(motion_regions) > 0
        result.has_intrusion = has_intrusion
        
        # Save clip if intrusion detected
        if has_intrusion:
            self.intrusions_detected += 1
            clip_path = self.clip_saver.save(frame_window, event.event_id)
            result.clip_path = clip_path
            logger.info(f"🚨 INTRUSION DETECTED: {result}")
        else:
            self.false_alarms_filtered += 1
            logger.info(f"✅ False alarm filtered: no intrusion for event {event.event_id}")
        
        return result
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "events_processed": self.events_processed,
            "intrusions_detected": self.intrusions_detected,
            "false_alarms_filtered": self.false_alarms_filtered,
            "false_alarm_rate": (
                self.false_alarms_filtered / self.events_processed 
                if self.events_processed > 0 else 0
            ),
            "model": self.model_type,
            "num_cameras": self.num_cameras,
            "buffer_stats": self.buffer_manager.get_buffer_stats()
        }
    
    def change_model(self, model_type: str) -> None:
        """
        Change the detection model at runtime.
        
        Useful for frontend model selection.
        
        Args:
            model_type: New model type ('mobilenet', 'yolov8n', etc.)
        """
        logger.info(f"Changing model from {self.model_type} to {model_type}")
        self.model_type = model_type
        self._detector = None  # Will lazy-load on next use
