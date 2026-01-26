"""
Video Source Module

Provides video input from files for testing the intrusion detection pipeline.
Supports MP4, AVI, and other formats that OpenCV can read.

Usage:
    source = VideoFileSource("path/to/video.mp4")
    source.start(on_frame_callback)
    
    # Or use VideoSourceManager for multiple cameras
    manager = VideoSourceManager()
    manager.add_source("cam_001", "videos/camera1.mp4")
    manager.add_source("cam_002", "videos/camera2.mp4")
    manager.start_all(on_frame_callback)
"""

import cv2
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimestampedFrame:
    """A frame with its capture timestamp."""
    frame: any  # numpy array (cv2 image)
    timestamp: datetime
    frame_number: int
    
    @property
    def age_seconds(self) -> float:
        """How old is this frame in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class VideoFileSource:
    """
    Reads frames from a video file and delivers them at real-time speed.
    
    Simulates a camera by reading video files at their natural FPS,
    useful for testing detection pipeline with recorded footage.
    """
    
    def __init__(
        self,
        video_path: str,
        camera_id: str = "video_001",
        loop: bool = True,
        playback_speed: float = 1.0
    ):
        """
        Initialize video file source.
        
        Args:
            video_path: Path to video file (mp4, avi, etc.)
            camera_id: Identifier for this video source
            loop: Whether to loop the video when it ends
            playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        self.video_path = Path(video_path)
        self.camera_id = camera_id
        self.loop = loop
        self.playback_speed = playback_speed
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video to get properties
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        cap.release()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._current_frame = 0
        
        logger.info(
            f"VideoFileSource '{camera_id}': {self.video_path.name} "
            f"({self.width}x{self.height} @ {self.fps:.1f}fps, "
            f"{self.duration:.1f}s, {self.frame_count} frames)"
        )
    
    def start(self, on_frame: Callable[[TimestampedFrame], None]) -> None:
        """
        Start reading video frames.
        
        Args:
            on_frame: Callback function called for each frame
        """
        if self._running:
            logger.warning(f"VideoFileSource {self.camera_id} already running")
            return
        
        self._callback = on_frame
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started VideoFileSource: {self.camera_id}")
    
    def stop(self) -> None:
        """Stop reading video frames."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(f"Stopped VideoFileSource: {self.camera_id}")
    
    def _read_loop(self) -> None:
        """Internal loop that reads frames at real-time speed."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return
        
        frame_delay = (1.0 / self.fps) / self.playback_speed
        self._current_frame = 0
        
        while self._running:
            frame_start = time.time()
            
            ret, frame = cap.read()
            
            if not ret:
                if self.loop:
                    # Reset to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame = 0
                    logger.debug(f"Looping video: {self.camera_id}")
                    continue
                else:
                    logger.info(f"Video ended: {self.camera_id}")
                    break
            
            self._current_frame += 1
            
            # Create timestamped frame
            ts_frame = TimestampedFrame(
                frame=frame,
                timestamp=datetime.now(),
                frame_number=self._current_frame
            )
            
            # Deliver frame
            if self._callback:
                try:
                    self._callback(ts_frame)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")
            
            # Maintain real-time playback speed
            elapsed = time.time() - frame_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()
    
    @property
    def is_running(self) -> bool:
        """Check if video source is running."""
        return self._running
    
    @property
    def progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if self.frame_count == 0:
            return 0.0
        return self._current_frame / self.frame_count


class VideoSourceManager:
    """
    Manages multiple video sources (simulating multiple cameras).
    
    Useful for testing multi-camera intrusion detection with video files.
    """
    
    def __init__(self):
        self.sources: Dict[str, VideoFileSource] = {}
    
    def add_source(
        self,
        camera_id: str,
        video_path: str,
        loop: bool = True,
        playback_speed: float = 1.0
    ) -> VideoFileSource:
        """
        Add a video source.
        
        Args:
            camera_id: Unique identifier for this source
            video_path: Path to video file
            loop: Whether to loop when video ends
            playback_speed: Playback speed multiplier
            
        Returns:
            The created VideoFileSource
        """
        source = VideoFileSource(
            video_path=video_path,
            camera_id=camera_id,
            loop=loop,
            playback_speed=playback_speed
        )
        self.sources[camera_id] = source
        return source
    
    def get_source(self, camera_id: str) -> Optional[VideoFileSource]:
        """Get a source by ID."""
        return self.sources.get(camera_id)
    
    def start_all(
        self,
        on_frame: Callable[[str, TimestampedFrame], None]
    ) -> None:
        """
        Start all video sources.
        
        Args:
            on_frame: Callback(camera_id, frame) for each frame
        """
        for camera_id, source in self.sources.items():
            # Create camera-specific callback
            source.start(lambda f, cid=camera_id: on_frame(cid, f))
    
    def stop_all(self) -> None:
        """Stop all video sources."""
        for source in self.sources.values():
            source.stop()
    
    def list_sources(self) -> list:
        """List all source camera IDs."""
        return list(self.sources.keys())
    
    def get_stats(self) -> dict:
        """Get stats for all sources."""
        return {
            camera_id: {
                "video": source.video_path.name,
                "fps": source.fps,
                "resolution": f"{source.width}x{source.height}",
                "duration": f"{source.duration:.1f}s",
                "running": source.is_running,
                "progress": f"{source.progress * 100:.1f}%"
            }
            for camera_id, source in self.sources.items()
        }


def find_videos(directory: str, extensions: list = None) -> list:
    """
    Find all video files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of extensions (default: mp4, avi, mov, mkv)
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    videos = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return videos
    
    for ext in extensions:
        videos.extend(dir_path.glob(f'*{ext}'))
        videos.extend(dir_path.glob(f'*{ext.upper()}'))
    
    return sorted(videos)
