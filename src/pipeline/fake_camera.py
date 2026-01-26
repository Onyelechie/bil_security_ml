"""
Fake camera source for testing the pipeline without real RTSP streams.
Generates synthetic frames or reads from a local video file.
"""

import time
import threading
from datetime import datetime
from typing import Optional, Callable, Generator
from dataclasses import dataclass
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class TimestampedFrame:
    """A frame with its associated timestamp."""
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    
    @property
    def timestamp_seconds(self) -> float:
        """Get timestamp as Unix seconds."""
        return self.timestamp.timestamp()


class FakeCamera:
    """
    A fake camera source that generates frames for testing.
    
    Can either generate synthetic frames with timestamps or read from a video file.
    """
    
    def __init__(
        self,
        camera_id: str,
        fps: int = 15,
        resolution: tuple[int, int] = (640, 480),
        video_file: Optional[str] = None
    ):
        """
        Initialize the fake camera.
        
        Args:
            camera_id: Unique identifier for this camera
            fps: Frames per second (for synthetic frames or playback speed)
            resolution: Frame resolution as (width, height)
            video_file: Optional path to a video file to read from
        """
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.video_file = video_file
        
        self._running = False
        self._frame_number = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._on_frame: Optional[Callable[[TimestampedFrame], None]] = None
        self._thread: Optional[threading.Thread] = None
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic test frame with timestamp overlay."""
        # Create a frame with some visual content
        width, height = self.resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            frame[y, :, 0] = int(255 * y / height)  # Blue gradient
            frame[y, :, 2] = int(255 * (1 - y / height))  # Red gradient
        
        # Add some "motion" - a moving rectangle
        box_size = 50
        x_pos = int((self._frame_number * 5) % (width - box_size))
        y_pos = int(height / 2 - box_size / 2)
        frame[y_pos:y_pos+box_size, x_pos:x_pos+box_size] = [0, 255, 0]  # Green box
        
        if CV2_AVAILABLE:
            # Add timestamp text
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(
                frame, 
                f"Camera: {self.camera_id}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            cv2.putText(
                frame, 
                timestamp_str, 
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            cv2.putText(
                frame, 
                f"Frame: {self._frame_number}", 
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        return frame
    
    def _read_video_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the video file."""
        if self._capture is None:
            return None
            
        ret, frame = self._capture.read()
        if not ret:
            # Loop the video
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._capture.read()
            if not ret:
                return None
        
        # Resize if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)
        
        return frame
    
    def _get_next_frame(self) -> Optional[TimestampedFrame]:
        """Get the next frame (synthetic or from video)."""
        if self.video_file and CV2_AVAILABLE:
            frame = self._read_video_frame()
        else:
            frame = self._generate_synthetic_frame()
        
        if frame is None:
            return None
        
        self._frame_number += 1
        
        return TimestampedFrame(
            frame=frame,
            timestamp=datetime.now(),
            frame_number=self._frame_number
        )
    
    def frames(self) -> Generator[TimestampedFrame, None, None]:
        """
        Generator that yields frames continuously.
        
        Yields:
            TimestampedFrame objects at the configured FPS
        """
        frame_interval = 1.0 / self.fps
        
        if self.video_file and CV2_AVAILABLE:
            self._capture = cv2.VideoCapture(self.video_file)
        
        self._running = True
        last_frame_time = time.time()
        
        try:
            while self._running:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed >= frame_interval:
                    frame = self._get_next_frame()
                    if frame:
                        yield frame
                    last_frame_time = current_time
                else:
                    # Sleep for remaining time
                    time.sleep(frame_interval - elapsed)
        finally:
            if self._capture:
                self._capture.release()
                self._capture = None
    
    def start(self, on_frame: Callable[[TimestampedFrame], None]) -> None:
        """
        Start the camera in a background thread.
        
        Args:
            on_frame: Callback function invoked for each new frame
        """
        self._on_frame = on_frame
        self._running = True
        
        def run():
            for frame in self.frames():
                if not self._running:
                    break
                if self._on_frame:
                    self._on_frame(frame)
        
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    @property
    def is_running(self) -> bool:
        """Check if the camera is running."""
        return self._running


class FakeCameraManager:
    """Manages multiple fake cameras."""
    
    def __init__(self):
        self.cameras: dict[str, FakeCamera] = {}
    
    def add_camera(
        self,
        camera_id: str,
        fps: int = 15,
        resolution: tuple[int, int] = (640, 480),
        video_file: Optional[str] = None
    ) -> FakeCamera:
        """Add a new fake camera."""
        camera = FakeCamera(
            camera_id=camera_id,
            fps=fps,
            resolution=resolution,
            video_file=video_file
        )
        self.cameras[camera_id] = camera
        return camera
    
    def get_camera(self, camera_id: str) -> Optional[FakeCamera]:
        """Get a camera by ID."""
        return self.cameras.get(camera_id)
    
    def start_all(self, on_frame: Callable[[str, TimestampedFrame], None]) -> None:
        """Start all cameras with a unified callback."""
        for camera_id, camera in self.cameras.items():
            camera.start(lambda f, cid=camera_id: on_frame(cid, f))
    
    def stop_all(self) -> None:
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()
