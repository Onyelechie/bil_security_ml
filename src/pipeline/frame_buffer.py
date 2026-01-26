"""
Ring buffer for storing timestamped frames.
Allows efficient extraction of frames within a time window around events.
"""

import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from .fake_camera import TimestampedFrame


@dataclass
class FrameWindow:
    """A collection of frames extracted from the ring buffer."""
    frames: list[TimestampedFrame]
    start_time: datetime
    end_time: datetime
    camera_id: str
    
    @property
    def duration_seconds(self) -> float:
        """Get the duration of the window in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def frame_count(self) -> int:
        """Get the number of frames in the window."""
        return len(self.frames)


class FrameRingBuffer:
    """
    Thread-safe ring buffer for storing timestamped frames.
    
    Maintains a sliding window of frames based on time duration,
    allowing extraction of frames around event timestamps.
    """
    
    def __init__(
        self,
        camera_id: str,
        max_duration_seconds: float = 30.0,
        max_frames: int = 500
    ):
        """
        Initialize the ring buffer.
        
        Args:
            camera_id: Identifier for the camera this buffer belongs to
            max_duration_seconds: Maximum time span to keep in buffer (default: 30s)
            max_frames: Maximum number of frames to store (safety limit)
        """
        self.camera_id = camera_id
        self.max_duration_seconds = max_duration_seconds
        self.max_frames = max_frames
        
        self._buffer: deque[TimestampedFrame] = deque(maxlen=max_frames)
        self._lock = threading.RLock()
    
    def add_frame(self, frame: TimestampedFrame) -> None:
        """
        Add a frame to the buffer.
        
        Old frames outside the time window are automatically removed.
        """
        with self._lock:
            self._buffer.append(frame)
            self._cleanup_old_frames()
    
    def _cleanup_old_frames(self) -> None:
        """Remove frames older than max_duration_seconds."""
        if not self._buffer:
            return
        
        cutoff_time = datetime.now() - timedelta(seconds=self.max_duration_seconds)
        
        while self._buffer and self._buffer[0].timestamp < cutoff_time:
            self._buffer.popleft()
    
    def get_frames_in_window(
        self,
        event_time: datetime,
        before_seconds: float = 2.0,
        after_seconds: float = 5.0
    ) -> FrameWindow:
        """
        Extract frames within a time window around an event.
        
        Args:
            event_time: The timestamp of the event
            before_seconds: Seconds before the event to include
            after_seconds: Seconds after the event to include
            
        Returns:
            FrameWindow containing the extracted frames
        """
        start_time = event_time - timedelta(seconds=before_seconds)
        end_time = event_time + timedelta(seconds=after_seconds)
        
        with self._lock:
            frames = [
                f for f in self._buffer
                if start_time <= f.timestamp <= end_time
            ]
        
        return FrameWindow(
            frames=frames,
            start_time=start_time,
            end_time=end_time,
            camera_id=self.camera_id
        )
    
    def get_latest_frame(self) -> Optional[TimestampedFrame]:
        """Get the most recent frame in the buffer."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None
    
    def get_oldest_frame(self) -> Optional[TimestampedFrame]:
        """Get the oldest frame in the buffer."""
        with self._lock:
            if self._buffer:
                return self._buffer[0]
            return None
    
    @property
    def frame_count(self) -> int:
        """Get the current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def buffer_duration(self) -> float:
        """Get the current time span of frames in the buffer (in seconds)."""
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            oldest = self._buffer[0].timestamp
            newest = self._buffer[-1].timestamp
            return (newest - oldest).total_seconds()
    
    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()


class MultiCameraBufferManager:
    """Manages ring buffers for multiple cameras."""
    
    def __init__(
        self,
        max_duration_seconds: float = 30.0,
        max_frames_per_camera: int = 500
    ):
        """
        Initialize the buffer manager.
        
        Args:
            max_duration_seconds: Maximum time span per buffer
            max_frames_per_camera: Maximum frames per camera buffer
        """
        self.max_duration_seconds = max_duration_seconds
        self.max_frames_per_camera = max_frames_per_camera
        self._buffers: dict[str, FrameRingBuffer] = {}
        self._lock = threading.Lock()
    
    def get_or_create_buffer(self, camera_id: str) -> FrameRingBuffer:
        """Get or create a buffer for a camera."""
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = FrameRingBuffer(
                    camera_id=camera_id,
                    max_duration_seconds=self.max_duration_seconds,
                    max_frames=self.max_frames_per_camera
                )
            return self._buffers[camera_id]
    
    def add_frame(self, camera_id: str, frame: TimestampedFrame) -> None:
        """Add a frame to the appropriate camera buffer."""
        buffer = self.get_or_create_buffer(camera_id)
        buffer.add_frame(frame)
    
    def get_frames_for_event(
        self,
        camera_id: str,
        event_time: datetime,
        before_seconds: float = 2.0,
        after_seconds: float = 5.0
    ) -> Optional[FrameWindow]:
        """
        Get frames from a specific camera around an event time.
        
        Returns None if no buffer exists for the camera.
        """
        with self._lock:
            buffer = self._buffers.get(camera_id)
        
        if buffer is None:
            return None
        
        return buffer.get_frames_in_window(
            event_time=event_time,
            before_seconds=before_seconds,
            after_seconds=after_seconds
        )
    
    def get_buffer_stats(self) -> dict[str, dict]:
        """Get statistics for all buffers."""
        stats = {}
        with self._lock:
            for camera_id, buffer in self._buffers.items():
                stats[camera_id] = {
                    "frame_count": buffer.frame_count,
                    "buffer_duration": buffer.buffer_duration
                }
        return stats
