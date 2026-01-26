"""
Event Frame Window Selector and Clip Saver.
Extracts frames around events and saves them as image folders or video clips.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .frame_buffer import FrameWindow

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventClipSaver:
    """
    Saves event frame windows to disk as image folders or video clips.
    """
    
    def __init__(
        self,
        output_dir: str = "event_clips",
        save_format: str = "images",  # "images" or "video"
        video_codec: str = "mp4v",
        image_format: str = "jpg",
        jpeg_quality: int = 95
    ):
        """
        Initialize the clip saver.
        
        Args:
            output_dir: Base directory for saving clips
            save_format: "images" for folder of images, "video" for mp4 clip
            video_codec: FourCC codec for video (default: mp4v)
            image_format: Image format for saving frames (jpg, png)
            jpeg_quality: JPEG quality (0-100)
        """
        self.output_dir = Path(output_dir)
        self.save_format = save_format
        self.video_codec = video_codec
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_clip_name(
        self,
        camera_id: str,
        event_time: datetime,
        event_id: Optional[str] = None
    ) -> str:
        """Generate a unique name for the clip folder/file."""
        timestamp_str = event_time.strftime("%Y%m%d_%H%M%S")
        if event_id:
            return f"{camera_id}_{timestamp_str}_{event_id[:8]}"
        return f"{camera_id}_{timestamp_str}"
    
    def save_as_images(
        self,
        frame_window: FrameWindow,
        event_id: Optional[str] = None
    ) -> Optional[Path]:
        """
        Save frame window as a folder of images.
        
        Args:
            frame_window: The frame window to save
            event_id: Optional event ID for naming
            
        Returns:
            Path to the created folder, or None if failed
        """
        if not frame_window.frames:
            logger.warning("No frames to save")
            return None
        
        clip_name = self._generate_clip_name(
            frame_window.camera_id,
            frame_window.start_time,
            event_id
        )
        clip_dir = self.output_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {frame_window.frame_count} frames to {clip_dir}")
        
        for i, frame_obj in enumerate(frame_window.frames):
            # Generate filename with timestamp
            ts_str = frame_obj.timestamp.strftime("%H%M%S_%f")[:-3]
            filename = f"frame_{i:04d}_{ts_str}.{self.image_format}"
            filepath = clip_dir / filename
            
            if CV2_AVAILABLE:
                if self.image_format == "jpg":
                    cv2.imwrite(
                        str(filepath),
                        frame_obj.frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                else:
                    cv2.imwrite(str(filepath), frame_obj.frame)
            else:
                # Fallback: save as raw numpy array
                np.save(str(filepath.with_suffix('.npy')), frame_obj.frame)
        
        # Save metadata
        metadata_path = clip_dir / "metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"camera_id: {frame_window.camera_id}\n")
            f.write(f"event_id: {event_id or 'N/A'}\n")
            f.write(f"start_time: {frame_window.start_time.isoformat()}\n")
            f.write(f"end_time: {frame_window.end_time.isoformat()}\n")
            f.write(f"duration_seconds: {frame_window.duration_seconds:.2f}\n")
            f.write(f"frame_count: {frame_window.frame_count}\n")
        
        logger.info(f"Saved clip to {clip_dir}")
        return clip_dir
    
    def save_as_video(
        self,
        frame_window: FrameWindow,
        event_id: Optional[str] = None,
        fps: int = 15
    ) -> Optional[Path]:
        """
        Save frame window as an MP4 video clip.
        
        Args:
            frame_window: The frame window to save
            event_id: Optional event ID for naming
            fps: Frames per second for the video
            
        Returns:
            Path to the created video file, or None if failed
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available, cannot save video")
            return None
        
        if not frame_window.frames:
            logger.warning("No frames to save")
            return None
        
        clip_name = self._generate_clip_name(
            frame_window.camera_id,
            frame_window.start_time,
            event_id
        )
        video_path = self.output_dir / f"{clip_name}.mp4"
        
        # Get frame dimensions from first frame
        height, width = frame_window.frames[0].frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            logger.error(f"Failed to create video writer for {video_path}")
            return None
        
        logger.info(f"Saving {frame_window.frame_count} frames as video to {video_path}")
        
        for frame_obj in frame_window.frames:
            writer.write(frame_obj.frame)
        
        writer.release()
        
        logger.info(f"Saved video clip to {video_path}")
        return video_path
    
    def save(
        self,
        frame_window: FrameWindow,
        event_id: Optional[str] = None,
        fps: int = 15
    ) -> Optional[Path]:
        """
        Save frame window using the configured format.
        
        Args:
            frame_window: The frame window to save
            event_id: Optional event ID for naming
            fps: Frames per second (for video format)
            
        Returns:
            Path to the saved clip (folder or file)
        """
        if self.save_format == "video":
            return self.save_as_video(frame_window, event_id, fps)
        else:
            return self.save_as_images(frame_window, event_id)


class EventFrameWindowSelector:
    """
    Coordinates frame extraction and saving for motion events.
    
    This is the main interface for the event processing pipeline.
    """
    
    def __init__(
        self,
        buffer_manager,  # MultiCameraBufferManager
        clip_saver: Optional[EventClipSaver] = None,
        before_seconds: float = 2.0,
        after_seconds: float = 5.0
    ):
        """
        Initialize the window selector.
        
        Args:
            buffer_manager: The multi-camera buffer manager
            clip_saver: Optional clip saver (creates default if not provided)
            before_seconds: Seconds before event to include
            after_seconds: Seconds after event to include
        """
        self.buffer_manager = buffer_manager
        self.clip_saver = clip_saver or EventClipSaver()
        self.before_seconds = before_seconds
        self.after_seconds = after_seconds
    
    def process_event(
        self,
        camera_id: str,
        event_time: datetime,
        event_id: Optional[str] = None,
        wait_for_future_frames: bool = True
    ) -> Optional[Path]:
        """
        Process a motion event and save the corresponding clip.
        
        Args:
            camera_id: ID of the camera that triggered the event
            event_time: Timestamp of the event
            event_id: Optional event ID for naming
            wait_for_future_frames: If True, wait for after_seconds of frames
            
        Returns:
            Path to the saved clip, or None if failed
        """
        import time
        
        if wait_for_future_frames:
            # Wait for future frames to accumulate
            logger.info(f"Waiting {self.after_seconds}s for future frames...")
            time.sleep(self.after_seconds + 0.5)  # Small buffer
        
        # Extract frames from buffer
        frame_window = self.buffer_manager.get_frames_for_event(
            camera_id=camera_id,
            event_time=event_time,
            before_seconds=self.before_seconds,
            after_seconds=self.after_seconds
        )
        
        if frame_window is None:
            logger.warning(f"No buffer found for camera {camera_id}")
            return None
        
        if not frame_window.frames:
            logger.warning(f"No frames found in window for event at {event_time}")
            return None
        
        logger.info(
            f"Extracted {frame_window.frame_count} frames "
            f"({frame_window.duration_seconds:.2f}s) for camera {camera_id}"
        )
        
        # Save the clip
        return self.clip_saver.save(frame_window, event_id)
