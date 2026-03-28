from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from .ml_evaluator import MLEvaluator
from .sender import ServerSender
from .video.ring_buffer import FrameItem

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Connects frame extraction → ML evaluation → alert sending.
    """

    def __init__(
        self,
        evaluator: MLEvaluator,
        sender: ServerSender,
        image_output_dir: Optional[str] = "storage/ws_alert_images",
        save_images: bool = True,
    ):
        self.evaluator = evaluator
        self.sender = sender
        self.image_output_dir = image_output_dir
        self.save_images = save_images

    def process_frames(
        self,
        camera_id: str,
        frames: Union[List[np.ndarray], List[FrameItem]],
        *,
        frame_timestamps: Optional[List[datetime]] = None,
    ) -> None:
        """
        Main pipeline entrypoint.

        :param camera_id: Camera that produced the frames
        :param frames: List of frames (np.ndarray) or FrameItem objects
        :param frame_timestamps: Optional timestamps aligned with frames
        """
        if isinstance(frames, np.ndarray):
            raise ValueError("frames must be a list of frames, not a single ndarray")
        if not frames:
            logger.debug("No frames provided to pipeline")
            return

        timestamps = frame_timestamps
        if timestamps is None and isinstance(frames[0], FrameItem):
            items: Sequence[FrameItem] = frames  # type: ignore[assignment]
            frames = [it.frame for it in items]
            timestamps = [it.ts for it in items]

        if not isinstance(frames, list):
            raise ValueError("frames must be provided as a list")
        if not all(isinstance(f, np.ndarray) for f in frames):
            raise ValueError("frames list must contain numpy arrays")

        result = self.evaluator.evaluate_frames(frames)

        if result is None:
            logger.debug("No valid detection → no alert")
            return

        detection = result["detection"]
        frame = result["frame"]
        frame_index = result.get("frame_index", -1)

        # Convert detection format for sender
        detections_payload = [
            {
                "class": detection["label"],
                "confidence": detection["confidence"],
            }
        ]

        # Save annotated frame to disk
        image_path = self._save_frame(camera_id, frame)

        timestamp = self._select_timestamp(timestamps, frame_index)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Send alert
        success = self.sender.send_alert(
            camera_id=camera_id,
            detections=detections_payload,
            timestamp=timestamp,
            image_path=image_path if image_path else None,
        )

        if success:
            logger.info(
                "ALERT sent: camera=%s class=%s conf=%.2f",
                camera_id,
                detection["label"],
                detection["confidence"],
            )
        else:
            logger.error("Failed to send alert")

    def _save_frame(self, camera_id: str, frame: np.ndarray) -> Optional[str]:
        """
        Save annotated frame as JPEG and return file path.
        """
        if not self.save_images or not self.image_output_dir:
            return None
        if frame is None:
            return None

        os.makedirs(self.image_output_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{timestamp}.jpg"
        path = os.path.join(self.image_output_dir, filename)

        try:
            ok = cv2.imwrite(path, frame)
            return path if ok else None
        except Exception as e:
            logger.error("Failed to save frame: %s", e)
            return None

    @staticmethod
    def _select_timestamp(
        frame_timestamps: Optional[List[datetime]], frame_index: int
    ) -> Optional[datetime]:
        if not frame_timestamps:
            return None
        if 0 <= frame_index < len(frame_timestamps):
            ts = frame_timestamps[frame_index]
            return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        return None
