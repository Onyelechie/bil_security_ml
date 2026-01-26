"""
Event models for the intrusion detection system.
Defines the internal event object structure for normalized motion events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class MotionEvent:
    """
    Internal event object representing a normalized motion event.
    
    Attributes:
        event_id: Unique identifier for the event
        camera_id: Identifier of the camera that triggered the event
        event_time: Timestamp when the event occurred
        event_type: Type of event (default: 'motion')
        raw_data: Optional raw data from the original message
    """
    camera_id: str
    event_time: datetime
    event_type: str = "motion"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_data: Optional[dict] = None
    
    def __str__(self) -> str:
        return (
            f"MotionEvent(\n"
            f"  event_id={self.event_id},\n"
            f"  camera_id={self.camera_id},\n"
            f"  event_time={self.event_time.isoformat()},\n"
            f"  event_type={self.event_type}\n"
            f")"
        )
    
    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "event_time": self.event_time.isoformat(),
            "event_type": self.event_type,
            "raw_data": self.raw_data
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "MotionEvent":
        """
        Create a MotionEvent from a JSON dictionary.
        
        Expected JSON format:
        {
            "camera_id": "camera_01",
            "event_time": "2026-01-21T14:30:00",  # ISO format or Unix timestamp
            "event_type": "motion"  # optional, defaults to "motion"
        }
        """
        # Parse event_time - support both ISO format and Unix timestamp
        event_time_raw = data.get("event_time")
        if event_time_raw is None:
            event_time = datetime.now()
        elif isinstance(event_time_raw, (int, float)):
            event_time = datetime.fromtimestamp(event_time_raw)
        else:
            event_time = datetime.fromisoformat(str(event_time_raw))
        
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            camera_id=data.get("camera_id", "unknown"),
            event_time=event_time,
            event_type=data.get("event_type", "motion"),
            raw_data=data
        )
