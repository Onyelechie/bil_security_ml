from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MotionEvent(BaseModel):
    """
    Parsed representation of one motion packet.

    Keep raw_xml for traceability/debugging. This is very useful when something breaks in production.
    """
    site_id: str

    camera_id: int
    camera_name: Optional[str] = None

    policy_id: Optional[int] = None
    policy_name: Optional[str] = None

    user_string: Optional[str] = None

    received_at: datetime
    raw_xml: str = Field(..., description="Original <Action>...</Action> XML payload")
