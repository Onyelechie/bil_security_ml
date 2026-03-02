from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class MotionEvent:
    received_at_utc: datetime
    site_id: str
    edge_pc_id: str

    camera_id: str | None = None
    camera_name: str | None = None
    policy_id: str | None = None
    policy_name: str | None = None
    user_string: str | None = None
    raw_xml: str | None = None
    source: str = "tcp"
