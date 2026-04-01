from datetime import datetime
from typing import List, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_serializer,
                      field_validator)

from bil_time import ensure_winnipeg, isoformat_winnipeg


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(..., alias="class")
    confidence: float


class AlertCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    site_id: str
    camera_id: str
    edge_pc_id: Optional[str] = None
    timestamp: datetime
    detections: List[Detection]
    image_path: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_winnipeg(value)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return isoformat_winnipeg(value)


class AlertOut(AlertCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str


# Heartbeat schemas
class HeartbeatIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    edge_pc_id: str
    site_name: str
    status: str
    timestamp: datetime
    site_id: Optional[str] = None
    uptime_seconds: Optional[int] = None


class HeartbeatOut(BaseModel):
    edge_pc_id: str
    site_name: str
    status: str
    last_heartbeat: datetime
    message: str


class EdgePCStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    edge_pc_id: str
    site_name: str
    status: str
    last_heartbeat: Optional[datetime] = None


class EdgePCStatusListOut(BaseModel):
    edges: List[EdgePCStatusOut]


class ServerLogEntryOut(BaseModel):
    id: int
    timestamp: datetime
    level: str
    logger: str
    message: str


class ServerLogListOut(BaseModel):
    logs: List[ServerLogEntryOut]
    latest_id: int
