from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime

class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(..., alias="class")
    confidence: float

class AlertCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    site_id: str
    camera_id: str
    timestamp: datetime
    detections: List[Detection]
    image_path: Optional[str] = None

class AlertOut(AlertCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str


# Heartbeat schemas
class HeartbeatIn(BaseModel):
    edge_pc_id: str
    site_name: str
    status: str
    timestamp: datetime

class HeartbeatOut(BaseModel):
    edge_pc_id: str
    site_name: str
    status: str
    last_heartbeat: datetime
    message: str
