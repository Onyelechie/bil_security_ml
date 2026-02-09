from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Detection(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float

class AlertCreate(BaseModel):
    site_id: str
    camera_id: str
    timestamp: datetime
    detections: List[Detection]
    image_path: Optional[str] = None

class AlertOut(AlertCreate):
    id: str
