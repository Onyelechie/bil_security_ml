import uuid

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String

from bil_time import now_in_winnipeg

from .base import Base


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    site_id = Column(String, nullable=False)
    camera_id = Column(String, nullable=False)
    edge_pc_id = Column(
        String,
        ForeignKey("edge_pcs.edge_pc_id", ondelete="RESTRICT"),
        nullable=False,
    )
    timestamp = Column(DateTime, nullable=False)
    received_at = Column(DateTime, nullable=True, default=now_in_winnipeg)
    detections = Column(JSON, nullable=False)
    image_path = Column(String, nullable=True)
