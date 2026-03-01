from sqlalchemy import Column, String, DateTime, JSON, ForeignKey
import uuid

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
    detections = Column(JSON, nullable=False)
    image_path = Column(String, nullable=True)
