from sqlalchemy import Column, String, DateTime
from .base import Base

class EdgePC(Base):
    __tablename__ = "edge_pcs"
    site_id = Column(String, primary_key=True)
    last_heartbeat = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="offline")
