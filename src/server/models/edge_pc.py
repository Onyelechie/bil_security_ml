from sqlalchemy import Column, String, DateTime

from .base import Base


class EdgePC(Base):
    __tablename__ = "edge_pcs"
    edge_pc_id = Column(String, primary_key=True)
    site_name = Column(String, nullable=False)
    last_heartbeat = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="offline")
