from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, Text

from .base import Base


class Device(Base):
    __tablename__ = "devices"

    device_id = Column(String(128), primary_key=True, index=True)
    public_key_b64 = Column(Text, nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    revoked_at = Column(DateTime, nullable=True)
    last_key_rotation_at = Column(DateTime, nullable=True)
