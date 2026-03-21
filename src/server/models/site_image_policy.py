from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from .base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SiteImagePolicy(Base):
    __tablename__ = "site_image_policies"

    site_id = Column(String, primary_key=True)
    retention_hours = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utc_now)
    updated_at = Column(DateTime, nullable=False, default=_utc_now)
