from sqlalchemy import Column, String, DateTime

from .base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"
    version = Column(String, primary_key=True)
    uploaded_at = Column(DateTime, nullable=False)
    file_path = Column(String, nullable=False)
