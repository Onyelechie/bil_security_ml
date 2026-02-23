from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import settings
from .models.base import Base

engine = create_engine(
    settings.database_url,
    connect_args=({"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
