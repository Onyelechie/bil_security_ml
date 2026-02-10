
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import settings
from .db import init_db
from .routes.alerts import router as alerts_router
from .routes.heartbeat import router as heartbeat_router


# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title="BIL Security ML Server",
    description="FastAPI server for distributed security camera system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts_router)
app.include_router(heartbeat_router)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "BIL Security ML Server"}
