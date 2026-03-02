from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import settings
from .db import init_db
from .routes.alerts import router as alerts_router
from .routes.heartbeat import router as heartbeat_router
from .routes.ws_alerts import router as ws_alerts_router
from .services.image_storage import ImageStorageService
from .services.ws_alert_dispatcher import WebSocketAlertDispatcher
from .services.ws_connection_manager import WebSocketConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    # Security guard: warn if running in non-debug with default secret
    try:
        from .config import settings

        # Avoid embedding a literal default secret in source (Bandit B105).
        # Warn if running in non-debug mode with an empty or placeholder secret.
        insecure_secret_values = {"", "your-secret-key-here"}
        if not settings.debug and settings.secret_key.strip().lower() in insecure_secret_values:
            logger.warning(
                "SECRET_KEY is empty or set to a development placeholder while DEBUG is False. "
                "Set a strong SECRET_KEY in environment or .env before production."
            )
    except Exception:
        # If config can't be imported for some reason, continue and let other errors surface
        logger.debug("Could not verify SECRET_KEY at startup")

    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

    app.state.ws_connection_manager = WebSocketConnectionManager(max_connections=settings.ws_max_connections)
    app.state.ws_alert_dispatcher = WebSocketAlertDispatcher(
        worker_count=settings.ws_alert_worker_count,
        queue_size=settings.ws_alert_queue_size,
    )
    app.state.ws_image_storage = ImageStorageService(settings.ws_image_storage_dir)
    app.state.ws_image_storage.ensure_ready()
    app.state.ws_max_image_bytes = settings.ws_max_image_bytes
    await app.state.ws_alert_dispatcher.start()
    logger.info(
        "WebSocket alert dispatcher started (workers=%s, queue_size=%s, max_image_bytes=%s, image_dir=%s)",
        settings.ws_alert_worker_count,
        settings.ws_alert_queue_size,
        settings.ws_max_image_bytes,
        settings.ws_image_storage_dir,
    )
    try:
        yield
    finally:
        await app.state.ws_alert_dispatcher.stop()
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
    allow_origins=settings.parsed_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts_router)
app.include_router(heartbeat_router)
app.include_router(ws_alerts_router)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "BIL Security ML Server"}
