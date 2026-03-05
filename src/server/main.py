import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db
from .routes.alerts import router as alerts_router
from .routes.dashboard import router as dashboard_router
from .routes.heartbeat import router as heartbeat_router
from .routes.logs import router as logs_router
from .routes.ws_alerts import router as ws_alerts_router
from .routes.ws_dashboard import router as ws_dashboard_router
from .services.dashboard_events import DashboardEventManager
from .services.image_storage import ImageStorageService
from .services.log_buffer import InMemoryLogBuffer, InMemoryLogHandler
from .services.ws_alert_dispatcher import WebSocketAlertDispatcher
from .services.ws_connection_manager import WebSocketConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def _run_ws_image_cleanup(
    *,
    image_storage: ImageStorageService,
    retention_hours: int,
    interval_hours: int,
) -> None:
    interval_seconds = interval_hours * 3600
    while True:
        try:
            removed = await asyncio.to_thread(
                image_storage.cleanup_older_than,
                hours=retention_hours,
            )
            if removed > 0:
                logger.info(
                    "WebSocket image cleanup removed %s file(s) older than %s hour(s)",
                    removed,
                    retention_hours,
                )
        except Exception:  # noqa: BLE001
            logger.exception("WebSocket image cleanup task failed")
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    # Security guard: warn if running in non-debug with default secret
    try:
        from .config import settings

        # Avoid embedding a literal default secret in source (Bandit B105).
        # Warn if running in non-debug mode with an empty or placeholder secret.
        insecure_secret_values = {"", "your-secret-key-here"}
        if (
            not settings.debug
            and settings.secret_key.strip().lower() in insecure_secret_values
        ):
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
    app.state.main_event_loop = asyncio.get_running_loop()
    app.state.dashboard_event_manager = DashboardEventManager()

    app.state.log_buffer = InMemoryLogBuffer(max_entries=settings.log_buffer_max_entries)
    app.state.log_handler = InMemoryLogHandler(app.state.log_buffer)
    app.state.log_handler._is_bil_log_buffer_handler = True  # type: ignore[attr-defined]
    logging.getLogger().addHandler(app.state.log_handler)

    app.state.ws_connection_manager = WebSocketConnectionManager(
        max_connections=settings.ws_max_connections
    )
    app.state.ws_alert_dispatcher = WebSocketAlertDispatcher(
        worker_count=settings.ws_alert_worker_count,
        queue_size=settings.ws_alert_queue_size,
    )
    app.state.ws_image_storage = ImageStorageService(settings.ws_image_storage_dir)
    app.state.ws_image_storage.ensure_ready()
    app.state.ws_max_image_bytes = settings.ws_max_image_bytes
    app.state.ws_image_cleanup_task = asyncio.create_task(
        _run_ws_image_cleanup(
            image_storage=app.state.ws_image_storage,
            retention_hours=settings.ws_image_retention_hours,
            interval_hours=settings.ws_image_cleanup_interval_hours,
        ),
        name="ws-image-cleanup",
    )
    await app.state.ws_alert_dispatcher.start()
    logger.info(
        (
            "WebSocket alert dispatcher started "
            "(workers=%s, queue_size=%s, max_image_bytes=%s, image_dir=%s, "
            "image_retention_hours=%s, image_cleanup_interval_hours=%s)"
        ),
        settings.ws_alert_worker_count,
        settings.ws_alert_queue_size,
        settings.ws_max_image_bytes,
        settings.ws_image_storage_dir,
        settings.ws_image_retention_hours,
        settings.ws_image_cleanup_interval_hours,
    )
    try:
        yield
    finally:
        app.state.ws_image_cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.ws_image_cleanup_task
        await app.state.ws_alert_dispatcher.stop()
        logging.getLogger().removeHandler(app.state.log_handler)
        with suppress(Exception):
            app.state.log_handler.close()
        logger.info("Shutting down application")


app = FastAPI(
    title="BIL Security ML Server",
    description="FastAPI server for distributed security camera system",
    version="0.1.0",
    lifespan=lifespan,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

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
app.include_router(logs_router)
app.include_router(ws_alerts_router)
app.include_router(ws_dashboard_router)
app.include_router(dashboard_router)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "BIL Security ML Server"}
