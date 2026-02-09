

from fastapi import FastAPI
from contextlib import asynccontextmanager
from .db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


from .routes.alerts import router as alerts_router
from .routes.heartbeat import router as heartbeat_router

app = FastAPI(lifespan=lifespan)
app.include_router(alerts_router)
app.include_router(heartbeat_router)

@app.get("/")
def health_check():
    return {"status": "ok"}
