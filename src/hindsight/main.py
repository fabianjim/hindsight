from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from hindsight.auth import require_service_token
from hindsight.config import get_settings
from hindsight.db import get_engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    get_settings()  # fail fast on missing config
    yield
    await get_engine().dispose()


app = FastAPI(title="hindsight", lifespan=lifespan)

v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_service_token)])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(v1)
