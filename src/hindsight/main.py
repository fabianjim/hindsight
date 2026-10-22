from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute

from hindsight.auth import require_service_token
from hindsight.config import get_settings
from hindsight.db import get_engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    get_settings()  # fail fast on missing config
    yield
    await get_engine().dispose()


def operation_id(route: APIRoute) -> str:
    return route.name


app = FastAPI(title="hindsight", lifespan=lifespan, generate_unique_id_function=operation_id)

v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_service_token)])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(v1)
