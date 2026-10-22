from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute

from hindsight import api
from hindsight.auth import require_service_token
from hindsight.config import get_settings
from hindsight.db import get_engine
from hindsight.tracing import init_tracing


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    langfuse = init_tracing(get_settings())
    yield
    if langfuse:
        langfuse.shutdown()
    await get_engine().dispose()


def operation_id(route: APIRoute) -> str:
    return route.name


app = FastAPI(title="hindsight", lifespan=lifespan, generate_unique_id_function=operation_id)

v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_service_token)])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


v1.include_router(api.router)
app.include_router(v1)
