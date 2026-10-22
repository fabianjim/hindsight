from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace.export import SpanExporter

from hindsight.config import Settings, get_settings

REDACTED = "[redacted]"

_capture_content: ContextVar[bool] = ContextVar("capture_content", default=False)


@contextmanager
def capture_content() -> Generator[None]:
    # only for demo and eval runs
    token = _capture_content.set(True)
    try:
        yield
    finally:
        _capture_content.reset(token)


def mask(*, data: Any, **_: Any) -> Any:
    return data if data is None or _capture_content.get() else REDACTED


def init_tracing(settings: Settings, exporter: SpanExporter | None = None) -> Langfuse | None:
    if not settings.tracing_enabled:
        return None
    assert settings.langfuse_secret_key
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        mask=mask,
        span_exporter=exporter,
    )


def trace_callbacks() -> list[BaseCallbackHandler]:
    return [CallbackHandler()] if get_settings().tracing_enabled else []
