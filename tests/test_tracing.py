from collections.abc import Callable

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hindsight.config import Settings, get_settings
from hindsight.tracing import capture_content, init_tracing, trace_callbacks
from tests.conftest import SERVICE_TOKEN
from tests.fakes import FakeToolModel

exporter = InMemorySpanExporter()


@pytest.fixture(scope="module")
def client() -> Langfuse:
    settings = Settings.model_validate(
        {
            "service_token": SERVICE_TOKEN,
            "database_url": "postgresql+psycopg://localhost/test",
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
        }
    )
    client = init_tracing(settings, exporter)
    assert client
    return client


@pytest.fixture
def langfuse(client: Langfuse, monkeypatch: pytest.MonkeyPatch) -> Langfuse:
    monkeypatch.setenv("HINDSIGHT_LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("HINDSIGHT_LANGFUSE_SECRET_KEY", "sk-test")
    get_settings.cache_clear()
    return client


def traced_text(langfuse: Langfuse, run: Callable[[], None]) -> str:
    exporter.clear()
    run()
    langfuse.flush()
    spans = exporter.get_finished_spans()
    assert spans
    return " ".join(str(v) for span in spans for v in (span.attributes or {}).values())


def run_agent() -> None:
    model = FakeToolModel(messages=iter([AIMessage("private answer")]))
    create_agent(model).invoke(
        {"messages": [HumanMessage("private question")]}, config={"callbacks": trace_callbacks()}
    )


def test_content_is_redacted_by_default(langfuse: Langfuse) -> None:
    text = traced_text(langfuse, run_agent)

    assert "private" not in text
    assert "[redacted]" in text


def test_content_is_sent_when_captured(langfuse: Langfuse) -> None:
    def run() -> None:
        with capture_content():
            run_agent()

    text = traced_text(langfuse, run)

    assert "private question" in text
    assert "private answer" in text


def test_tracing_is_off_without_keys() -> None:
    assert init_tracing(get_settings()) is None
    assert trace_callbacks() == []
