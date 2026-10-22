import uuid
from pathlib import Path

import httpx
import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from hindsight.config import get_settings
from hindsight.llm import UserContext, get_chat_model
from tests.fakes import FakeToolModel

ENTRIES = {1: ["bought NVDA on the earnings dip"], 2: ["sold AMD, felt like FOMO"]}


@tool
def list_entries(runtime: ToolRuntime[UserContext]) -> list[str]:
    """List the user's journal entries."""
    return ENTRIES[runtime.context.user_id]


def test_tools_only_see_the_user_from_context() -> None:
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "list_entries", "args": {}, "id": "1"}]),
                AIMessage(content="done"),
            ]
        )
    )
    agent = create_agent(model, tools=[list_entries], context_schema=UserContext)

    result = agent.invoke(
        {"messages": [HumanMessage("what did i write?")]},
        context=UserContext(user_id=1, generation_id=uuid.uuid4()),
    )

    [tool_message] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "NVDA" in tool_message.text
    assert "AMD" not in tool_message.text


def test_user_id_is_not_a_tool_argument() -> None:
    assert "user_id" not in list_entries.args
    assert "runtime" not in list_entries.args


def test_builds_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()

    model = get_chat_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4.1-mini-2025-04-14"
    for client in (model.root_client, model.root_async_client):
        assert client.max_retries == 0
        assert client.timeout == httpx.Timeout(30.0, connect=5.0)
    assert model._get_request_payload([HumanMessage("hi")])["store"] is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("key", [None, ""])
def test_refuses_to_build_without_its_own_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str | None
) -> None:
    # away from any local .env
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stray")
    if key is None:
        monkeypatch.delenv("HINDSIGHT_OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("HINDSIGHT_OPENAI_API_KEY", key)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="HINDSIGHT_OPENAI_API_KEY"):
        get_chat_model()
