import uuid
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy, ToolStrategy
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatResult
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.config import Settings
from hindsight.llm import UserContext
from hindsight.models import Call, Generation, Status
from hindsight.pricing import Price, price_of
from hindsight.spend import BudgetMiddleware, response_schemas
from tests.fakes import FakeToolModel

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
PRICE = price_of(MODEL)
MAX_OUTPUT = 256


@tool
def lookup() -> str:
    """Look something up."""
    return "found it"


class Thesis(BaseModel):
    summary: str


def usage(input_tokens: int, output_tokens: int) -> UsageMetadata:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


@pytest.fixture
def ledger(sessions: Sessions) -> BudgetLedger:
    return BudgetLedger(sessions, Limits(-1, 1_000_000, 100))


async def run(ledger: BudgetLedger, model: FakeToolModel, allowance: int = 100_000) -> uuid.UUID:
    job = await ledger.start(uuid.uuid4().hex * 2, allowance)
    agent = create_agent(
        model,
        tools=[lookup],
        context_schema=UserContext,
        middleware=[BudgetMiddleware(ledger, MODEL, "test", MAX_OUTPUT)],
    )
    await agent.ainvoke(
        {"messages": [HumanMessage("look it up")]},
        context=UserContext(user_id=1, generation_id=job.id),
    )
    return job.id


async def calls_of(sessions: Sessions) -> list[Call]:
    async with sessions() as session:
        return list(await session.scalars(select(Call).order_by(Call.step)))


async def test_reserves_and_settles_every_model_call(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup", "args": {}, "id": "1"}],
                    usage_metadata=usage(1_000, 50),
                ),
                AIMessage(content="done", usage_metadata=usage(1_200, 20)),
            ]
        )
    )

    job_id = await run(ledger, model)
    await ledger.finish(job_id, success=True)

    first, second = await calls_of(sessions)
    assert (first.step, second.step) == ("test-1", "test-2")
    assert first.settled and second.settled
    assert (first.charged_micros, second.charged_micros) == (480, 512)
    assert all(c.reserved_micros >= PRICE.cost(0, MAX_OUTPUT) for c in (first, second))
    assert [c["max_tokens"] for c in model.calls] == [MAX_OUTPUT, MAX_OUTPUT]
    async with sessions() as session:
        job = await session.get(Generation, job_id)
        assert job is not None
        assert (job.status, job.charged_micros) == (Status.SUCCEEDED, 992)


async def test_refused_reservation_never_reaches_the_model(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    model = FakeToolModel(messages=iter([AIMessage(content="done", usage_metadata=usage(1, 1))]))

    with pytest.raises(BudgetError):
        await run(ledger, model, allowance=10)

    assert model.calls == []
    assert await calls_of(sessions) == []


async def test_missing_usage_keeps_the_full_reservation(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    model = FakeToolModel(messages=iter([AIMessage(content="done")]))
    job = await ledger.start("a" * 64, 100_000)
    agent = create_agent(
        model,
        context_schema=UserContext,
        middleware=[BudgetMiddleware(ledger, MODEL, "test", MAX_OUTPUT)],
    )

    with pytest.raises(BudgetError):
        await agent.ainvoke(
            {"messages": [HumanMessage("hi")]},
            context=UserContext(user_id=1, generation_id=job.id),
        )
    await ledger.finish(job.id, success=True)

    [call] = await calls_of(sessions)
    assert not call.settled
    async with sessions() as session:
        stored = await session.get(Generation, job.id)
        assert stored is not None
        assert (stored.status, stored.charged_micros) == (Status.UNKNOWN, call.reserved_micros)


class BrokenModel(FakeToolModel):
    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, *_: Any, **__: Any
    ) -> ChatResult:
        raise ConnectionError("provider went away")


async def test_provider_error_leaves_the_call_unsettled(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    with pytest.raises(ConnectionError):
        await run(ledger, BrokenModel(messages=iter([])))

    [call] = await calls_of(sessions)
    assert not call.settled
    assert call.charged_micros == call.reserved_micros


async def test_overrun_settles_and_blocks_all_spending(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    model = FakeToolModel(
        messages=iter([AIMessage(content="done", usage_metadata=usage(1_000_000, 0))])
    )

    with pytest.raises(BudgetError):
        await run(ledger, model)

    [call] = await calls_of(sessions)
    assert call.settled
    assert call.charged_micros == 400_000
    assert not await ledger.can_start(1)


async def test_sync_invoke_fails_instead_of_skipping_the_budget(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([AIMessage(content="done", usage_metadata=usage(1, 1))]))
    job = await ledger.start("b" * 64, 100_000)
    agent = create_agent(
        model,
        context_schema=UserContext,
        middleware=[BudgetMiddleware(ledger, MODEL, "test", MAX_OUTPUT)],
    )

    with pytest.raises(NotImplementedError):
        agent.invoke(
            {"messages": [HumanMessage("hi")]},
            context=UserContext(user_id=1, generation_id=job.id),
        )
    assert model.calls == []


def test_unknown_models_are_refused() -> None:
    ledger = BudgetLedger(async_sessionmaker(), Limits(-1, 1, 1))

    with pytest.raises(ValueError):
        BudgetMiddleware(ledger, "openai:gpt-4.1-mini", "test", MAX_OUTPUT)


def test_default_model_has_a_price() -> None:
    assert price_of(Settings.model_fields["chat_model"].default)


def test_cost_rounds_up_to_the_next_micro() -> None:
    price = Price(400_000, 1_600_000)

    assert price.cost(1, 0) == 1
    assert price.cost(5, 0) == 2
    assert price.cost(0, 1) == 2
    assert price.cost(1_000, 50) == 480


def test_bounds_structured_output_schemas() -> None:
    for response_format in (ToolStrategy(Thesis), AutoStrategy(Thesis)):
        [schema] = response_schemas(response_format)
        assert "summary" in str(schema)
    with pytest.raises(ValueError):
        response_schemas(object())
