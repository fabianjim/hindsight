import json
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight import enrichment
from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.citations import Citation
from hindsight.enrichment import (
    Claim,
    Emotion,
    EntryEnricher,
    EntryInput,
    EntryType,
    Extraction,
    Horizon,
    IdeaSource,
    input_hash,
    render,
)
from hindsight.models import Generation, Status
from tests.fakes import FakeToolModel

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
USER_ID = 4242
ENTRY = EntryInput(
    id=7,
    entry_type=EntryType.BUY,
    body=(
        "Bought 20 NVDA after the earnings dip. Data center demand looks durable for years. "
        "I'll reconsider if gross margins fall below 60%. Honestly a bit of FOMO too."
    ),
    ticker="NVDA",
    timestamp=datetime(2026, 3, 3, 15, 30, tzinfo=UTC),
    price_snapshot=118.0,
    tags=["semis"],
)
EXTRACTION: dict[str, Any] = {
    "thesis": {
        "summary": "Data center demand stays strong for years",
        "quotes": ["Data center demand looks durable for years"],
    },
    "exit_condition": {
        "summary": "Gross margins fall below 60%",
        "quotes": ["reconsider if gross margins fall below 60%"],
    },
    "horizon": "years",
    "conviction": None,
    "emotions": ["fomo"],
    "idea_source": None,
    "tickers": ["NVDA"],
    "themes": ["AI capex"],
}


def reply(extraction: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content=json.dumps(extraction),
        usage_metadata={"input_tokens": 600, "output_tokens": 120, "total_tokens": 720},
    )


@pytest.fixture
def ledger(sessions: Sessions) -> BudgetLedger:
    return BudgetLedger(sessions, Limits(-1, 1_000_000, 100))


async def statuses(sessions: Sessions) -> list[Status]:
    async with sessions() as session:
        return list(await session.scalars(select(Generation.status)))


async def test_extracts_a_grounded_insight(ledger: BudgetLedger, sessions: Sessions) -> None:
    model = FakeToolModel(messages=iter([reply(EXTRACTION)]))

    result = await EntryEnricher(ledger, model, MODEL).enrich(ENTRY, USER_ID)

    insight = result.insight
    assert insight.thesis == Claim(
        summary="Data center demand stays strong for years",
        citations=[Citation(entry_id=7, quote="Data center demand looks durable for years")],
    )
    assert (
        insight.exit_condition and insight.exit_condition.summary == "Gross margins fall below 60%"
    )
    assert (insight.horizon, insight.conviction, insight.idea_source) == (Horizon.YEARS, None, None)
    assert (insight.emotions, insight.tickers, insight.themes) == (
        [Emotion.FOMO],
        ["NVDA"],
        ["AI capex"],
    )
    assert (result.entry_id, result.model, result.prompt_version) == (7, MODEL, "entry-insight-v1")
    assert result.input_hash == input_hash(render(ENTRY), MODEL)
    assert await statuses(sessions) == [Status.SUCCEEDED]


async def test_drops_claims_whose_quotes_are_not_in_the_entry(ledger: BudgetLedger) -> None:
    invented = {
        **EXTRACTION,
        "thesis": {"summary": "Margins will expand", "quotes": ["margins will expand next year"]},
        "exit_condition": {"summary": "Price hits 200", "quotes": []},
    }
    model = FakeToolModel(messages=iter([reply(invented)]))

    insight = (await EntryEnricher(ledger, model, MODEL).enrich(ENTRY, USER_ID)).insight

    assert insight.thesis is None
    assert insight.exit_condition is None
    assert insight.tickers == ["NVDA"]


async def test_keeps_only_the_quotes_that_check_out(ledger: BudgetLedger) -> None:
    mixed = {
        **EXTRACTION,
        "thesis": {
            "summary": "Demand stays strong",
            "quotes": ["demand looks durable", "the CEO is a genius"],
        },
    }
    model = FakeToolModel(messages=iter([reply(mixed)]))

    insight = (await EntryEnricher(ledger, model, MODEL).enrich(ENTRY, USER_ID)).insight

    assert insight.thesis
    assert [c.quote for c in insight.thesis.citations] == ["demand looks durable"]


async def test_prompt_carries_the_entry_but_never_the_user(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(EXTRACTION)]))

    await EntryEnricher(ledger, model, MODEL).enrich(ENTRY, USER_ID)

    [[system, human]] = model.prompts
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert json.loads(human.text) == {
        "type": "BUY",
        "ticker": "NVDA",
        "date": "2026-03-03",
        "tags": ["semis"],
        "body": ENTRY.body,
    }
    assert str(USER_ID) not in system.text + human.text
    assert model.calls[0]["max_tokens"] == enrichment.MAX_OUTPUT_TOKENS


async def test_budget_refusal_never_reaches_the_model(sessions: Sessions) -> None:
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 0))
    model = FakeToolModel(messages=iter([reply(EXTRACTION)]))

    with pytest.raises(BudgetError):
        await EntryEnricher(ledger, model, MODEL).enrich(ENTRY, USER_ID)

    assert model.calls == []
    assert await statuses(sessions) == []


class BrokenModel(FakeToolModel):
    def _generate(self, messages: list[BaseMessage], *_: Any, **__: Any) -> ChatResult:
        raise ConnectionError("provider went away")


async def test_model_failure_still_finishes_the_job(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    with pytest.raises(ConnectionError):
        await EntryEnricher(ledger, BrokenModel(messages=iter([])), MODEL).enrich(ENTRY, USER_ID)

    # the provider call is unresolved, so the job keeps its reservation as unknown
    assert await statuses(sessions) == [Status.UNKNOWN]


async def test_each_attempt_is_a_new_budget_job(ledger: BudgetLedger, sessions: Sessions) -> None:
    model = FakeToolModel(messages=iter([reply(EXTRACTION), reply(EXTRACTION)]))
    enricher = EntryEnricher(ledger, model, MODEL)

    first = await enricher.enrich(ENTRY, USER_ID)
    second = await enricher.enrich(ENTRY, USER_ID)

    assert first.input_hash == second.input_hash
    assert await statuses(sessions) == [Status.SUCCEEDED, Status.SUCCEEDED]


def test_input_hash_tracks_what_the_model_sees(monkeypatch: pytest.MonkeyPatch) -> None:
    message = render(ENTRY)
    original = input_hash(message, MODEL)

    assert input_hash(render(ENTRY.model_copy(update={"price_snapshot": 999.0})), MODEL) == original
    assert input_hash(render(ENTRY.model_copy(update={"body": "Sold NVDA."})), MODEL) != original
    assert input_hash(message, "openai:other-model") != original
    monkeypatch.setattr(enrichment, "PROMPT", enrichment.PROMPT + "\nBe brief.")
    assert input_hash(message, MODEL) != original


@pytest.mark.parametrize(
    "change",
    [
        {"conviction": 7},
        {"emotions": ["greedy"]},
        {"tickers": ["nvda"]},
        {"emotions": ["fomo"] * 4},
    ],
)
def test_schema_rejects_out_of_range_values(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Extraction.model_validate({**EXTRACTION, **change})


def test_schema_accepts_every_documented_value() -> None:
    for horizon in Horizon:
        for source in IdeaSource:
            Extraction.model_validate({**EXTRACTION, "horizon": horizon, "idea_source": source})
    Extraction.model_validate({**EXTRACTION, "emotions": list(Emotion)[:3], "conviction": 5})
