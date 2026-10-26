import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight import reflection
from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.citations import Citation
from hindsight.enrichment import Claim, EntryInput
from hindsight.models import Generation, Status
from hindsight.pipeline import EntrySource
from hindsight.reflection import (
    MAX_LATER_ENTRIES,
    RESPONSE_FORMAT,
    ReflectionAidGenerator,
    Since,
    Trade,
)
from hindsight.vocab import EntryType
from tests.fakes import FakeToolModel
from tests.test_enrichment import objects

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
USER_ID = 4242
NOW = datetime(2026, 10, 25, 16, 0, tzinfo=UTC)
ENTRY = EntryInput(
    id=7,
    entry_type=EntryType.BUY,
    body="Buying the dip on NVDA. Data center demand looks durable.",
    ticker="NVDA",
    timestamp=datetime(2026, 3, 3, 15, 30, tzinfo=UTC),
    price_snapshot=118.0,
)
LATER = EntryInput(
    id=52,
    entry_type=EntryType.INSIGHT,
    body="Still holding. In hindsight the dip was overdone and I should have bought more.",
    ticker="NVDA",
    timestamp=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
)
SINCE = Since(
    price_now=142.3,
    shares_now=15,
    trades=[Trade(day=date(2026, 3, 4), type="BUY", shares=5, price=131.2)],
)
DRAFT: dict[str, Any] = {
    "recap": [
        {
            "summary": "You later called the dip overdone",
            "entry_id": 52,
            "quotes": ["the dip was overdone"],
        }
    ],
    "questions": ["What made this look like a dip rather than the start of a decline?"],
}


def reply(draft: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content=json.dumps(draft),
        usage_metadata={"input_tokens": 900, "output_tokens": 150, "total_tokens": 1050},
    )


@pytest.fixture
def ledger(sessions: Sessions) -> BudgetLedger:
    return BudgetLedger(sessions, Limits(-1, 1_000_000, 100))


def generator(ledger: BudgetLedger, *drafts: dict[str, Any]) -> ReflectionAidGenerator:
    model = FakeToolModel(messages=iter([reply(d) for d in drafts]))
    return ReflectionAidGenerator(ledger, model, MODEL, clock=lambda: NOW)


async def statuses(sessions: Sessions) -> list[Status]:
    async with sessions() as session:
        return list(await session.scalars(select(Generation.status)))


async def test_returns_a_grounded_aid(ledger: BudgetLedger, sessions: Sessions) -> None:
    aid = await generator(ledger, DRAFT).generate(ENTRY, [LATER], SINCE, USER_ID)

    assert aid.recap == [
        Claim(
            summary="You later called the dip overdone",
            citations=[Citation(entry_id=52, quote="the dip was overdone")],
        )
    ]
    assert aid.questions == DRAFT["questions"]
    assert await statuses(sessions) == [Status.SUCCEEDED]


@pytest.mark.parametrize(
    "point",
    [
        {"summary": "You said it was cheap", "entry_id": 52, "quotes": ["it was cheap"]},
        # quotes from the original entry are not something that happened since
        {"summary": "You bought the dip", "entry_id": 7, "quotes": ["Buying the dip"]},
        {"summary": "Unknown", "entry_id": 99, "quotes": ["the dip was overdone"]},
        {"summary": " ", "entry_id": 52, "quotes": ["the dip was overdone"]},
    ],
)
async def test_drops_recap_points_that_do_not_check_out(
    ledger: BudgetLedger, point: dict[str, Any]
) -> None:
    draft = {**DRAFT, "recap": [point]}

    aid = await generator(ledger, draft).generate(ENTRY, [LATER], SINCE, USER_ID)

    assert aid.recap == []
    assert aid.questions == DRAFT["questions"]


async def test_cleans_up_questions(ledger: BudgetLedger) -> None:
    draft = {"recap": [], "questions": [" Why then? ", "Why then?", "  "]}

    aid = await generator(ledger, draft).generate(ENTRY, [], SINCE, USER_ID)

    assert aid.questions == ["Why then?"]


async def test_prompt_carries_the_facts_but_never_the_user(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(DRAFT)]))
    gen = ReflectionAidGenerator(ledger, model, MODEL, clock=lambda: NOW)

    await gen.generate(ENTRY, [LATER, ENTRY], SINCE, USER_ID)

    [[system, human]] = model.prompts
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    sent = json.loads(human.text)
    assert sent["entry"]["body"] == ENTRY.body
    assert sent["since"] == {
        "days_passed": 236,
        "price_then": 118.0,
        "price_now": 142.3,
        "price_change_pct": 20.6,
        "shares_now": 15,
        "trades": [{"date": "2026-03-04", "type": "BUY", "shares": 5, "price": 131.2}],
    }
    # the entry itself is never repeated as a later entry
    assert [e["id"] for e in sent["later_entries"]] == [52]
    assert str(USER_ID) not in system.text + human.text
    assert model.calls[0]["max_tokens"] == reflection.MAX_OUTPUT_TOKENS


async def test_entry_without_a_ticker_gets_no_price_facts(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(DRAFT)]))
    gen = ReflectionAidGenerator(ledger, model, MODEL, clock=lambda: NOW)
    note = ENTRY.model_copy(update={"ticker": None, "price_snapshot": None})

    await gen.generate(note, [], SINCE, USER_ID)

    since = json.loads(model.prompts[0][1].text)["since"]
    assert (since["price_then"], since["price_now"], since["shares_now"]) == (None, None, None)


async def test_a_zero_price_snapshot_counts_as_unknown(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(DRAFT)]))
    gen = ReflectionAidGenerator(ledger, model, MODEL, clock=lambda: NOW)

    await gen.generate(ENTRY.model_copy(update={"price_snapshot": 0.0}), [], SINCE, USER_ID)

    since = json.loads(model.prompts[0][1].text)["since"]
    assert (since["price_then"], since["price_change_pct"]) == (None, None)


async def test_sends_only_the_newest_later_entries_oldest_first(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(DRAFT)]))
    gen = ReflectionAidGenerator(ledger, model, MODEL, clock=lambda: NOW)
    later = [
        LATER.model_copy(update={"id": 100 + i, "timestamp": LATER.timestamp + timedelta(days=i)})
        for i in range(MAX_LATER_ENTRIES + 2)
    ]

    await gen.generate(ENTRY, reversed(later), SINCE, USER_ID)

    sent = json.loads(model.prompts[0][1].text)["later_entries"]
    assert [e["id"] for e in sent] == [100 + i for i in range(2, MAX_LATER_ENTRIES + 2)]


async def test_reservation_covers_the_largest_possible_request(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    # quotes, backslashes and multi-byte characters all grow when re-encoded
    body = ('"\\é' * 700)[:2000]
    worst = ENTRY.model_copy(update={"body": body})
    later = [
        worst.model_copy(update={"id": 100 + i, "timestamp": LATER.timestamp})
        for i in range(MAX_LATER_ENTRIES)
    ]
    trades = [Trade(day=date(2026, 3, 4), type="SELL", shares=1e9, price=1e9)] * 20

    await generator(ledger, DRAFT).generate(worst, later, Since(trades=trades), USER_ID)

    assert await statuses(sessions) == [Status.SUCCEEDED]


async def test_budget_refusal_never_reaches_the_model(sessions: Sessions) -> None:
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 0))
    model = FakeToolModel(messages=iter([reply(DRAFT)]))

    with pytest.raises(BudgetError):
        await ReflectionAidGenerator(ledger, model, MODEL).generate(ENTRY, [], SINCE, USER_ID)

    assert model.calls == []


async def test_blank_entry_is_refused_before_any_spend(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    with pytest.raises(ValueError):
        await generator(ledger).generate(
            ENTRY.model_copy(update={"body": "  "}), [], SINCE, USER_ID
        )

    assert await statuses(sessions) == []


def test_strict_schema_closes_every_object() -> None:
    kwargs = RESPONSE_FORMAT.to_model_kwargs()
    found = objects(kwargs["response_format"]["json_schema"]["schema"])

    assert len(found) == 2
    assert all(o.get("additionalProperties") is False for o in found)


async def test_reads_only_that_users_entries(sessions: Sessions) -> None:
    async with sessions.begin() as session:
        await session.execute(
            text(
                "insert into core_read.journal_entries"
                " (id, user_id, entry_type, body, ticker, timestamp, price_snapshot, tags)"
                " values (7, 1, 'BUY', 'mine', 'NVDA', now(), 118.0, '{semis}'),"
                " (8, 2, 'BUY', 'theirs', 'NVDA', now(), null, '{}')"
            )
        )

    found = await EntrySource(sessions).for_user(1, [7, 8, 9])

    assert list(found) == [7]
    assert (found[7].body, found[7].tags) == ("mine", ["semis"])
