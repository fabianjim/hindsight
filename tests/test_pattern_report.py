import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight import pattern_report
from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.citations import Citation
from hindsight.enrichment import EntryInput
from hindsight.models import Generation, Status
from hindsight.pattern_report import RESPONSE_FORMAT, PatternNarrator
from hindsight.pattern_reports import MAX_PATTERNS, Figure, Pattern
from hindsight.patterns import MAX_ENTRIES_PER_PATTERN, Candidate
from hindsight.vocab import EntryType, PatternKind
from tests.fakes import FakeToolModel
from tests.test_enrichment import objects

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
USER_ID = 4242
T0 = datetime(2026, 3, 3, 15, 30, tzinfo=UTC)


def entry(entry_id: int, body: str, days: int = 0) -> EntryInput:
    return EntryInput(
        id=entry_id,
        entry_type=EntryType.BUY,
        body=body,
        ticker="NVDA",
        timestamp=T0 + timedelta(days=days),
    )


ENTRIES = {
    12: entry(12, "I'm nervous about this, but buying anyway.", 2),
    13: entry(13, "Anxious again. Added a few more.", 1),
    20: entry(20, "Thesis: data center demand stays strong for years."),
}
FELT = Candidate(
    kind=PatternKind.FELT_VS_DID,
    subject="anxious",
    figures=[
        Figure(key="entries", value=4, unit="count"),
        Figure(key="followed_by_buy", value=3, unit="count"),
        Figure(key="followed_by_sell", value=0, unit="count"),
    ],
    support=4,
    entry_ids=[12, 13],
)
SILENT = Candidate(
    kind=PatternKind.SILENT_TRADES,
    subject=None,
    figures=[
        Figure(key="trades", value=6, unit="count"),
        Figure(key="without_entry", value=4, unit="count"),
    ],
    support=6,
    entry_ids=[],
)
OPEN = Candidate(
    kind=PatternKind.OPEN_THESES,
    subject=None,
    figures=[
        Figure(key="buys_with_thesis", value=3, unit="count"),
        Figure(key="not_revisited", value=3, unit="count"),
    ],
    support=3,
    entry_ids=[20],
)
CANDIDATES = [SILENT, FELT, OPEN]
ITEMS: list[dict[str, Any]] = [
    {
        "index": 0,
        "observation": "Most of your trades have no note written around them.",
        "question": "What was on your mind for the ones you didn't write about?",
        "quotes": [],
    },
    {
        "index": 1,
        "observation": "Most times you wrote about feeling anxious, you bought soon after.",
        "question": "What did buying settle for you in those moments?",
        "quotes": [{"entry_id": 12, "quote": "I'm nervous about this"}],
    },
    {
        "index": 2,
        "observation": "You wrote a thesis for these buys and haven't come back to it.",
        "question": "How does the thesis read to you now?",
        "quotes": [{"entry_id": 20, "quote": "data center demand stays strong"}],
    },
]


def reply(items: list[dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content=json.dumps({"items": items}),
        usage_metadata={"input_tokens": 2000, "output_tokens": 400, "total_tokens": 2400},
    )


@pytest.fixture
def ledger(sessions: Sessions) -> BudgetLedger:
    return BudgetLedger(sessions, Limits(-1, 1_000_000, 100))


def narrator(ledger: BudgetLedger, items: list[dict[str, Any]]) -> PatternNarrator:
    return PatternNarrator(ledger, FakeToolModel(messages=iter([reply(items)])), MODEL)


async def narrate(ledger: BudgetLedger, items: list[dict[str, Any]]) -> list[Pattern]:
    return await narrator(ledger, items).narrate(CANDIDATES, ENTRIES, USER_ID)


def with_item(index: int, **changes: Any) -> list[dict[str, Any]]:
    return [{**item, **changes} if item["index"] == index else item for item in ITEMS]


async def statuses(sessions: Sessions) -> list[Status]:
    async with sessions() as session:
        return list(await session.scalars(select(Generation.status)))


async def test_returns_grounded_patterns_in_ranked_order(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    patterns = await narrate(ledger, list(reversed(ITEMS)))

    assert [p.kind for p in patterns] == [c.kind for c in CANDIDATES]
    felt = patterns[1]
    assert (felt.subject, felt.figures) == ("anxious", FELT.figures)
    assert felt.observation == ITEMS[1]["observation"]
    assert felt.question == ITEMS[1]["question"]
    assert felt.citations == [Citation(entry_id=12, quote="I'm nervous about this")]
    assert await statuses(sessions) == [Status.SUCCEEDED]


@pytest.mark.parametrize(
    "changes",
    [
        {"observation": "You bought 3 times after feeling anxious."},
        {"question": "What happened in the 2 weeks after?"},
        {"observation": "You bought ３ times."},  # any unicode digit
        {"observation": "You bought three times after feeling anxious."},
        {"question": "Why did you sell only once?"},
        {"observation": "  "},
        {"question": ""},
    ],
)
async def test_drops_a_pattern_whose_text_breaks_the_rules(
    ledger: BudgetLedger, changes: dict[str, Any]
) -> None:
    patterns = await narrate(ledger, with_item(1, **changes))

    assert [p.kind for p in patterns] == [PatternKind.SILENT_TRADES, PatternKind.OPEN_THESES]


@pytest.mark.parametrize(
    "quotes",
    [
        [],
        [{"entry_id": 12, "quote": "I was terrified"}],
        # a real quote, but from another pattern's entry
        [{"entry_id": 20, "quote": "data center demand stays strong"}],
        [{"entry_id": 99, "quote": "I'm nervous about this"}],
    ],
)
async def test_drops_a_pattern_left_without_citations(
    ledger: BudgetLedger, quotes: list[dict[str, Any]]
) -> None:
    patterns = await narrate(ledger, with_item(1, quotes=quotes))

    assert PatternKind.FELT_VS_DID not in [p.kind for p in patterns]


async def test_keeps_only_the_quotes_that_check_out(ledger: BudgetLedger) -> None:
    quotes = [
        {"entry_id": 12, "quote": "I'm nervous about this"},
        {"entry_id": 13, "quote": "I panicked"},
        {"entry_id": 13, "quote": "Anxious again"},
    ]

    patterns = await narrate(ledger, with_item(1, quotes=quotes))

    assert patterns[1].citations == [
        Citation(entry_id=12, quote="I'm nervous about this"),
        Citation(entry_id=13, quote="Anxious again"),
    ]


async def test_silent_trades_needs_no_citation(ledger: BudgetLedger) -> None:
    patterns = await narrate(ledger, ITEMS)

    assert patterns[0].kind == PatternKind.SILENT_TRADES
    assert patterns[0].citations == []


@pytest.mark.parametrize(
    "items",
    [
        [ITEMS[0], ITEMS[2]],  # one pattern left out
        [ITEMS[0], {**ITEMS[1], "index": 7}, ITEMS[2]],  # an index that matches nothing
    ],
)
async def test_drops_a_pattern_without_a_matching_item(
    ledger: BudgetLedger, items: list[dict[str, Any]]
) -> None:
    patterns = await narrate(ledger, items)

    assert PatternKind.FELT_VS_DID not in [p.kind for p in patterns]


async def test_the_first_item_for_an_index_wins(ledger: BudgetLedger) -> None:
    second = {**ITEMS[2], "observation": "Something else entirely."}

    patterns = await narrate(ledger, [*ITEMS, second])

    assert patterns[2].observation == ITEMS[2]["observation"]


async def test_prompt_carries_the_patterns_but_never_the_user(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(ITEMS)]))

    await PatternNarrator(ledger, model, MODEL).narrate(CANDIDATES, ENTRIES, USER_ID)

    [[system, human]] = model.prompts
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    sent = json.loads(human.text)["patterns"]
    assert [p["index"] for p in sent] == [0, 1, 2]
    assert sent[1] == {
        "index": 1,
        "kind": "felt_vs_did",
        "subject": "anxious",
        "figures": [
            {"key": "entries", "value": 4, "unit": "count"},
            {"key": "followed_by_buy", "value": 3, "unit": "count"},
            {"key": "followed_by_sell", "value": 0, "unit": "count"},
        ],
        "entries": [
            {
                "id": 12,
                "type": "BUY",
                "ticker": "NVDA",
                "date": "2026-03-05",
                "body": ENTRIES[12].body,
            },
            {
                "id": 13,
                "type": "BUY",
                "ticker": "NVDA",
                "date": "2026-03-04",
                "body": ENTRIES[13].body,
            },
        ],
    }
    assert sent[0]["entries"] == []
    assert str(USER_ID) not in system.text + human.text
    assert model.calls[0]["max_tokens"] == pattern_report.MAX_OUTPUT_TOKENS


async def test_reservation_covers_the_largest_possible_request(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    # quotes, backslashes and multi-byte characters all grow when re-encoded
    body = ('"\\é' * 700)[:2000]
    entries = {i: entry(i, body) for i in range(MAX_PATTERNS * MAX_ENTRIES_PER_PATTERN)}
    figures = [Figure(key="gains_median_days", value=1e9 + 0.5, unit="days")] * 6
    candidates = [
        Candidate(
            kind=PatternKind.OUTCOMES_BY_STATE,
            subject="idea_source:friend_or_family",
            figures=figures,
            support=1000,
            entry_ids=list(range(p * MAX_ENTRIES_PER_PATTERN, (p + 1) * MAX_ENTRIES_PER_PATTERN)),
        )
        for p in range(MAX_PATTERNS)
    ]

    await narrator(ledger, ITEMS).narrate(candidates, entries, USER_ID)

    assert await statuses(sessions) == [Status.SUCCEEDED]


async def test_budget_refusal_never_reaches_the_model(sessions: Sessions) -> None:
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 0))
    model = FakeToolModel(messages=iter([reply(ITEMS)]))

    with pytest.raises(BudgetError):
        await PatternNarrator(ledger, model, MODEL).narrate(CANDIDATES, ENTRIES, USER_ID)

    assert model.calls == []


@pytest.mark.parametrize("candidates", [[], [FELT] * (MAX_PATTERNS + 1)])
async def test_refuses_before_any_spend(
    ledger: BudgetLedger, sessions: Sessions, candidates: list[Candidate]
) -> None:
    with pytest.raises(ValueError):
        await narrator(ledger, ITEMS).narrate(candidates, ENTRIES, USER_ID)

    assert await statuses(sessions) == []


def test_strict_schema_closes_every_object() -> None:
    kwargs = RESPONSE_FORMAT.to_model_kwargs()
    found = objects(kwargs["response_format"]["json_schema"]["schema"])

    assert len(found) == 3
    assert all(o.get("additionalProperties") is False for o in found)
