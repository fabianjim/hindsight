import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight import behavior_report
from hindsight.behavior_report import RESPONSE_FORMAT, BehaviorNarrator, fits_card
from hindsight.behavior_reports import MAX_BEHAVIORS, Behavior, Figure, Previous
from hindsight.behaviors import MAX_NOTE_LENGTH, MAX_NOTES, MAX_SHOWN_ENTRIES, FoundBehavior
from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.citations import Citation
from hindsight.enrichment import EntryInput
from hindsight.models import Generation, Status
from hindsight.vocab import BehaviorKind, EntryType
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


def found(
    kind: BehaviorKind,
    figures: list[Figure],
    shown: list[int],
    subject: str | None = None,
    previous: Previous | None = None,
    feedback_notes: list[str] | None = None,
    note_entry_ids: list[int] | None = None,
) -> FoundBehavior:
    return FoundBehavior(
        kind=kind,
        subject=subject,
        figures=figures,
        previous=previous,
        trade_ids=[41, 33, 19, 7],
        entry_ids=[*shown, 77],
        shown_entry_ids=shown,
        feedback_notes=feedback_notes or [],
        note_entry_ids=note_entry_ids or [],
        score=100.0,
    )


ENTRIES = {
    12: entry(12, "It's running, I need to be in before it leaves without me.", 2),
    13: entry(13, "Up again this week. Buying more before it's gone.", 1),
    20: entry(20, "Nervous about this one, it keeps dropping.", 3),
    90: entry(90, "I know I chase things that are already up."),
}
UNDOCUMENTED = found(
    BehaviorKind.UNDOCUMENTED,
    [
        Figure(key="decisions", value=6, unit="count"),
        Figure(key="pnl", value=-310.25, unit="money"),
    ],
    shown=[],
)
RUN_UP = found(
    BehaviorKind.RUN_UP,
    [
        Figure(key="decisions", value=5, unit="count"),
        Figure(key="avg_return", value=-0.061, unit="fraction"),
        Figure(key="baseline_avg_return", value=0.083, unit="fraction"),
        Figure(key="median_run_up", value=0.142, unit="fraction"),
    ],
    shown=[12, 13],
    previous=Previous(
        day=date(2026, 3, 1), figures=[Figure(key="decisions", value=4, unit="count")]
    ),
    feedback_notes=["I've noticed this myself."],
    note_entry_ids=[90],
)
EMOTIONAL = found(
    BehaviorKind.EMOTIONAL,
    [Figure(key="decisions", value=3, unit="count")],
    shown=[20],
    subject="anxious",
)
BEHAVIORS = [UNDOCUMENTED, RUN_UP, EMOTIONAL]
ITEMS: list[dict[str, Any]] = [
    {
        "index": 0,
        "observation": "Most of your trades have no note written around them.",
        "question": "What was on your mind for the ones you didn't write about?",
        "quotes": [],
    },
    {
        "index": 1,
        "observation": "Your buys after a strong rise ended lower on average than your other buys.",
        "question": "What were you seeing in those moments that made waiting feel costly?",
        "quotes": [{"entry_id": 12, "quote": "I need to be in before it leaves without me"}],
    },
    {
        "index": 2,
        "observation": "Trades you made after writing that you felt anxious mostly ended lower.",
        "question": "What did trading settle for you in those moments?",
        "quotes": [{"entry_id": 20, "quote": "Nervous about this one"}],
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


def narrator(ledger: BudgetLedger, items: list[dict[str, Any]]) -> BehaviorNarrator:
    return BehaviorNarrator(ledger, FakeToolModel(messages=iter([reply(items)])), MODEL)


async def narrate(ledger: BudgetLedger, items: list[dict[str, Any]]) -> list[Behavior]:
    return await narrator(ledger, items).narrate(BEHAVIORS, ENTRIES, USER_ID)


def with_item(index: int, **changes: Any) -> list[dict[str, Any]]:
    return [{**item, **changes} if item["index"] == index else item for item in ITEMS]


async def statuses(sessions: Sessions) -> list[Status]:
    async with sessions() as session:
        return list(await session.scalars(select(Generation.status)))


async def test_returns_grounded_behaviors_in_ranked_order(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    behaviors = await narrate(ledger, list(reversed(ITEMS)))

    assert [b.kind for b in behaviors] == [b.kind for b in BEHAVIORS]
    assert behaviors[1] == Behavior(
        kind=BehaviorKind.RUN_UP,
        subject=None,
        figures=RUN_UP.figures,
        previous=RUN_UP.previous,
        observation=ITEMS[1]["observation"],
        question=ITEMS[1]["question"],
        citations=[Citation(entry_id=12, quote="I need to be in before it leaves without me")],
        trade_ids=[41, 33, 19, 7],
        entry_ids=[12, 13, 77],
    )
    assert behaviors[2].subject == "anxious"
    assert await statuses(sessions) == [Status.SUCCEEDED]


@pytest.mark.parametrize(
    "changes",
    [
        {"observation": "You bought 3 times after a strong rise."},
        {"question": "What happened in the 2 weeks after?"},
        {"observation": "You bought ３ times."},  # any unicode digit
        {"observation": "You bought three times after a strong rise."},
        {"question": "Why did you wait only once?"},
        {"observation": "  "},
        {"question": ""},
        {"observation": "These ended lower. Your others ended higher. That held every time."},
        {"observation": " ".join(["word"] * 25) + "."},
    ],
)
async def test_drops_a_behavior_whose_text_breaks_the_rules(
    ledger: BudgetLedger, changes: dict[str, Any]
) -> None:
    behaviors = await narrate(ledger, with_item(1, **changes))

    assert [b.kind for b in behaviors] == [BehaviorKind.UNDOCUMENTED, BehaviorKind.EMOTIONAL]


@pytest.mark.parametrize(
    "observation",
    [
        "Every one of these buys ended lower than your other buys.",
        "Each one of these buys ended lower than your other buys.",
    ],
)
async def test_every_one_and_each_one_mean_all_not_a_count(
    ledger: BudgetLedger, observation: str
) -> None:
    behaviors = await narrate(ledger, with_item(1, observation=observation))

    assert behaviors[1].observation == observation


@pytest.mark.parametrize(
    "observation",
    ["Only one of these buys ended higher.", "One of these buys ended higher."],
)
async def test_a_single_count_is_still_dropped(ledger: BudgetLedger, observation: str) -> None:
    behaviors = await narrate(ledger, with_item(1, observation=observation))

    assert BehaviorKind.RUN_UP not in [b.kind for b in behaviors]


@pytest.mark.parametrize(
    ("observation", "fits"),
    [
        ("Most of these ended lower.", True),
        ("Most of these ended lower. Your other buys mostly ended higher.", True),
        ("Most of these ended lower! Your others? Higher.", False),
        (" ".join(["word"] * 24) + ".", True),
        (" ".join(["word"] * 25) + ".", False),
        ("Short. " + " ".join(["word"] * 25), False),
    ],
)
def test_the_observation_must_fit_the_card(observation: str, fits: bool) -> None:
    assert fits_card(observation) is fits


@pytest.mark.parametrize(
    "quotes",
    [
        [],
        [{"entry_id": 12, "quote": "I was terrified"}],
        # a real quote, but from another behavior's entry
        [{"entry_id": 20, "quote": "Nervous about this one"}],
        # a real quote from this behavior's note, which is context only
        [{"entry_id": 90, "quote": "I chase things that are already up"}],
        [{"entry_id": 99, "quote": "I need to be in"}],
    ],
)
async def test_drops_a_behavior_left_without_citations(
    ledger: BudgetLedger, quotes: list[dict[str, Any]]
) -> None:
    behaviors = await narrate(ledger, with_item(1, quotes=quotes))

    assert BehaviorKind.RUN_UP not in [b.kind for b in behaviors]


async def test_keeps_only_the_quotes_that_check_out(ledger: BudgetLedger) -> None:
    quotes = [
        {"entry_id": 12, "quote": "I need to be in"},
        {"entry_id": 13, "quote": "I panicked"},
        {"entry_id": 13, "quote": "Buying more before it's gone"},
    ]

    behaviors = await narrate(ledger, with_item(1, quotes=quotes))

    assert behaviors[1].citations == [
        Citation(entry_id=12, quote="I need to be in"),
        Citation(entry_id=13, quote="Buying more before it's gone"),
    ]


async def test_undocumented_needs_no_citation(ledger: BudgetLedger) -> None:
    behaviors = await narrate(ledger, ITEMS)

    assert behaviors[0].kind == BehaviorKind.UNDOCUMENTED
    assert behaviors[0].citations == []


@pytest.mark.parametrize(
    "items",
    [
        [ITEMS[0], ITEMS[2]],  # one behavior left out
        [ITEMS[0], {**ITEMS[1], "index": 7}, ITEMS[2]],  # an index that matches nothing
    ],
)
async def test_drops_a_behavior_without_a_matching_item(
    ledger: BudgetLedger, items: list[dict[str, Any]]
) -> None:
    behaviors = await narrate(ledger, items)

    assert BehaviorKind.RUN_UP not in [b.kind for b in behaviors]


async def test_the_first_item_for_an_index_wins(ledger: BudgetLedger) -> None:
    second = {**ITEMS[2], "observation": "Something else entirely."}

    behaviors = await narrate(ledger, [*ITEMS, second])

    assert behaviors[2].observation == ITEMS[2]["observation"]


async def test_prompt_carries_the_behaviors_but_never_the_user(ledger: BudgetLedger) -> None:
    model = FakeToolModel(messages=iter([reply(ITEMS)]))

    await BehaviorNarrator(ledger, model, MODEL).narrate(BEHAVIORS, ENTRIES, USER_ID)

    [[system, human]] = model.prompts
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    sent = json.loads(human.text)["behaviors"]
    assert [b["index"] for b in sent] == [0, 1, 2]
    assert sent[1] == {
        "index": 1,
        "kind": "run_up",
        "subject": None,
        "figures": [
            {"key": "decisions", "value": 5, "unit": "count"},
            {"key": "avg_return", "value": -0.061, "unit": "fraction"},
            {"key": "baseline_avg_return", "value": 0.083, "unit": "fraction"},
            {"key": "median_run_up", "value": 0.142, "unit": "fraction"},
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
        "notes": ["I've noticed this myself.", ENTRIES[90].body],
    }
    assert sent[0]["entries"] == [] and sent[0]["notes"] == []
    # vestry shows the change since the last report; the model never sees it
    assert "previous" not in sent[1]
    assert str(USER_ID) not in system.text + human.text
    assert model.calls[0]["max_tokens"] == behavior_report.MAX_OUTPUT_TOKENS


async def test_reservation_covers_the_largest_possible_request(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    # quotes, backslashes and multi-byte characters all grow when re-encoded
    body = ('"\\é' * 700)[:2000]
    note = ('"\\é' * 700)[:MAX_NOTE_LENGTH]
    per = MAX_SHOWN_ENTRIES + MAX_NOTES
    entries = {i: entry(i, body) for i in range(MAX_BEHAVIORS * per)}
    figures = [Figure(key="baseline_open_losses_held_longer", value=1e9 + 0.5, unit="money")] * 20
    behaviors = [
        found(
            BehaviorKind.RISK_CONCENTRATION,
            figures,
            shown=list(range(b * per, b * per + MAX_SHOWN_ENTRIES)),
            subject="x" * 40,
            # notes from feedback and from entries, at most MAX_NOTES together; the longest mix
            feedback_notes=[note],
            note_entry_ids=list(range(b * per + MAX_SHOWN_ENTRIES + 1, (b + 1) * per)),
        )
        for b in range(MAX_BEHAVIORS)
    ]

    await narrator(ledger, ITEMS).narrate(behaviors, entries, USER_ID)

    assert await statuses(sessions) == [Status.SUCCEEDED]


async def test_budget_refusal_never_reaches_the_model(sessions: Sessions) -> None:
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 0))
    model = FakeToolModel(messages=iter([reply(ITEMS)]))

    with pytest.raises(BudgetError):
        await BehaviorNarrator(ledger, model, MODEL).narrate(BEHAVIORS, ENTRIES, USER_ID)

    assert model.calls == []


@pytest.mark.parametrize("behaviors", [[], [RUN_UP] * (MAX_BEHAVIORS + 1)])
async def test_refuses_before_any_spend(
    ledger: BudgetLedger, sessions: Sessions, behaviors: list[FoundBehavior]
) -> None:
    with pytest.raises(ValueError):
        await narrator(ledger, ITEMS).narrate(behaviors, ENTRIES, USER_ID)

    assert await statuses(sessions) == []


def test_strict_schema_closes_every_object() -> None:
    kwargs = RESPONSE_FORMAT.to_model_kwargs()
    found_objects = objects(kwargs["response_format"]["json_schema"]["schema"])

    assert len(found_objects) == 3
    assert all(o.get("additionalProperties") is False for o in found_objects)
