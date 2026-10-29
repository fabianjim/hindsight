from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.behavior_reports import (
    HISTORY_LIMIT,
    MAX_LISTED,
    Behavior,
    BehaviorReport,
    BehaviorReportStore,
    BehaviorSummary,
    Figure,
    Need,
    PastBehavior,
    PastReport,
    Previous,
)
from hindsight.citations import Citation
from hindsight.vocab import AttemptStatus, BehaviorKind

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
TODAY = date(2026, 10, 28)
AT = datetime(2026, 10, 28, 14, 10, 2, tzinfo=UTC)
FIGURES = [
    Figure(key="decisions", value=5, unit="count"),
    Figure(key="avg_return", value=-0.061, unit="fraction"),
]
REPORT = BehaviorReport(
    day=TODAY,
    generated_at=AT,
    as_of=TODAY,
    entries_up_to=datetime(2026, 10, 28, 13, 58, tzinfo=UTC),
    saved=True,
    behaviors=[
        Behavior(
            kind=BehaviorKind.RUN_UP,
            subject=None,
            figures=FIGURES,
            previous=Previous(
                day=date(2026, 10, 14), figures=[Figure(key="decisions", value=4, unit="count")]
            ),
            observation="Most of your buys after a strong rise ended lower than where you bought.",
            question="What were you seeing in those moments that made waiting feel costly?",
            citations=[Citation(entry_id=12, quote="it's running, I need to be in")],
            trade_ids=[7, 19, 33],
            entry_ids=[12, 20],
        ),
        Behavior(
            kind=BehaviorKind.EMOTIONAL,
            subject="anxious",
            figures=[Figure(key="pnl", value=-420.5, unit="money")],
            previous=None,
            observation="Several trades you made while anxious were sold at a loss.",
            question="What did trading settle for you in those moments?",
            citations=[Citation(entry_id=20, quote="I can't sleep over this")],
            trade_ids=[19, 41],
            entry_ids=[20],
        ),
    ],
    needs=[Need(kind=BehaviorKind.PRICE_ACTION, subject=None, more=2, reason="ENTRIES")],
)


@pytest.fixture
def store(sessions: Sessions) -> BehaviorReportStore:
    return BehaviorReportStore(sessions)


async def save(
    store: BehaviorReportStore,
    user_id: int = 1,
    day: date = TODAY,
    report: BehaviorReport = REPORT,
) -> BehaviorReport:
    report = report.model_copy(update={"day": day})
    return await store.save(user_id, report, "openai:gpt-4.1-mini", "behavior-report-v1")


async def test_round_trips_a_report(store: BehaviorReportStore) -> None:
    assert await save(store) == REPORT

    assert await store.for_day(1, TODAY) == REPORT


async def test_round_trips_a_report_without_entries(store: BehaviorReportStore) -> None:
    empty = REPORT.model_copy(update={"entries_up_to": None, "behaviors": [], "needs": []})

    await save(store, report=empty)

    assert await store.for_day(1, TODAY) == empty


async def test_stored_reports_are_marked_saved(store: BehaviorReportStore) -> None:
    assert (await save(store, report=REPORT.model_copy(update={"saved": False}))).saved

    stored = await store.for_day(1, TODAY)
    assert stored is not None and stored.saved


async def test_the_first_save_of_the_day_wins(store: BehaviorReportStore) -> None:
    await save(store)
    later = REPORT.model_copy(update={"behaviors": [], "generated_at": AT + timedelta(hours=1)})

    assert await save(store, report=later) == REPORT
    assert await store.for_day(1, TODAY) == REPORT


async def test_stores_ids_in_snake_case(store: BehaviorReportStore, sessions: Sessions) -> None:
    await save(store)

    async with sessions() as session:
        behavior = await session.scalar(
            text("select report->'behaviors'->0 from ai.behavior_reports")
        )
    assert behavior["trade_ids"] == [7, 19, 33]
    assert behavior["entry_ids"] == [12, 20]


async def test_reads_are_scoped_by_user(store: BehaviorReportStore) -> None:
    await save(store, user_id=1)

    assert await store.for_day(2, TODAY) is None
    assert (await store.page(2)).reports == []
    assert await store.history(2, TODAY + timedelta(days=1), [TODAY]) == []


async def test_lists_summaries_newest_first(store: BehaviorReportStore) -> None:
    for days_ago in (3, 0, 7):
        await save(store, day=TODAY - timedelta(days=days_ago))

    listing = await store.page(1)

    assert [r.day for r in listing.reports] == [
        TODAY,
        TODAY - timedelta(days=3),
        TODAY - timedelta(days=7),
    ]
    assert not listing.has_older
    assert listing.reports[0].generated_at == AT
    assert listing.reports[0].behaviors == [
        BehaviorSummary(kind=BehaviorKind.RUN_UP, subject=None, figures=FIGURES),
        BehaviorSummary(
            kind=BehaviorKind.EMOTIONAL,
            subject="anxious",
            figures=[Figure(key="pnl", value=-420.5, unit="money")],
        ),
    ]


async def test_summaries_carry_no_narration_or_ids(store: BehaviorReportStore) -> None:
    await save(store)

    listing = await store.page(1)

    behavior = listing.model_dump(by_alias=True)["reports"][0]["behaviors"][0]
    assert set(behavior) == {"kind", "subject", "figures"}


async def test_pages_with_a_keyset_on_day(store: BehaviorReportStore) -> None:
    for days_ago in range(5):
        await save(store, day=TODAY - timedelta(days=days_ago))

    first = await store.page(1, limit=2)
    assert [r.day for r in first.reports] == [TODAY, TODAY - timedelta(days=1)]
    assert first.has_older

    second = await store.page(1, before=first.reports[-1].day, limit=2)
    assert [r.day for r in second.reports] == [
        TODAY - timedelta(days=2),
        TODAY - timedelta(days=3),
    ]
    assert second.has_older

    last = await store.page(1, before=second.reports[-1].day, limit=2)
    assert [r.day for r in last.reports] == [TODAY - timedelta(days=4)]
    assert not last.has_older


async def test_before_is_exclusive(store: BehaviorReportStore) -> None:
    await save(store)

    assert (await store.page(1, before=TODAY)).reports == []


async def test_caps_the_page_size(store: BehaviorReportStore) -> None:
    for days_ago in range(MAX_LISTED + 1):
        await save(store, day=TODAY - timedelta(days=days_ago))

    listing = await store.page(1, limit=MAX_LISTED + 10)

    assert len(listing.reports) == MAX_LISTED
    assert listing.has_older


async def test_history_holds_earlier_reports_newest_first(store: BehaviorReportStore) -> None:
    await save(store, day=TODAY - timedelta(days=14))
    await save(store, day=TODAY - timedelta(days=7))
    await save(store, day=TODAY)

    history = await store.history(1, TODAY)

    assert history == [
        PastReport(
            day=TODAY - timedelta(days=days_ago),
            behaviors=[
                PastBehavior("run_up", None, FIGURES, [7, 19, 33]),
                PastBehavior(
                    "emotional",
                    "anxious",
                    [Figure(key="pnl", value=-420.5, unit="money")],
                    [19, 41],
                ),
            ],
        )
        for days_ago in (7, 14)
    ]


async def test_history_is_capped_but_adds_the_named_days(store: BehaviorReportStore) -> None:
    oldest = TODAY - timedelta(days=HISTORY_LIMIT + 5)
    for days_ago in range(1, HISTORY_LIMIT + 6):
        await save(store, day=TODAY - timedelta(days=days_ago))

    history = await store.history(1, TODAY, [oldest, TODAY - timedelta(days=1)])

    days = [r.day for r in history]
    assert len(days) == HISTORY_LIMIT + 1
    assert days[0] == TODAY - timedelta(days=1)
    assert days[-1] == oldest
    assert days == sorted(days, reverse=True)


async def test_history_skips_named_days_without_a_report(store: BehaviorReportStore) -> None:
    await save(store, day=TODAY - timedelta(days=1))

    history = await store.history(1, TODAY, [TODAY - timedelta(days=9)])

    assert [r.day for r in history] == [TODAY - timedelta(days=1)]


async def test_starts_one_attempt_per_day(store: BehaviorReportStore) -> None:
    assert await store.start_attempt(1, TODAY)
    assert not await store.start_attempt(1, TODAY)
    assert await store.start_attempt(2, TODAY)
    assert await store.start_attempt(1, TODAY + timedelta(days=1))


async def test_an_unfinished_attempt_reads_as_failed(store: BehaviorReportStore) -> None:
    await store.start_attempt(1, TODAY)

    attempt = await store.attempt(1, TODAY)

    assert attempt is not None and attempt.status == "FAILED"


@pytest.mark.parametrize(
    ("status", "shown"), [(AttemptStatus.SAVED, "SAVED"), (AttemptStatus.FAILED, "FAILED")]
)
async def test_finishing_sets_the_status_and_time(
    store: BehaviorReportStore, status: AttemptStatus, shown: str
) -> None:
    await store.start_attempt(1, TODAY)
    started = await store.attempt(1, TODAY)

    await store.finish_attempt(1, TODAY, status)

    attempt = await store.attempt(1, TODAY)
    assert attempt is not None and started is not None
    assert attempt.status == shown
    assert attempt.at >= started.at


async def test_finishing_is_scoped_by_user(store: BehaviorReportStore) -> None:
    await store.start_attempt(1, TODAY)
    await store.start_attempt(2, TODAY)

    await store.finish_attempt(2, TODAY, AttemptStatus.SAVED)

    mine = await store.attempt(1, TODAY)
    assert mine is not None and mine.status == "FAILED"


async def test_lists_the_newest_attempt(store: BehaviorReportStore) -> None:
    assert (await store.page(1)).last_attempt is None

    await store.start_attempt(1, TODAY - timedelta(days=1))
    await store.finish_attempt(1, TODAY - timedelta(days=1), AttemptStatus.SAVED)
    await store.start_attempt(1, TODAY)
    await store.start_attempt(2, TODAY + timedelta(days=1))

    last = (await store.page(1)).last_attempt
    assert last is not None
    assert (last.day, last.status) == (TODAY, "FAILED")
