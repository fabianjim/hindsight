from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.citations import Citation
from hindsight.pattern_reports import Figure, Need, Pattern, PatternReport, PatternReportStore
from hindsight.vocab import PatternKind

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
TODAY = date(2026, 10, 27)
REPORT = PatternReport(
    generated_on=TODAY,
    as_of=TODAY,
    entries_up_to=datetime(2026, 10, 27, 14, 3, tzinfo=UTC),
    saved=True,
    patterns=[
        Pattern(
            kind=PatternKind.FELT_VS_DID,
            subject="anxious",
            figures=[
                Figure(key="entries", value=4, unit="count"),
                Figure(key="followed_by_buy", value=3, unit="count"),
            ],
            observation="Most times you wrote about feeling anxious, you bought soon after.",
            question="What did buying settle for you in those moments?",
            citations=[Citation(entry_id=12, quote="I'm nervous about this")],
        ),
        Pattern(
            kind=PatternKind.SILENT_TRADES,
            subject=None,
            figures=[Figure(key="trades", value=5, unit="count")],
            observation="Some of your trades have no note around them.",
            question="What was on your mind for the ones you didn't write about?",
            citations=[],
        ),
    ],
    needs=[Need(kind=PatternKind.GAINS_VS_LOSSES, more=2)],
)


@pytest.fixture
def store(sessions: Sessions) -> PatternReportStore:
    return PatternReportStore(sessions)


async def save(
    store: PatternReportStore,
    user_id: int = 1,
    day: date = TODAY,
    report: PatternReport = REPORT,
) -> PatternReport:
    report = report.model_copy(update={"generated_on": day})
    return await store.save(user_id, report, "openai:gpt-4.1-mini", "pattern-report-v1")


async def test_round_trips_a_report(store: PatternReportStore) -> None:
    assert await save(store) == REPORT

    assert await store.for_day(1, TODAY) == REPORT


async def test_round_trips_a_report_without_entries(store: PatternReportStore) -> None:
    empty = REPORT.model_copy(update={"entries_up_to": None, "patterns": [], "needs": []})

    await save(store, report=empty)

    assert await store.for_day(1, TODAY) == empty


async def test_stored_reports_are_marked_saved(store: PatternReportStore) -> None:
    assert (await save(store, report=REPORT.model_copy(update={"saved": False}))).saved

    stored = await store.for_day(1, TODAY)
    assert stored is not None and stored.saved


async def test_stores_the_report_in_snake_case(
    store: PatternReportStore, sessions: Sessions
) -> None:
    await save(store)

    async with sessions() as session:
        stored = await session.scalar(text("select report from ai.pattern_reports"))
    assert set(stored) == {"patterns", "needs"}
    assert stored["patterns"][0]["citations"] == [
        {"entry_id": 12, "quote": "I'm nervous about this"}
    ]


async def test_first_save_of_the_day_wins(store: PatternReportStore) -> None:
    await save(store)
    later = REPORT.model_copy(update={"patterns": []})

    assert await save(store, report=later) == REPORT
    assert await store.for_day(1, TODAY) == REPORT


async def test_latest_reads_the_newest_day(store: PatternReportStore) -> None:
    assert await store.latest(1) is None
    await save(store, day=TODAY - timedelta(days=2))
    await save(store, day=TODAY - timedelta(days=1))
    await save(store, day=TODAY - timedelta(days=3))

    latest = await store.latest(1)

    assert latest is not None and latest.generated_on == TODAY - timedelta(days=1)
    assert await store.for_day(1, TODAY) is None


async def test_reads_are_scoped_to_the_user(store: PatternReportStore) -> None:
    await save(store, user_id=1)

    assert await store.for_day(2, TODAY) is None
    assert await store.latest(2) is None
