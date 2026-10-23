import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetLedger, Limits
from hindsight.enrichment import EntryEnricher, EntryInput
from hindsight.insights import InsightStore
from hindsight.models import Generation
from hindsight.pipeline import EntrySource, Outcome, SourceEntry, process_entry
from hindsight.vocab import EntryType
from tests.fakes import FakeToolModel

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
BODY = "Bought NVDA after the dip. Data center demand looks durable."
WHEN = datetime(2026, 3, 3, 15, 30, tzinfo=UTC)


def reply(ticker: str) -> AIMessage:
    extraction = {
        "thesis": None,
        "exit_condition": None,
        "horizon": None,
        "conviction": None,
        "emotions": [],
        "idea_source": None,
        "tickers": [ticker],
        "themes": [],
    }
    return AIMessage(
        content=json.dumps(extraction),
        usage_metadata={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550},
    )


class Pipeline:
    def __init__(self, sessions: Sessions, *tickers: str) -> None:
        self.sessions = sessions
        self.model = FakeToolModel(messages=iter([reply(t) for t in tickers]))
        ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 100))
        self.enricher = EntryEnricher(ledger, self.model, MODEL)
        self.store = InsightStore(sessions)
        self.source = EntrySource(sessions)

    async def run(self, entry_id: int = 7) -> Outcome:
        return await process_entry(entry_id, self.source, self.store, self.enricher)

    async def write(self, sql: str, **params: object) -> None:
        async with self.sessions.begin() as session:
            await session.execute(text(sql), params)

    async def add_entry(self, body: str = BODY, entry_id: int = 7) -> None:
        await self.write(
            "insert into core_read.journal_entries"
            " (id, user_id, entry_type, body, ticker, timestamp, price_snapshot, tags)"
            " values (:id, 1, 'BUY', :body, 'NVDA', :when, 118.0, '{semis}')",
            id=entry_id,
            body=body,
            when=WHEN,
        )

    async def tickers(self, entry_id: int = 7) -> list[str] | None:
        found = await self.store.for_user(1, [entry_id])
        return found[entry_id].tickers if entry_id in found else None

    async def jobs(self) -> int:
        async with self.sessions() as session:
            return await session.scalar(select(func.count()).select_from(Generation)) or 0


async def test_reads_entries_from_the_vestry_view(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.add_entry()

    assert await pipeline.source.get(7) == SourceEntry(
        user_id=1,
        entry=EntryInput(
            id=7,
            entry_type=EntryType.BUY,
            body=BODY,
            ticker="NVDA",
            timestamp=WHEN,
            price_snapshot=118.0,
            tags=["semis"],
        ),
    )
    assert await pipeline.source.get(8) is None


async def test_saves_an_insight_for_a_new_entry(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions, "NVDA")
    await pipeline.add_entry()

    assert await pipeline.run() == Outcome.SAVED
    assert await pipeline.tickers() == ["NVDA"]


async def test_unchanged_entry_spends_nothing(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions, "NVDA")
    await pipeline.add_entry()
    await pipeline.run()
    await pipeline.write("update core_read.journal_entries set price_snapshot = 999")

    assert await pipeline.run() == Outcome.UNCHANGED
    assert len(pipeline.model.calls) == 1
    assert await pipeline.jobs() == 1


async def test_edited_entry_is_extracted_again(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions, "NVDA", "AMD")
    await pipeline.add_entry()
    await pipeline.run()
    await pipeline.write("update core_read.journal_entries set body = 'Switched to AMD.'")

    assert await pipeline.run() == Outcome.SAVED
    assert await pipeline.tickers() == ["AMD"]


async def test_deleted_entry_loses_its_insight(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions, "NVDA")
    await pipeline.add_entry()
    await pipeline.run()
    await pipeline.write("delete from core_read.journal_entries")

    assert await pipeline.run() == Outcome.REMOVED
    assert await pipeline.tickers() is None
    assert len(pipeline.model.calls) == 1


@pytest.mark.parametrize("body", ["", "   \n"])
async def test_blank_entry_spends_nothing(sessions: Sessions, body: str) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.add_entry(body=body)

    assert await pipeline.run() == Outcome.REMOVED
    assert pipeline.model.calls == []
    assert await pipeline.jobs() == 0


class EditedMidRun:
    def __init__(self, *versions: SourceEntry | None) -> None:
        self.versions = list(versions)

    async def get(self, entry_id: int) -> SourceEntry | None:
        return self.versions.pop(0)


def version(body: str) -> SourceEntry:
    return SourceEntry(1, EntryInput(id=7, entry_type=EntryType.BUY, body=body, timestamp=WHEN))


@pytest.mark.parametrize("latest", [version("Actually I sold it."), None])
async def test_never_saves_over_an_entry_that_changed_mid_run(
    sessions: Sessions, latest: SourceEntry | None
) -> None:
    pipeline = Pipeline(sessions, "NVDA")
    source = EditedMidRun(version(BODY), latest)

    assert await process_entry(7, source, pipeline.store, pipeline.enricher) == Outcome.STALE
    assert await pipeline.tickers() is None


async def test_enrich_refuses_blank_text_before_reserving(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)

    with pytest.raises(ValueError):
        await pipeline.enricher.enrich(version("  ").entry, user_id=1)
    assert await pipeline.jobs() == 0
