from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import openai
from sqlalchemy import exc, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetError
from hindsight.enrichment import EntryEnricher, EntryInput
from hindsight.insights import InsightStore

# failures of the same text before it's treated as poison and never tried again
POISON_AFTER = 3

# outages and infrastructure, not the entry; the queue and sweep retry these freely.
# the sdk wraps transport failures in these classes, langchain's included
# provider-specific: update when switching providers
TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    exc.OperationalError,
    exc.InterfaceError,
    ConnectionError,
    TimeoutError,
)

# vestry owns this view; its columns are part of the contract between the repos
ENTRY_QUERY = text(
    "select id, user_id, entry_type, body, ticker, timestamp, price_snapshot, tags"
    " from core_read.journal_entries where id = :id"
)


@dataclass(frozen=True)
class SourceEntry:
    user_id: int
    entry: EntryInput


class EntryReader(Protocol):
    async def get(self, entry_id: int) -> SourceEntry | None: ...


class EntrySource:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, entry_id: int) -> SourceEntry | None:
        async with self._sessions() as session:
            row = (await session.execute(ENTRY_QUERY, {"id": entry_id})).mappings().first()
        if row is None:
            return None
        entry = EntryInput(
            id=row["id"],
            entry_type=row["entry_type"],
            body=row["body"],
            ticker=row["ticker"],
            timestamp=row["timestamp"],
            price_snapshot=row["price_snapshot"],
            tags=row["tags"] or [],
        )
        return SourceEntry(user_id=row["user_id"], entry=entry)


class Outcome(StrEnum):
    SAVED = "saved"
    UNCHANGED = "unchanged"
    STALE = "stale"
    REMOVED = "removed"
    POISONED = "poisoned"


async def process_entry(
    entry_id: int, source: EntryReader, store: InsightStore, enricher: EntryEnricher
) -> Outcome:
    found = await source.get(entry_id)
    if found is None or not found.entry.body.strip():
        await store.delete(entry_id)
        return Outcome.REMOVED
    expected = enricher.hash_of(found.entry)
    if await store.current_hash(entry_id) == expected:
        return Outcome.UNCHANGED
    if await store.failures(entry_id, expected) >= POISON_AFTER:
        return Outcome.POISONED

    try:
        result = await enricher.enrich(found.entry, found.user_id)
    except (BudgetError, *TRANSIENT_ERRORS):
        # not this entry's fault
        raise
    except Exception as error:
        await store.record_failure(entry_id, expected, type(error).__name__)
        raise

    # the entry may have been edited or deleted while the model ran; its own event handles that
    latest = await source.get(entry_id)
    if latest is None or enricher.hash_of(latest.entry) != result.input_hash:
        return Outcome.STALE
    await store.save(result, found.user_id)
    await store.clear_failures(entry_id)
    return Outcome.SAVED
