import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.enrichment import EntryInput
from hindsight.insights import InsightStore
from hindsight.pipeline import POISON_AFTER, EntrySource, SourceEntry

PAGE_SIZE = 500
# a broken or empty view must never wipe paid-for insights
ORPHAN_SHARE_LIMIT = 0.10
ORPHAN_FLOOR = 50

ORPHANS = "where not exists (select 1 from core_read.journal_entries e where e.id = t.entry_id)"

log = logging.getLogger("hindsight.sweep")


@dataclass(frozen=True)
class SweepReport:
    scanned: int
    queued: int
    orphans_removed: int
    orphans_held: int


class Sweeper:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        hash_of: Callable[[EntryInput], str],
        queue: str,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self._sessions = sessions
        self._hash_of = hash_of
        self._queue = queue
        self._page_size = page_size
        self._source = EntrySource(sessions)
        self._store = InsightStore(sessions)

    async def run(self) -> SweepReport:
        scanned = queued = 0
        after = 0
        while page := await self._source.page(after, self._page_size):
            due = await self._due(page)
            await self._enqueue(due)
            scanned, queued, after = scanned + len(page), queued + len(due), page[-1].entry.id
        removed, held = await self._remove_orphans()
        report = SweepReport(scanned, queued, removed, held)
        log.info(
            "swept %d entries, queued %d, removed %d orphans, held %d",
            report.scanned,
            report.queued,
            report.orphans_removed,
            report.orphans_held,
        )
        return report

    # the same rules process_entry applies, checked in bulk so nothing is queued for no reason
    async def _due(self, page: list[SourceEntry]) -> list[int]:
        ids = [found.entry.id for found in page]
        stored = await self._store.hashes(ids)
        failures = await self._store.failure_counts(ids)
        due: list[int] = []
        for found in page:
            entry_id = found.entry.id
            if not found.entry.body.strip():
                if entry_id in stored:
                    due.append(entry_id)
                continue
            current = self._hash_of(found.entry)
            if stored.get(entry_id) == current:
                continue
            failed_hash, count = failures.get(entry_id, ("", 0))
            if failed_hash == current and count >= POISON_AFTER:
                continue
            due.append(entry_id)
        return due

    async def _enqueue(self, entry_ids: list[int]) -> None:
        if not entry_ids:
            return
        messages = [json.dumps({"entryId": entry_id}) for entry_id in entry_ids]
        async with self._sessions.begin() as session:
            await session.execute(
                text("select pgmq.send_batch(:queue, cast(:messages as jsonb[]))"),
                {"queue": self._queue, "messages": messages},
            )

    async def _remove_orphans(self) -> tuple[int, int]:
        async with self._sessions.begin() as session:
            total = await session.scalar(text("select count(*) from ai.entry_insights")) or 0
            orphans = (
                await session.scalar(text(f"select count(*) from ai.entry_insights t {ORPHANS}"))
                or 0
            )
            if orphans > max(ORPHAN_FLOOR, total * ORPHAN_SHARE_LIMIT):
                log.warning("%d of %d insights look orphaned, not deleting", orphans, total)
                return 0, orphans
            for table in ("ai.entry_insights", "ai.entry_failures", "ai.reflection_aids"):
                await session.execute(text(f"delete from {table} t {ORPHANS}"))
        return orphans, 0
