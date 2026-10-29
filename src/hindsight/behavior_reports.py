from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import Field
from sqlalchemy import Update, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.citations import Citation
from hindsight.models import BehaviorReportAttemptRecord, BehaviorReportRecord
from hindsight.vocab import AttemptStatus, BehaviorKind
from hindsight.wire import WireModel

MAX_BEHAVIORS = 5
MAX_LISTED = 50
# earlier reports read for "previous"; reports named by feedback are read on top
HISTORY_LIMIT = 30


# vestry words the figures and skips keys it doesn't know; fractions show as percents
class Figure(WireModel):
    key: str
    value: float = Field(allow_inf_nan=False)
    unit: Literal["count", "money", "fraction", "days"]


# the same identity in the newest earlier saved report
class Previous(WireModel):
    day: date
    figures: list[Figure]


class Behavior(WireModel):
    kind: BehaviorKind
    subject: str | None
    figures: list[Figure]
    previous: Previous | None
    # no digits or number words; the figures carry the numbers
    observation: str
    question: str
    # at least one, except for undocumented, whose subject is a missing entry
    citations: list[Citation]
    # every decision and linked entry in the group
    trade_ids: list[int]
    entry_ids: list[int]


# a kind (and subject) that didn't qualify and what it waits for
class Need(WireModel):
    kind: BehaviorKind
    subject: str | None
    more: int = Field(ge=1)
    reason: Literal["ENTRIES", "TRADES", "PRICES", "FIELDS"]


class BehaviorReport(WireModel):
    # the New York day the report belongs to
    day: date
    generated_at: datetime
    as_of: date
    # the newest entry with extracted fields when the report was built
    entries_up_to: datetime | None
    # false only for an empty report that wasn't stored
    saved: bool
    behaviors: list[Behavior] = Field(max_length=MAX_BEHAVIORS)
    needs: list[Need]


class SavedBehaviorReport(WireModel):
    report: BehaviorReport | None


class BehaviorSummary(WireModel):
    kind: BehaviorKind
    subject: str | None
    figures: list[Figure]


class BehaviorReportSummary(WireModel):
    day: date
    generated_at: datetime
    behaviors: list[BehaviorSummary]


# once a model call starts the day is used; a crashed attempt reads as FAILED
class Attempt(WireModel):
    day: date
    status: Literal["SAVED", "FAILED"]
    at: datetime


class BehaviorReportList(WireModel):
    # newest first
    reports: list[BehaviorReportSummary]
    has_older: bool
    # the user's newest attempt
    last_attempt: Attempt | None


# a behavior identity in an earlier saved report
@dataclass(frozen=True)
class PastBehavior:
    kind: str
    subject: str | None
    figures: list[Figure]
    trade_ids: list[int]


@dataclass(frozen=True)
class PastReport:
    day: date
    behaviors: list[PastBehavior]


class BehaviorReportStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def for_day(self, user_id: int, day: date) -> BehaviorReport | None:
        async with self._sessions() as session:
            record = await session.get(BehaviorReportRecord, (user_id, day))
            return load(record) if record else None

    # the first save of the day wins, so a racing duplicate returns what was already stored
    async def save(
        self, user_id: int, report: BehaviorReport, model: str, prompt_version: str
    ) -> BehaviorReport:
        statement = (
            insert(BehaviorReportRecord)
            .values(
                user_id=user_id,
                day=report.day,
                generated_at=report.generated_at,
                as_of=report.as_of,
                entries_up_to=report.entries_up_to,
                report=report.model_dump(mode="json", include={"behaviors", "needs"}),
                model=model,
                prompt_version=prompt_version,
            )
            .on_conflict_do_nothing()
            .returning(BehaviorReportRecord.day)
        )
        async with self._sessions.begin() as session:
            if await session.scalar(statement) is not None:
                # in the same transaction, so a saved report never sits beside a failed attempt
                await session.execute(finished(user_id, report.day, AttemptStatus.SAVED))
                return report.model_copy(update={"saved": True})
        stored = await self.for_day(user_id, report.day)
        assert stored is not None
        return stored

    # newest first, strictly before `before` when given
    async def page(
        self, user_id: int, before: date | None = None, limit: int = 10
    ) -> BehaviorReportList:
        limit = max(1, min(limit, MAX_LISTED))
        reports = (
            select(BehaviorReportRecord)
            .where(BehaviorReportRecord.user_id == user_id)
            .order_by(BehaviorReportRecord.day.desc())
            .limit(limit + 1)
        )
        if before is not None:
            reports = reports.where(BehaviorReportRecord.day < before)
        attempt = (
            select(BehaviorReportAttemptRecord)
            .where(BehaviorReportAttemptRecord.user_id == user_id)
            .order_by(BehaviorReportAttemptRecord.day.desc())
            .limit(1)
        )
        async with self._sessions() as session:
            records = list(await session.scalars(reports))
            last = await session.scalar(attempt)
        return BehaviorReportList(
            reports=[summarize(r) for r in records[:limit]],
            has_older=len(records) > limit,
            last_attempt=as_attempt(last) if last else None,
        )

    # the newest saved reports before `before`, plus the ones on `days` (feedback's report days)
    async def history(
        self, user_id: int, before: date, days: Iterable[date] = ()
    ) -> list[PastReport]:
        newest = (
            select(BehaviorReportRecord)
            .where(BehaviorReportRecord.user_id == user_id, BehaviorReportRecord.day < before)
            .order_by(BehaviorReportRecord.day.desc())
            .limit(HISTORY_LIMIT)
        )
        named = select(BehaviorReportRecord).where(
            BehaviorReportRecord.user_id == user_id, BehaviorReportRecord.day.in_(list(days))
        )
        async with self._sessions() as session:
            records = {r.day: r for r in await session.scalars(newest)}
            for record in await session.scalars(named):
                records.setdefault(record.day, record)
        return [past(records[day]) for day in sorted(records, reverse=True)]

    # false when the day already has an attempt, so a racing duplicate doesn't call the model
    async def start_attempt(self, user_id: int, day: date) -> bool:
        statement = (
            insert(BehaviorReportAttemptRecord)
            .values(user_id=user_id, day=day)
            .on_conflict_do_nothing()
            .returning(BehaviorReportAttemptRecord.day)
        )
        async with self._sessions.begin() as session:
            return await session.scalar(statement) is not None

    async def finish_attempt(self, user_id: int, day: date, status: AttemptStatus) -> None:
        async with self._sessions.begin() as session:
            await session.execute(finished(user_id, day, status))

    # for a refusal that came after the attempt was recorded but before the model was called
    async def discard_attempt(self, user_id: int, day: date) -> None:
        statement = delete(BehaviorReportAttemptRecord).where(
            BehaviorReportAttemptRecord.user_id == user_id,
            BehaviorReportAttemptRecord.day == day,
            BehaviorReportAttemptRecord.status.is_(None),
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def attempt(self, user_id: int, day: date) -> Attempt | None:
        async with self._sessions() as session:
            record = await session.get(BehaviorReportAttemptRecord, (user_id, day))
            return as_attempt(record) if record else None


def finished(user_id: int, day: date, status: AttemptStatus) -> Update:
    return (
        update(BehaviorReportAttemptRecord)
        .where(
            BehaviorReportAttemptRecord.user_id == user_id,
            BehaviorReportAttemptRecord.day == day,
        )
        .values(status=status, finished_at=func.now())
    )


def load(record: BehaviorReportRecord) -> BehaviorReport:
    return BehaviorReport.model_validate(
        {
            **record.report,
            "day": record.day,
            "generated_at": record.generated_at,
            "as_of": record.as_of,
            "entries_up_to": record.entries_up_to,
            "saved": True,
        }
    )


def summarize(record: BehaviorReportRecord) -> BehaviorReportSummary:
    return BehaviorReportSummary(
        day=record.day,
        generated_at=record.generated_at,
        behaviors=[BehaviorSummary.model_validate(b) for b in record.report["behaviors"]],
    )


def past(record: BehaviorReportRecord) -> PastReport:
    behaviors = [Behavior.model_validate(b) for b in record.report["behaviors"]]
    return PastReport(
        day=record.day,
        behaviors=[PastBehavior(b.kind, b.subject, b.figures, b.trade_ids) for b in behaviors],
    )


# an attempt that never finished counts as failed
def as_attempt(record: BehaviorReportAttemptRecord) -> Attempt:
    return Attempt(
        day=record.day,
        status="SAVED" if record.status == AttemptStatus.SAVED else "FAILED",
        at=record.finished_at or record.started_at,
    )
