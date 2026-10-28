from datetime import date, datetime
from typing import Literal

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.citations import Citation
from hindsight.models import PatternReportRecord
from hindsight.vocab import PatternKind
from hindsight.wire import WireModel

MAX_PATTERNS = 5


# vestry words the figures; vestry ignores keys it doesn't know
class Figure(WireModel):
    key: str
    value: float
    unit: Literal["count", "days"]


class Pattern(WireModel):
    kind: PatternKind
    subject: str | None
    figures: list[Figure]
    # no digits; the figures carry the numbers
    observation: str
    question: str
    # at least one, except for silent_trades, whose subject is a missing entry
    citations: list[Citation]


# a kind that didn't qualify and how many more examples it needs
class Need(WireModel):
    kind: PatternKind
    more: int = Field(ge=1)


class PatternReport(WireModel):
    generated_on: date
    as_of: date
    # the newest entry with extracted fields when the report was built
    entries_up_to: datetime | None
    # false only for an empty report that wasn't stored
    saved: bool
    patterns: list[Pattern] = Field(max_length=MAX_PATTERNS)
    needs: list[Need]


class PatternReportStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def for_day(self, user_id: int, day: date) -> PatternReport | None:
        async with self._sessions() as session:
            record = await session.get(PatternReportRecord, (user_id, day))
            return load(record) if record else None

    # the newest report from any day; reading never generates
    async def latest(self, user_id: int) -> PatternReport | None:
        statement = (
            select(PatternReportRecord)
            .where(PatternReportRecord.user_id == user_id)
            .order_by(PatternReportRecord.day.desc())
            .limit(1)
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
            return load(record) if record else None

    # the first save of the day wins, so a racing duplicate returns what was already stored
    async def save(
        self, user_id: int, report: PatternReport, model: str, prompt_version: str
    ) -> PatternReport:
        statement = (
            insert(PatternReportRecord)
            .values(
                user_id=user_id,
                day=report.generated_on,
                report=report.model_dump(mode="json", include={"patterns", "needs"}),
                as_of=report.as_of,
                entries_up_to=report.entries_up_to,
                model=model,
                prompt_version=prompt_version,
            )
            .on_conflict_do_nothing()
            .returning(PatternReportRecord.day)
        )
        async with self._sessions.begin() as session:
            if await session.scalar(statement) is not None:
                return report.model_copy(update={"saved": True})
        stored = await self.for_day(user_id, report.generated_on)
        assert stored is not None
        return stored


def load(record: PatternReportRecord) -> PatternReport:
    return PatternReport.model_validate(
        {
            **record.report,
            "generated_on": record.day,
            "as_of": record.as_of,
            "entries_up_to": record.entries_up_to,
            "saved": True,
        }
    )
