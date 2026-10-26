from datetime import date

from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.enrichment import Claim
from hindsight.models import ReflectionAidRecord
from hindsight.wire import WireModel

# saved aids per user and new york day; failed attempts save nothing, so they don't count
DAILY_AIDS = 10


class ReflectionAid(WireModel):
    recap: list[Claim] = Field(max_length=3)
    questions: list[str] = Field(max_length=3)


class ReflectionAidStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def for_day(self, user_id: int, entry_id: int, day: date) -> ReflectionAid | None:
        async with self._sessions() as session:
            record = await session.get(ReflectionAidRecord, (user_id, day, entry_id))
            return load(record) if record else None

    async def remaining(self, user_id: int, day: date) -> int:
        async with self._sessions() as session:
            used = await session.scalar(
                select(func.count())
                .select_from(ReflectionAidRecord)
                .where(ReflectionAidRecord.user_id == user_id, ReflectionAidRecord.day == day)
            )
        return max(0, DAILY_AIDS - (used or 0))

    # the first save of the day wins, so a racing duplicate returns what was already stored
    async def save(
        self,
        user_id: int,
        entry_id: int,
        day: date,
        aid: ReflectionAid,
        model: str,
        prompt_version: str,
    ) -> ReflectionAid:
        statement = (
            insert(ReflectionAidRecord)
            .values(
                user_id=user_id,
                day=day,
                entry_id=entry_id,
                recap=[claim.model_dump(mode="json") for claim in aid.recap],
                questions=aid.questions,
                model=model,
                prompt_version=prompt_version,
            )
            .on_conflict_do_nothing()
            .returning(ReflectionAidRecord.entry_id)
        )
        async with self._sessions.begin() as session:
            if await session.scalar(statement) is not None:
                return aid
        stored = await self.for_day(user_id, entry_id, day)
        assert stored is not None
        return stored


def load(record: ReflectionAidRecord) -> ReflectionAid:
    return ReflectionAid.model_validate({"recap": record.recap, "questions": record.questions})
