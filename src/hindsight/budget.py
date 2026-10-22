import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Self
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.config import Settings
from hindsight.models import Budget, Call, Generation, Status

MAX_JOB_TIME = timedelta(minutes=5)
BUDGET_ZONE = ZoneInfo("America/New_York")
REQUEST_KEY = re.compile(r"[a-f0-9]{64}")
STEP = re.compile(r"[a-zA-Z0-9._-]{1,80}")


class BudgetError(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    lifetime_micros: int
    daily_micros: int
    daily_generations: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            settings.lifetime_budget_micros,
            settings.daily_budget_micros,
            settings.daily_generations,
        )

    def permits_lifetime_spend(self, charged: int, additional: int) -> bool:
        return self.lifetime_micros == -1 or additional <= self.lifetime_micros - charged


@dataclass(frozen=True)
class Start:
    id: uuid.UUID
    created: bool


def utcnow() -> datetime:
    return datetime.now(UTC)


# reserve before every model call, settle after; each method is its own transaction
class BudgetLedger:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        limits: Limits,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._sessions = sessions
        self._limits = limits
        self._clock = clock

    # a hint only; start() enforces exact costs under the lock
    async def can_start(self) -> bool:
        async with self._sessions() as session:
            budget = await session.get(Budget, 1)
            if budget is None or budget.blocked:
                return False
            day = budget_day(self._clock())
            return (
                await count_on(session, day) < self._limits.daily_generations
                and self._limits.permits_lifetime_spend(await total_charged(session), 1)
                and await charged_on(session, day) < self._limits.daily_micros
            )

    async def start(self, key: str, allowance: int) -> Start:
        if not REQUEST_KEY.fullmatch(key) or allowance <= 0:
            raise ValueError("invalid AI generation reservation")
        async with self._sessions.begin() as session:
            await lock_available(session)
            now = self._clock()
            # a crashed worker is never replayed and keeps its full reservation, even after midnight
            await session.execute(
                update(Generation)
                .where(Generation.status == Status.RUNNING)
                .where(Generation.started_at <= now - MAX_JOB_TIME)
                .values(status=Status.UNKNOWN)
            )
            existing = await session.scalar(select(Generation).where(Generation.request_key == key))
            if existing:
                return Start(existing.id, False)
            day = budget_day(now)
            if (
                await count_on(session, day) >= self._limits.daily_generations
                or not self._limits.permits_lifetime_spend(await total_charged(session), allowance)
                or allowance > self._limits.daily_micros - await charged_on(session, day)
            ):
                raise BudgetError("AI allowance reached")
            job = Generation(
                request_key=key,
                status=Status.RUNNING,
                started_at=now,
                budget_day=day,
                reserved_micros=allowance,
                charged_micros=allowance,
            )
            session.add(job)
            await session.flush()
            return Start(job.id, True)

    async def reserve_call(self, job_id: uuid.UUID, step: str, allowance: int) -> uuid.UUID:
        if not STEP.fullmatch(step) or allowance <= 0:
            raise ValueError("invalid AI call reservation")
        async with self._sessions.begin() as session:
            await lock_available(session)
            job = await get_job(session, job_id)
            if job.status != Status.RUNNING or job.started_at + MAX_JOB_TIME <= self._clock():
                raise BudgetError("AI generation is no longer active")
            previous = await calls_of(session, job_id)
            if any(call.step == step or not call.settled for call in previous):
                raise BudgetError("AI call already attempted or unresolved")
            used = sum(call.charged_micros for call in previous)
            if allowance > job.reserved_micros - used:
                raise BudgetError("AI generation allowance reached")
            call = Call(
                generation_id=job_id,
                step=step,
                reserved_micros=allowance,
                charged_micros=allowance,
                settled=False,
            )
            session.add(call)
            await session.flush()
            return call.id

    async def settle_call(self, call_id: uuid.UUID, actual_micros: int) -> None:
        async with self._sessions.begin() as session:
            budget = await lock(session)
            call = await session.get(Call, call_id)
            if call is None:
                raise BudgetError("unknown AI call")
            if call.settled:
                return
            if actual_micros < 0:
                raise ValueError("invalid AI usage")
            call.charged_micros = actual_micros
            call.settled = True
            # our cost bound was wrong, so stop all spending until someone reviews it
            if actual_micros > call.reserved_micros:
                budget.blocked = True

    async def finish(self, job_id: uuid.UUID, success: bool) -> None:
        async with self._sessions.begin() as session:
            await lock(session)
            job = await get_job(session, job_id)
            if job.status != Status.RUNNING:
                return
            attempted = await calls_of(session, job_id)
            if any(not call.settled for call in attempted):
                job.status = Status.UNKNOWN
            else:
                job.status = Status.SUCCEEDED if success else Status.FAILED
            job.charged_micros = sum(call.charged_micros for call in attempted)


def budget_day(now: datetime) -> date:
    return now.astimezone(BUDGET_ZONE).date()


async def lock(session: AsyncSession) -> Budget:
    budget = await session.scalar(select(Budget).where(Budget.id == 1).with_for_update())
    if budget is None:
        raise BudgetError("AI budget is not initialized")
    return budget


async def lock_available(session: AsyncSession) -> None:
    if (await lock(session)).blocked:
        raise BudgetError("AI budget requires review")


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Generation:
    job = await session.get(Generation, job_id)
    if job is None:
        raise BudgetError("unknown AI generation")
    return job


async def calls_of(session: AsyncSession, job_id: uuid.UUID) -> list[Call]:
    return list(await session.scalars(select(Call).where(Call.generation_id == job_id)))


async def count_on(session: AsyncSession, day: date) -> int:
    return (
        await session.scalar(
            select(func.count()).select_from(Generation).where(Generation.budget_day == day)
        )
        or 0
    )


async def total_charged(session: AsyncSession) -> int:
    return await session.scalar(select(func.coalesce(func.sum(Generation.charged_micros), 0))) or 0


async def charged_on(session: AsyncSession, day: date) -> int:
    return (
        await session.scalar(
            select(func.coalesce(func.sum(Generation.charged_micros), 0)).where(
                Generation.budget_day == day
            )
        )
        or 0
    )
