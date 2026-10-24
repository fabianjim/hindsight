import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetError, BudgetLedger, Limits, total_charged
from hindsight.models import Generation, Status

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
LIMITS = Limits(lifetime_micros=30_000, daily_micros=20_000, daily_generations=5)


class Clock:
    def __init__(self) -> None:
        # still october 1 in new york
        self.now = datetime(2026, 10, 2, 3, 59, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ledger(sessions: Sessions, clock: Clock) -> BudgetLedger:
    return BudgetLedger(sessions, LIMITS, clock)


def key(n: int) -> str:
    return f"{n:064x}"


async def charged(sessions: Sessions) -> int:
    async with sessions() as session:
        return await total_charged(session)


async def job(sessions: Sessions, job_id: uuid.UUID) -> Generation:
    async with sessions() as session:
        found = await session.get(Generation, job_id)
        assert found is not None
        return found


async def test_reserves_before_work_and_deduplicates(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    assert await ledger.can_start(10_000)
    start = await ledger.start(key(1), 10_000)
    assert await charged(sessions) == 10_000
    assert (await job(sessions, start.id)).budget_day == date(2026, 10, 1)

    duplicate = await ledger.start(key(1), 10_000)
    assert not duplicate.created
    assert duplicate.id == start.id

    call = await ledger.reserve_call(start.id, "summary", 4_000)
    with pytest.raises(BudgetError):
        await ledger.reserve_call(start.id, "other", 1_000)
    await ledger.settle_call(call, 900)
    # a replayed settlement must not refund anything
    await ledger.settle_call(call, 0)
    with pytest.raises(BudgetError):
        await ledger.reserve_call(start.id, "summary", 4_000)

    await ledger.finish(start.id, success=True)
    assert await charged(sessions) == 900
    assert await ledger.can_start(10_000)
    assert (await job(sessions, start.id)).status == Status.SUCCEEDED
    assert not (await ledger.start(key(1), 10_000)).created


async def test_concurrent_jobs_share_the_daily_allowance(ledger: BudgetLedger) -> None:
    assert (await ledger.start(key(1), 15_000)).created
    assert await ledger.can_start(5_000)
    assert not await ledger.can_start(5_001)
    with pytest.raises(BudgetError):
        await ledger.start(key(2), 5_001)
    assert (await ledger.start(key(2), 5_000)).created


async def test_uncertain_call_keeps_its_reservation_and_cannot_resume(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    start = await ledger.start(key(1), 10_000)
    await ledger.reserve_call(start.id, "summary", 4_000)
    await ledger.finish(start.id, success=False)

    assert await charged(sessions) == 4_000
    assert (await job(sessions, start.id)).status == Status.UNKNOWN
    with pytest.raises(BudgetError):
        await ledger.reserve_call(start.id, "retry", 4_000)


async def test_abandoned_job_keeps_full_allowance_across_midnight(
    ledger: BudgetLedger, sessions: Sessions, clock: Clock
) -> None:
    abandoned = await ledger.start(key(1), 15_000)
    await ledger.reserve_call(abandoned.id, "summary", 4_000)
    clock.now += timedelta(minutes=6)

    assert not (await ledger.start(key(1), 15_000)).created
    assert (await job(sessions, abandoned.id)).status == Status.UNKNOWN
    with pytest.raises(BudgetError):
        await ledger.reserve_call(abandoned.id, "retry", 1_000)
    with pytest.raises(BudgetError):
        await ledger.start(key(2), 15_001)
    assert (await ledger.start(key(2), 15_000)).created
    assert await charged(sessions) == 30_000

    # a late worker must not release a crashed job's allowance
    await ledger.finish(abandoned.id, success=False)
    assert await charged(sessions) == 30_000


async def test_enforces_daily_money_and_job_allowance(ledger: BudgetLedger) -> None:
    first = await ledger.start(key(1), 15_000)
    with pytest.raises(BudgetError):
        await ledger.reserve_call(first.id, "oversized", 15_001)
    call = await ledger.reserve_call(first.id, "summary", 15_000)
    await ledger.settle_call(call, 15_000)
    await ledger.finish(first.id, success=True)

    with pytest.raises(BudgetError):
        await ledger.start(key(2), 5_001)
    assert (await ledger.start(key(2), 5_000)).created


async def test_failures_still_consume_daily_generation_slots(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    for n in range(1, 6):
        await ledger.finish((await ledger.start(key(n), 1_000)).id, success=False)

    assert await charged(sessions) == 0
    assert not await ledger.can_start(1_000)
    with pytest.raises(BudgetError):
        await ledger.start(key(6), 1_000)


async def test_database_lock_prevents_concurrent_overspend(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    async def attempt(n: int) -> bool:
        try:
            return (await ledger.start(key(n), 15_000)).created
        except BudgetError:
            return False

    assert sorted(await asyncio.gather(attempt(1), attempt(2))) == [False, True]
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Generation)) == 1
    assert await charged(sessions) == 15_000


async def test_unexpected_overrun_stops_all_spending(
    ledger: BudgetLedger, sessions: Sessions
) -> None:
    start = await ledger.start(key(1), 10_000)
    call = await ledger.reserve_call(start.id, "summary", 4_000)
    await ledger.settle_call(call, 4_001)

    with pytest.raises(BudgetError):
        await ledger.reserve_call(start.id, "other", 1_000)
    await ledger.finish(start.id, success=False)
    assert await charged(sessions) == 4_001
    assert not await ledger.can_start(1_000)
    with pytest.raises(BudgetError):
        await ledger.start(key(2), 1_000)


async def test_uncapped_lifetime_still_enforces_daily_money(
    sessions: Sessions, clock: Clock
) -> None:
    ledger = BudgetLedger(sessions, Limits(-1, 20_000, 5), clock)
    for day in range(3):
        clock.now = datetime(2026, 10, 2, 12, tzinfo=UTC) + timedelta(days=day)
        assert await ledger.can_start(20_000)
        start = await ledger.start(key(day + 1), 20_000)
        await ledger.settle_call(await ledger.reserve_call(start.id, "digest", 20_000), 20_000)
        await ledger.finish(start.id, success=True)
        assert not await ledger.can_start(1)
        with pytest.raises(BudgetError):
            await ledger.start(key(99), 1)

    assert await charged(sessions) == 60_000


async def test_uncapped_lifetime_still_enforces_daily_generation_count(
    sessions: Sessions, clock: Clock
) -> None:
    ledger = BudgetLedger(sessions, Limits(-1, 20_000, 2), clock)
    for n in (1, 2):
        await ledger.finish((await ledger.start(key(n), 1_000)).id, success=False)

    assert await charged(sessions) == 0
    assert not await ledger.can_start(1_000)
    with pytest.raises(BudgetError):
        await ledger.start(key(3), 1_000)


async def test_missing_budget_row_fails_closed(ledger: BudgetLedger, sessions: Sessions) -> None:
    async with sessions.begin() as session:
        await session.execute(text("delete from ai.budget"))

    assert not await ledger.can_start(1_000)
    with pytest.raises(BudgetError):
        await ledger.start(key(1), 1_000)
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Generation)) == 0


@pytest.mark.parametrize(
    ("request_key", "allowance"), [("not-a-key", 1_000), ("A" * 64, 1_000), (key(1), 0)]
)
async def test_rejects_malformed_reservations(
    ledger: BudgetLedger, request_key: str, allowance: int
) -> None:
    with pytest.raises(ValueError):
        await ledger.start(request_key, allowance)
