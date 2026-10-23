import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetError
from hindsight.pipeline import Outcome, process_entry
from hindsight.worker import MAX_READS, Message, PgmqQueue, Worker
from tests.test_pipeline import Pipeline

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]


class Handler:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.seen: list[int] = []

    async def __call__(self, entry_id: int) -> object:
        self.seen.append(entry_id)
        if self.error:
            raise self.error
        return Outcome.SAVED


async def no_sweep() -> object:
    raise AssertionError("sweep not expected")


async def always() -> bool:
    return True


async def never() -> bool:
    return False


def worker(
    sessions: Sessions, handle: Handler, ready: Callable[[], Awaitable[bool]] = always
) -> Worker:
    # visibility 0 makes a failed message readable again on the next pass
    queue = PgmqQueue(sessions, visibility_seconds=0, poll_seconds=1)
    return Worker(queue, handle, no_sweep, ready, asyncio.Event(), pause_seconds=0)


async def send(sessions: Sessions, *bodies: object) -> None:
    async with sessions.begin() as session:
        for body in bodies:
            await session.execute(
                text("select pgmq.send('entry_events', cast(:body as jsonb))"),
                {"body": json.dumps(body)},
            )


async def counts(sessions: Sessions) -> tuple[int, int]:
    async with sessions() as session:
        queued = await session.scalar(text("select count(*) from pgmq.q_entry_events"))
        archived = await session.scalar(text("select count(*) from pgmq.a_entry_events"))
    return queued or 0, archived or 0


async def test_handled_messages_are_deleted(sessions: Sessions) -> None:
    handle = Handler()
    await send(sessions, {"entryId": 7}, {"entryId": 8, "userId": 1})

    await worker(sessions, handle).run_once()

    assert handle.seen == [7, 8]
    assert await counts(sessions) == (0, 0)


async def test_failed_message_is_retried_then_dead_lettered(sessions: Sessions) -> None:
    handle = Handler(RuntimeError("boom"))
    await send(sessions, {"entryId": 7})
    consumer = worker(sessions, handle)

    for _ in range(MAX_READS + 1):
        await consumer.run_once()

    assert handle.seen == [7] * MAX_READS
    assert await counts(sessions) == (0, 1)


async def test_malformed_message_is_dead_lettered_untouched(sessions: Sessions) -> None:
    handle = Handler()
    await send(sessions, {"entry": "seven"})

    await worker(sessions, handle).run_once()

    assert handle.seen == []
    assert await counts(sessions) == (0, 1)


async def test_spent_budget_pauses_without_reading(sessions: Sessions) -> None:
    handle = Handler()
    await send(sessions, {"entryId": 7})

    await worker(sessions, handle, ready=never).run_once()

    assert handle.seen == []
    async with sessions() as session:
        assert await session.scalar(text("select read_ct from pgmq.q_entry_events")) == 0


async def test_budget_refusal_keeps_the_batch_for_later(sessions: Sessions) -> None:
    handle = Handler(BudgetError("AI allowance reached"))
    await send(sessions, {"entryId": 7}, {"entryId": 8})

    await worker(sessions, handle).run_once()

    assert handle.seen == [7]
    assert await counts(sessions) == (2, 0)


async def test_stop_leaves_the_rest_of_the_batch(sessions: Sessions) -> None:
    consumer: Worker | None = None
    seen: list[int] = []

    async def handle(entry_id: int) -> object:
        seen.append(entry_id)
        assert consumer
        consumer._stop.set()  # pyright: ignore[reportPrivateUsage]
        return Outcome.SAVED

    await send(sessions, {"entryId": 7}, {"entryId": 8})
    queue = PgmqQueue(sessions, visibility_seconds=0, poll_seconds=1)
    consumer = Worker(queue, handle, no_sweep, always, asyncio.Event(), pause_seconds=0)

    await asyncio.wait_for(consumer.run(), timeout=5)

    assert seen == [7]
    assert await counts(sessions) == (1, 0)


class BrokenQueue:
    async def read(self) -> list[Message]:
        raise ConnectionError("database went away")

    async def delete(self, message_id: int) -> None: ...

    async def archive(self, message_id: int) -> None: ...


def test_rejects_a_poll_that_would_never_read() -> None:
    with pytest.raises(ValueError):
        PgmqQueue(async_sessionmaker(), poll_seconds=0)


async def test_queue_outage_does_not_crash_the_loop() -> None:
    consumer = Worker(BrokenQueue(), Handler(), no_sweep, always, asyncio.Event(), pause_seconds=0)

    await consumer.run_once()


async def test_logs_never_carry_exception_text(
    sessions: Sessions, caplog: pytest.LogCaptureFixture
) -> None:
    handle = Handler(RuntimeError("Bought NVDA because my friend said so"))
    await send(sessions, {"entryId": 7})

    with caplog.at_level(logging.INFO, logger="hindsight.worker"):
        await worker(sessions, handle).run_once()

    assert "entry 7 failed (RuntimeError) on read 1" in caplog.text
    assert "NVDA" not in caplog.text


async def test_events_flow_through_to_saved_insights(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions, "NVDA")
    await pipeline.add_entry()
    await send(sessions, {"entryId": 7})

    async def handle(entry_id: int) -> object:
        return await process_entry(entry_id, pipeline.source, pipeline.store, pipeline.enricher)

    queue = PgmqQueue(sessions, visibility_seconds=0, poll_seconds=1)
    await Worker(queue, handle, no_sweep, always, asyncio.Event(), pause_seconds=0).run_once()

    assert await pipeline.tickers() == ["NVDA"]
    assert await counts(sessions) == (0, 0)


async def test_sweep_message_runs_the_sweep(sessions: Sessions) -> None:
    sweeps: list[str] = []

    async def sweep() -> object:
        sweeps.append("ran")
        return None

    handle = Handler()
    await send(sessions, {"sweep": "entries"}, {"sweep": "everything"})
    queue = PgmqQueue(sessions, visibility_seconds=0, poll_seconds=1)

    await Worker(queue, handle, sweep, always, asyncio.Event(), pause_seconds=0).run_once()

    assert sweeps == ["ran"]
    assert handle.seen == []
    assert await counts(sessions) == (0, 1)
