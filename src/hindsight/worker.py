import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.config import get_settings
from hindsight.db import get_engine, get_sessionmaker
from hindsight.enrichment import EntryEnricher
from hindsight.insights import InsightStore
from hindsight.llm import get_chat_model
from hindsight.pipeline import EntrySource, process_entry
from hindsight.tracing import init_tracing
from hindsight.wire import WireModel

QUEUE = "entry_events"
VISIBILITY_SECONDS = 60
POLL_SECONDS = 5
BATCH_SIZE = 10
MAX_READS = 5
PAUSE_SECONDS = 600
RETRY_READ_SECONDS = 5

log = logging.getLogger("hindsight.worker")


class EntryEvent(WireModel):
    entry_id: int


@dataclass(frozen=True)
class Message:
    id: int
    reads: int
    body: Any


class MessageQueue(Protocol):
    async def read(self) -> list[Message]: ...
    async def delete(self, message_id: int) -> None: ...
    async def archive(self, message_id: int) -> None: ...


class PgmqQueue:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        name: str = QUEUE,
        visibility_seconds: int = VISIBILITY_SECONDS,
        poll_seconds: int = POLL_SECONDS,
    ) -> None:
        # read_with_poll checks its deadline before reading, so 0 would never return a message
        if poll_seconds < 1:
            raise ValueError("poll_seconds must be at least 1")
        self._sessions = sessions
        self._name = name
        self._visibility = visibility_seconds
        self._poll = poll_seconds

    async def read(self) -> list[Message]:
        async with self._sessions.begin() as session:
            rows = await session.execute(
                text(
                    "select msg_id, read_ct, message"
                    " from pgmq.read_with_poll(:queue, :vt, :qty, :poll)"
                ),
                {
                    "queue": self._name,
                    "vt": self._visibility,
                    "qty": BATCH_SIZE,
                    "poll": self._poll,
                },
            )
            return [Message(row.msg_id, row.read_ct, row.message) for row in rows]

    async def delete(self, message_id: int) -> None:
        await self._call("select pgmq.delete(:queue, cast(:id as bigint))", message_id)

    async def archive(self, message_id: int) -> None:
        await self._call("select pgmq.archive(:queue, cast(:id as bigint))", message_id)

    async def _call(self, sql: str, message_id: int) -> None:
        async with self._sessions.begin() as session:
            await session.execute(text(sql), {"queue": self._name, "id": message_id})


class Worker:
    # logs carry ids and exception classes only, never entry text or provider output
    def __init__(
        self,
        queue: MessageQueue,
        handle: Callable[[int], Awaitable[object]],
        ready: Callable[[], Awaitable[bool]],
        stop: asyncio.Event,
        pause_seconds: float = PAUSE_SECONDS,
    ) -> None:
        self._queue = queue
        self._handle = handle
        self._ready = ready
        self._stop = stop
        self._pause_seconds = pause_seconds

    async def run(self) -> None:
        while not self._stop.is_set():
            await self.run_once()

    async def run_once(self) -> None:
        # checked before reading so a spent budget doesn't burn messages' read counts
        if not await self._ready():
            log.info("budget unavailable, pausing")
            await self._pause(self._pause_seconds)
            return
        try:
            messages = await self._queue.read()
        except Exception as error:
            log.warning("queue read failed (%s)", type(error).__name__)
            await self._pause(RETRY_READ_SECONDS)
            return
        for message in messages:
            # unprocessed messages reappear once their visibility timeout ends
            if self._stop.is_set() or not await self._process(message):
                return

    async def _process(self, message: Message) -> bool:
        if message.reads > MAX_READS:
            log.warning("message %d dead-lettered after %d reads", message.id, MAX_READS)
            await self._queue.archive(message.id)
            return True
        try:
            event = EntryEvent.model_validate(message.body)
        except ValidationError:
            log.warning("message %d is malformed, dead-lettered", message.id)
            await self._queue.archive(message.id)
            return True
        try:
            outcome = await self._handle(event.entry_id)
        except BudgetError:
            log.info("budget refused entry %d, pausing", event.entry_id)
            await self._pause(self._pause_seconds)
            return False
        except Exception as error:
            log.warning(
                "entry %d failed (%s) on read %d",
                event.entry_id,
                type(error).__name__,
                message.reads,
            )
            return True
        await self._queue.delete(message.id)
        log.info("entry %d %s", event.entry_id, outcome)
        return True

    async def _pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), seconds)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    langfuse = init_tracing(settings)
    sessions = get_sessionmaker()
    ledger = BudgetLedger(sessions, Limits.from_settings(settings))
    enricher = EntryEnricher(ledger, get_chat_model(), settings.chat_model)
    source, store = EntrySource(sessions), InsightStore(sessions)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)

    async def handle(entry_id: int) -> object:
        return await process_entry(entry_id, source, store, enricher)

    log.info("worker started")
    try:
        await Worker(PgmqQueue(sessions), handle, ledger.can_start, stop).run()
    finally:
        if langfuse:
            langfuse.shutdown()
        await get_engine().dispose()
        log.info("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
