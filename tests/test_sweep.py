import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.pipeline import POISON_AFTER
from hindsight.sweep import ORPHAN_FLOOR, Sweeper
from hindsight.worker import QUEUE
from tests.test_pipeline import Pipeline

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]


def sweeper(pipeline: Pipeline, page_size: int = 500) -> Sweeper:
    return Sweeper(pipeline.sessions, pipeline.enricher.hash_of, QUEUE, page_size)


async def queued(sessions: Sessions) -> list[int]:
    async with sessions() as session:
        rows = await session.scalars(
            text("select (message->>'entryId')::int from pgmq.q_entry_events order by msg_id")
        )
        return list(rows)


async def insight_ids(sessions: Sessions) -> list[int]:
    async with sessions() as session:
        return list(
            await session.scalars(text("select entry_id from ai.entry_insights order by 1"))
        )


async def hash_for(pipeline: Pipeline, entry_id: int) -> str:
    found = await pipeline.source.get(entry_id)
    assert found
    return pipeline.enricher.hash_of(found.entry)


async def store_insight(pipeline: Pipeline, entry_id: int, input_hash: str) -> None:
    await pipeline.write(
        "insert into ai.entry_insights (entry_id, user_id, input_hash, model, prompt_version)"
        " values (:id, 1, :hash, 'm', 'v')",
        id=entry_id,
        hash=input_hash,
    )


async def test_queues_every_entry_without_a_current_insight(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    for entry_id in (1, 2, 3, 4, 5):
        await pipeline.add_entry(entry_id=entry_id)
    await store_insight(pipeline, 2, "0" * 64)
    await store_insight(pipeline, 3, await hash_for(pipeline, 3))

    report = await sweeper(pipeline, page_size=2).run()

    # 2 is stale, 3 is current, the rest have none; page_size 2 forces three pages
    assert await queued(sessions) == [1, 2, 4, 5]
    assert (report.scanned, report.queued) == (5, 4)


async def test_skips_poison_but_not_entries_still_under_the_cap(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    for entry_id, failures in ((1, POISON_AFTER), (2, POISON_AFTER - 1)):
        await pipeline.add_entry(entry_id=entry_id)
        for _ in range(failures):
            await pipeline.store.record_failure(
                entry_id, await hash_for(pipeline, entry_id), "RuntimeError"
            )

    await sweeper(pipeline).run()

    assert await queued(sessions) == [2]


async def test_poison_of_older_text_does_not_block_new_text(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.add_entry(entry_id=1)
    for _ in range(POISON_AFTER):
        await pipeline.store.record_failure(1, "f" * 64, "RuntimeError")

    await sweeper(pipeline).run()

    assert await queued(sessions) == [1]


async def test_blank_entries_are_queued_only_to_clear_an_old_insight(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.add_entry(entry_id=1, body="  ")
    await pipeline.add_entry(entry_id=2, body="  ")
    await store_insight(pipeline, 2, "a" * 64)

    await sweeper(pipeline).run()

    assert await queued(sessions) == [2]


async def test_removes_insights_whose_entry_is_gone(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.add_entry(entry_id=1)
    for entry_id in (1, 98, 99):
        await store_insight(pipeline, entry_id, "a" * 64)
    await pipeline.store.record_failure(99, "b" * 64, "RuntimeError")

    report = await sweeper(pipeline).run()

    assert await insight_ids(sessions) == [1]
    assert await pipeline.store.failure_counts([99]) == {}
    assert (report.orphans_removed, report.orphans_held) == (2, 0)


async def test_removes_reflection_aids_whose_entry_is_gone(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.add_entry(entry_id=1)
    for entry_id in (1, 99):
        await pipeline.write(
            "insert into ai.reflection_aids"
            " (user_id, day, entry_id, recap, questions, model, prompt_version)"
            " values (1, current_date, :id, '[]', '{}', 'm', 'v')",
            id=entry_id,
        )

    await sweeper(pipeline).run()

    async with sessions() as session:
        rows = await session.scalars(text("select entry_id from ai.reflection_aids"))
        assert list(rows) == [1]


async def test_never_mass_deletes_when_the_view_looks_broken(sessions: Sessions) -> None:
    pipeline = Pipeline(sessions)
    await pipeline.write(
        "insert into ai.entry_insights (entry_id, user_id, input_hash, model, prompt_version)"
        " select id, 1, repeat('a', 64), 'm', 'v' from generate_series(1, :n) id",
        n=ORPHAN_FLOOR + 1,
    )

    report = await sweeper(pipeline).run()

    assert len(await insight_ids(sessions)) == ORPHAN_FLOOR + 1
    assert (report.orphans_removed, report.orphans_held) == (0, ORPHAN_FLOOR + 1)
