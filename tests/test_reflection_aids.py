from datetime import date, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.citations import Citation
from hindsight.enrichment import Claim
from hindsight.reflection_aids import DAILY_AIDS, ReflectionAid, ReflectionAidStore

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
TODAY = date(2026, 10, 25)
AID = ReflectionAid(
    recap=[
        Claim(
            summary="You later called the dip overdone",
            citations=[Citation(entry_id=52, quote="the dip was overdone")],
        )
    ],
    questions=["What made this look like a dip rather than the start of a decline?"],
)


@pytest.fixture
def store(sessions: Sessions) -> ReflectionAidStore:
    return ReflectionAidStore(sessions)


async def save(
    store: ReflectionAidStore,
    user_id: int = 1,
    entry_id: int = 7,
    day: date = TODAY,
    aid: ReflectionAid = AID,
) -> ReflectionAid:
    return await store.save(user_id, entry_id, day, aid, "openai:gpt-4.1-mini", "reflection-aid-v1")


async def test_round_trips_an_aid(store: ReflectionAidStore) -> None:
    assert await save(store) == AID

    assert await store.for_day(1, 7, TODAY) == AID


async def test_round_trips_an_empty_aid(store: ReflectionAidStore) -> None:
    empty = ReflectionAid(recap=[], questions=[])

    await save(store, aid=empty)

    assert await store.for_day(1, 7, TODAY) == empty


async def test_first_save_of_the_day_wins(store: ReflectionAidStore) -> None:
    await save(store)
    later = ReflectionAid(recap=[], questions=["A different question?"])

    assert await save(store, aid=later) == AID
    assert await store.for_day(1, 7, TODAY) == AID


async def test_a_new_day_is_a_new_aid(store: ReflectionAidStore) -> None:
    await save(store, day=TODAY - timedelta(days=1))

    assert await store.for_day(1, 7, TODAY) is None


async def test_reads_are_scoped_to_the_user(store: ReflectionAidStore) -> None:
    await save(store, user_id=1)

    assert await store.for_day(2, 7, TODAY) is None


async def test_remaining_counts_only_that_users_day(store: ReflectionAidStore) -> None:
    assert await store.remaining(1, TODAY) == DAILY_AIDS
    for entry_id in range(3):
        await save(store, entry_id=entry_id)
    await save(store, user_id=2)
    await save(store, day=TODAY - timedelta(days=1))

    assert await store.remaining(1, TODAY) == DAILY_AIDS - 3


async def test_remaining_never_goes_negative(store: ReflectionAidStore) -> None:
    for entry_id in range(DAILY_AIDS + 1):
        await save(store, entry_id=entry_id)

    assert await store.remaining(1, TODAY) == 0
