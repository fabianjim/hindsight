import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.citations import Citation
from hindsight.enrichment import Claim, Enrichment, EntryInsight
from hindsight.insights import InsightStore
from hindsight.vocab import Emotion, Horizon, IdeaSource

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
INSIGHT = EntryInsight(
    thesis=Claim(
        summary="Data center demand stays strong for years",
        citations=[Citation(entry_id=7, quote="Data center demand looks durable")],
    ),
    exit_condition=None,
    horizon=Horizon.YEARS,
    conviction=4,
    emotions=[Emotion.FOMO, Emotion.CONFIDENT],
    idea_source=IdeaSource.OWN_RESEARCH,
    tickers=["NVDA", "BRK.B"],
    themes=["AI capex"],
)


def enrichment(entry_id: int = 7, input_hash: str = "a" * 64, **changes: object) -> Enrichment:
    return Enrichment(
        entry_id=entry_id,
        insight=INSIGHT.model_copy(update=changes),
        input_hash=input_hash,
        model="openai:gpt-4.1-mini-2025-04-14",
        prompt_version="entry-insight-v1",
    )


@pytest.fixture
def store(sessions: Sessions) -> InsightStore:
    return InsightStore(sessions)


async def test_round_trips_an_insight(store: InsightStore) -> None:
    assert await store.save(enrichment(), user_id=1)

    assert await store.for_user(1, [7]) == {7: INSIGHT}
    assert await store.current_hash(7) == "a" * 64


async def test_round_trips_an_empty_insight(store: InsightStore) -> None:
    empty = EntryInsight(
        thesis=None,
        exit_condition=None,
        horizon=None,
        conviction=None,
        emotions=[],
        idea_source=None,
        tickers=[],
        themes=[],
    )
    record = Enrichment(7, empty, "a" * 64, "openai:gpt-4.1-mini-2025-04-14", "entry-insight-v1")

    assert await store.save(record, user_id=1)
    assert await store.for_user(1, [7]) == {7: empty}


async def test_same_input_is_not_rewritten(store: InsightStore, sessions: Sessions) -> None:
    await store.save(enrichment(), user_id=1)
    async with sessions() as session:
        before = await session.scalar(text("select updated_at from ai.entry_insights"))

    assert not await store.save(enrichment(conviction=1), user_id=1)

    async with sessions() as session:
        assert await session.scalar(text("select updated_at from ai.entry_insights")) == before
    assert (await store.for_user(1, [7]))[7].conviction == 4


async def test_new_input_replaces_the_insight(store: InsightStore) -> None:
    await store.save(enrichment(), user_id=1)

    assert await store.save(enrichment(input_hash="b" * 64, conviction=2), user_id=1)

    assert (await store.for_user(1, [7]))[7].conviction == 2
    assert await store.current_hash(7) == "b" * 64


async def test_reads_are_scoped_to_the_user(store: InsightStore) -> None:
    await store.save(enrichment(entry_id=7), user_id=1)
    await store.save(enrichment(entry_id=8), user_id=2)

    assert set(await store.for_user(1, [7, 8, 9])) == {7}
    assert await store.for_user(2, [7]) == {}
    assert await store.for_user(1, []) == {}
    assert await store.current_hash(9) is None


@pytest.mark.parametrize(
    ("column", "value"),
    [("conviction", "6"), ("horizon", "'decades'"), ("idea_source", "'tiktok'")],
)
async def test_database_rejects_values_outside_the_vocabulary(
    store: InsightStore, sessions: Sessions, column: str, value: str
) -> None:
    await store.save(enrichment(), user_id=1)

    with pytest.raises(IntegrityError):
        async with sessions.begin() as session:
            await session.execute(text(f"update ai.entry_insights set {column} = {value}"))
