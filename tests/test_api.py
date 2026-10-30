from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.api import MAX_ENTRY_IDS, get_store
from hindsight.citations import Citation
from hindsight.enrichment import Claim, Enrichment, EntryInsight
from hindsight.insights import InsightStore
from hindsight.main import app
from hindsight.vocab import Emotion, Horizon
from tests.conftest import SERVICE_TOKEN

pytestmark = pytest.mark.anyio

INSIGHT = EntryInsight(
    thesis=Claim(
        summary="Data center demand stays strong",
        citations=[Citation(entry_id=7, quote="Data center demand looks durable")],
    ),
    exit_condition=None,
    horizon=Horizon.YEARS,
    conviction=4,
    emotions=[Emotion.FOMO],
    idea_source=None,
    tickers=["NVDA"],
    themes=["AI capex"],
)


@pytest.fixture
async def client(sessions: async_sessionmaker[AsyncSession]) -> AsyncIterator[httpx.AsyncClient]:
    store = InsightStore(sessions)
    for entry_id, user_id in [(7, 1), (8, 1), (9, 2)]:
        record = Enrichment(entry_id, INSIGHT, "a" * 64, "model", "entry-insight-v1")
        await store.save(record, user_id)
    app.dependency_overrides[get_store] = lambda: store
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://hindsight",
        headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        yield client
    app.dependency_overrides.clear()


async def get(client: httpx.AsyncClient, user_id: int, entry_ids: list[int]) -> httpx.Response:
    return await client.get(f"/v1/users/{user_id}/insights", params={"entryIds": entry_ids})


async def test_returns_the_users_insights_in_camel_case(client: httpx.AsyncClient) -> None:
    response = await get(client, 1, [8, 7, 7])

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert [item["entryId"] for item in body["insights"]] == [7, 8]
    assert body["insights"][0]["insight"] == {
        "thesis": {
            "summary": "Data center demand stays strong",
            "citations": [{"entryId": 7, "quote": "Data center demand looks durable"}],
        },
        "exitCondition": None,
        "horizon": "years",
        "conviction": 4,
        "emotions": ["fomo"],
        "ideaSource": None,
        "tickers": ["NVDA"],
        "themes": ["AI capex"],
        "reasoningBasis": None,
        "thesisOutcome": None,
    }


async def test_never_returns_another_users_insights(client: httpx.AsyncClient) -> None:
    response = await get(client, 1, [9, 404])

    assert response.status_code == 200
    assert response.json() == {"insights": []}


@pytest.mark.parametrize("entry_ids", [[], list(range(MAX_ENTRY_IDS + 1))])
async def test_rejects_empty_or_oversized_batches(
    client: httpx.AsyncClient, entry_ids: list[int]
) -> None:
    assert (await get(client, 1, entry_ids)).status_code == 422


async def test_requires_the_service_token(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/v1/users/1/insights", params={"entryIds": [7]}, headers={"Authorization": ""}
    )

    assert response.status_code == 401
