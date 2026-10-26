import json
from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.api import get_aid_store, get_generator, get_source, get_today
from hindsight.budget import BudgetLedger, Limits
from hindsight.main import app
from hindsight.pipeline import EntrySource
from hindsight.reflection import ReflectionAidGenerator
from hindsight.reflection_aids import DAILY_AIDS, ReflectionAid, ReflectionAidStore
from tests.conftest import SERVICE_TOKEN
from tests.fakes import FakeToolModel
from tests.test_reflection import DRAFT, MODEL, reply

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
TODAY = date(2026, 10, 25)
URL = "/v1/users/1/entries/7/reflection-aid"
REQUEST = {
    "laterEntryIds": [52, 53, 60],
    "since": {"priceNow": 142.3, "sharesNow": 15, "trades": []},
}


class Harness:
    def __init__(self, sessions: Sessions, limits: Limits | None = None) -> None:
        self.sessions = sessions
        self.store = ReflectionAidStore(sessions)
        self.model = FakeToolModel(messages=iter([reply(DRAFT)]))
        ledger = BudgetLedger(sessions, limits or Limits(-1, 1_000_000, 100))
        self.generator: ReflectionAidGenerator | None = ReflectionAidGenerator(
            ledger, self.model, MODEL
        )

    async def add_entry(self, entry_id: int, user_id: int, body: str, when: str) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                text(
                    "insert into core_read.journal_entries"
                    " (id, user_id, entry_type, body, ticker, timestamp, price_snapshot, tags)"
                    " values (:id, :user_id, 'INSIGHT', :body, 'NVDA', :when, 118.0, '{}')"
                ),
                {"id": entry_id, "user_id": user_id, "body": body, "when": when},
            )

    def later_ids_sent(self) -> list[int]:
        return [e["id"] for e in json.loads(self.model.prompts[0][1].text)["later_entries"]]


@pytest.fixture
async def harness(sessions: Sessions) -> AsyncIterator[Harness]:
    harness = Harness(sessions)
    await harness.add_entry(7, 1, "Buying the dip on NVDA.", "2026-03-03T15:30:00Z")
    await harness.add_entry(52, 1, "In hindsight the dip was overdone.", "2026-06-01T14:00:00Z")
    await harness.add_entry(53, 2, "Someone else's entry.", "2026-06-02T14:00:00Z")
    await harness.add_entry(60, 1, "Written before the entry.", "2026-01-01T14:00:00Z")
    app.dependency_overrides[get_aid_store] = lambda: harness.store
    app.dependency_overrides[get_source] = lambda: EntrySource(sessions)
    app.dependency_overrides[get_generator] = lambda: harness.generator
    app.dependency_overrides[get_today] = lambda: TODAY
    yield harness
    app.dependency_overrides.clear()


@pytest.fixture
async def client(harness: Harness) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://hindsight",
        headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        yield client


async def use_up_today(harness: Harness) -> None:
    empty = ReflectionAid(recap=[], questions=[])
    for entry_id in range(1000, 1000 + DAILY_AIDS):
        await harness.store.save(1, entry_id, TODAY, empty, MODEL, "reflection-aid-v1")


async def test_generates_saves_and_returns_camel_case(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 200
    assert response.json() == {
        "aid": {
            "recap": [
                {
                    "summary": "You later called the dip overdone",
                    "citations": [{"entryId": 52, "quote": "the dip was overdone"}],
                }
            ],
            "questions": DRAFT["questions"],
        },
        "generatedOn": "2026-10-25",
        "remainingToday": DAILY_AIDS - 1,
    }
    assert await harness.store.for_day(1, 7, TODAY) is not None


async def test_sends_only_the_users_later_entries(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await client.post(URL, json=REQUEST)

    # 53 belongs to someone else and 60 predates the entry
    assert harness.later_ids_sent() == [52]


async def test_second_request_of_the_day_reuses_the_aid(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    first = await client.post(URL, json=REQUEST)
    second = await client.post(URL, json=REQUEST)

    assert second.json() == first.json()
    assert len(harness.model.calls) == 1


async def test_another_users_entry_is_not_found(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post("/v1/users/2/entries/7/reflection-aid", json=REQUEST)

    assert response.status_code == 404
    assert harness.model.calls == []


async def test_used_up_day_is_refused_without_spending(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await use_up_today(harness)

    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 429
    assert harness.model.calls == []


async def test_todays_aid_is_still_served_at_the_limit(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await client.post(URL, json=REQUEST)
    await use_up_today(harness)

    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 200
    assert response.json()["remainingToday"] == 0


async def test_ai_off_refuses_new_aids_but_serves_saved_ones(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await client.post(URL, json=REQUEST)
    harness.generator = None

    assert (await client.post(URL, json=REQUEST)).status_code == 200
    response = await client.post("/v1/users/1/entries/52/reflection-aid", json=REQUEST)
    assert response.status_code == 503


async def test_budget_refusal_is_unavailable_and_saves_nothing(
    client: httpx.AsyncClient, harness: Harness, sessions: Sessions
) -> None:
    harness.generator = Harness(sessions, Limits(-1, 1_000_000, 0)).generator

    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 503
    assert await harness.store.remaining(1, TODAY) == DAILY_AIDS


async def test_request_defaults_to_no_later_entries_or_facts(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(URL, json={})

    assert response.status_code == 200
    assert harness.later_ids_sent() == []


async def test_requires_the_service_token(harness: Harness) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://hindsight"
    ) as anonymous:
        response = await anonymous.post(URL, json=REQUEST)

    assert response.status_code == 401
    assert harness.model.calls == []


def test_no_api_key_means_no_generator() -> None:
    get_generator.cache_clear()
    try:
        assert get_generator() is None
    finally:
        get_generator.cache_clear()
