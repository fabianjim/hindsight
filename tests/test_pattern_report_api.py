from collections.abc import AsyncIterator
from datetime import date
from typing import Any

import httpx
import pytest

from hindsight.api import get_today
from hindsight.main import app
from hindsight.patterns import MAX_OUTCOMES
from tests.conftest import SERVICE_TOKEN

pytestmark = pytest.mark.anyio

TODAY = date(2026, 10, 27)
URL = "/v1/users/1/pattern-report"
OUTCOME = {"entryId": 12, "price": 141.2, "kind": "LATEST", "on": "2026-10-27"}
TRADE = {
    "tradeId": 7,
    "ticker": "NVDA",
    "side": "BUY",
    "shares": 3.0,
    "price": 120.5,
    "timestamp": "2026-03-03T15:00:00Z",
    "initial": False,
}
REQUEST: dict[str, Any] = {
    "asOf": "2026-10-27",
    "outcomes": [OUTCOME],
    "trades": [TRADE],
    "holdingPeriods": [
        {"ticker": "NVDA", "openedAt": "2026-03-03T15:00:00Z", "closedAt": None, "pnl": 62.1}
    ],
    "autoEntryIds": [3, 9],
}


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_today] = lambda: TODAY
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://hindsight",
        headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        yield client
    app.dependency_overrides.clear()


async def test_get_returns_no_report_yet(client: httpx.AsyncClient) -> None:
    response = await client.get(URL)

    assert response.status_code == 200
    assert response.json() == {"report": None}


async def test_post_returns_an_unsaved_empty_report_in_camel_case(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 200
    assert response.json() == {
        "generatedOn": "2026-10-27",
        "asOf": "2026-10-27",
        "entriesUpTo": None,
        "saved": False,
        "patterns": [],
        "needs": [],
    }


async def test_post_accepts_snake_case_and_only_requires_as_of(
    client: httpx.AsyncClient,
) -> None:
    assert (await client.post(URL, json={"as_of": "2026-10-27"})).status_code == 200
    snake = {**REQUEST, "auto_entry_ids": [3], "holding_periods": []}
    assert (await client.post(URL, json=snake)).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"asOf": None},
        {"outcomes": [OUTCOME] * (MAX_OUTCOMES + 1)},
        {"outcomes": [{**OUTCOME, "price": 0}]},
        {"outcomes": [{**OUTCOME, "kind": "CLOSED"}]},
        {"trades": [{**TRADE, "ticker": "nvda"}]},
        {"trades": [{**TRADE, "side": "HOLD"}]},
        {"trades": [{**TRADE, "shares": -1}]},
    ],
)
async def test_post_rejects_malformed_facts(
    client: httpx.AsyncClient, change: dict[str, Any]
) -> None:
    response = await client.post(URL, json={**REQUEST, **change})

    assert response.status_code == 422


@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_requires_the_service_token(client: httpx.AsyncClient, method: str) -> None:
    response = await client.request(method, URL, json=REQUEST, headers={"Authorization": ""})

    assert response.status_code == 401
