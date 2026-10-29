from collections.abc import AsyncIterator
from datetime import date
from typing import Any

import httpx
import pytest

from hindsight.api import get_today
from hindsight.behaviors import MAX_TRADES
from hindsight.main import app
from tests.conftest import SERVICE_TOKEN

pytestmark = pytest.mark.anyio

TODAY = date(2026, 10, 28)
URL = "/v1/users/1/behavior-reports"
BUY: dict[str, Any] = {
    "tradeId": 7,
    "ticker": "NVDA",
    "side": "BUY",
    "shares": 3.0,
    "price": 120.5,
    "timestamp": "2026-03-03T15:00:00Z",
    "initial": False,
    "runUp": 0.142,
    "runUpStatus": "KNOWN",
    "weightAtBuy": 0.18,
    "outcomePrice": 141.2,
    "outcomeKind": "LATEST",
    "outcomeOn": "2026-10-28",
}
SELL: dict[str, Any] = {
    "tradeId": 8,
    "ticker": "NVDA",
    "side": "SELL",
    "shares": 1.0,
    "price": 130.0,
    "timestamp": "2026-04-03T15:00:00Z",
    "initial": False,
    "runUp": None,
    "runUpStatus": "UNAVAILABLE",
    "weightAtBuy": None,
    "outcomePrice": None,
    "outcomeKind": None,
    "outcomeOn": None,
}
# the integration doc's example
REQUEST: dict[str, Any] = {
    "asOf": "2026-10-28",
    "portfolioValue": 48210.5,
    "trades": [BUY, SELL],
    "holdingPeriods": [
        {
            "ticker": "NVDA",
            "openedAt": "2026-03-03T15:00:00Z",
            "closedAt": None,
            "cost": 361.5,
            "pnl": 62.1,
        }
    ],
    "holdings": [
        {
            "ticker": "NVDA",
            "marketValue": 9120.0,
            "weight": 0.19,
            "sector": "Technology",
            "possibleMonthlyLoss": 1480.0,
        }
    ],
    "autoEntryIds": [3, 9],
    "feedback": [
        {
            "kind": "fomo",
            "subject": None,
            "reaction": "DOESNT_FIT",
            "note": "I was rebalancing",
            "reportDay": "2026-10-20",
            "at": "2026-10-20T18:02:00Z",
        }
    ],
    "behaviorNotes": [{"kind": "fomo", "subject": None, "entryId": 88}],
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


def with_trade(**changes: Any) -> dict[str, Any]:
    return {**REQUEST, "trades": [{**BUY, **changes}]}


async def test_stub_returns_an_unsaved_empty_report(client: httpx.AsyncClient) -> None:
    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 200
    report = response.json()
    assert report.pop("generatedAt")
    assert report == {
        "day": "2026-10-28",
        "asOf": "2026-10-28",
        "entriesUpTo": None,
        "saved": False,
        "behaviors": [],
        "needs": [],
    }


async def test_accepts_snake_case_and_omitted_optional_fields(client: httpx.AsyncClient) -> None:
    trade = {k: v for k, v in BUY.items() if not k.startswith(("runUp", "outcome", "weight"))}
    trade["trade_id"] = trade.pop("tradeId")
    response = await client.post(URL, json={"as_of": "2026-10-28", "trades": [trade]})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        with_trade(ticker="nvda"),
        with_trade(shares=0),
        with_trade(price=-1.0),
        # set exactly when the status is KNOWN
        with_trade(runUpStatus="PENDING"),
        with_trade(runUp=None),
        # all three outcome fields or none
        with_trade(outcomeKind=None),
        with_trade(side="HOLD"),
        {**REQUEST, "holdingPeriods": [{**REQUEST["holdingPeriods"][0], "cost": 0}]},
        {**REQUEST, "holdings": [{**REQUEST["holdings"][0], "marketValue": 0}]},
        {**REQUEST, "feedback": [{**REQUEST["feedback"][0], "reaction": "MAYBE"}]},
        {**REQUEST, "trades": [BUY] * (MAX_TRADES + 1)},
    ],
)
async def test_one_bad_row_fails_the_request(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> None:
    response = await client.post(URL, json=body)

    assert response.status_code == 422


# jackson writes non-finite numbers as these strings
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
async def test_rejects_non_finite_fractions(client: httpx.AsyncClient, value: str) -> None:
    response = await client.post(URL, json=with_trade(runUp=value))

    assert response.status_code == 422


async def test_unknown_feedback_kinds_are_accepted(client: httpx.AsyncClient) -> None:
    body = {**REQUEST, "feedback": [{**REQUEST["feedback"][0], "kind": "felt_vs_did"}]}

    response = await client.post(URL, json=body)

    assert response.status_code == 200


async def test_stub_lists_no_reports(client: httpx.AsyncClient) -> None:
    response = await client.get(URL, params={"before": "2026-10-28", "limit": 50})

    assert response.status_code == 200
    assert response.json() == {"reports": [], "hasOlder": False, "lastAttempt": None}


@pytest.mark.parametrize("limit", [0, 51])
async def test_list_limit_is_bounded(client: httpx.AsyncClient, limit: int) -> None:
    response = await client.get(URL, params={"limit": limit})

    assert response.status_code == 422


async def test_stub_has_no_report_for_a_day(client: httpx.AsyncClient) -> None:
    response = await client.get(f"{URL}/2026-10-28")

    assert response.status_code == 200
    assert response.json() == {"report": None}


async def test_v1_routes_are_gone(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/users/1/pattern-report")

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", URL), ("GET", URL), ("GET", f"{URL}/2026-10-28")],
)
async def test_routes_need_the_service_token(method: str, path: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://hindsight"
    ) as client:
        response = await client.request(method, path, json=REQUEST)

    assert response.status_code == 401
