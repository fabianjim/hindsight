import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight import api
from hindsight.api import get_narrator, get_pattern_store, get_source, get_store, get_today
from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.enrichment import Enrichment, EntryInsight
from hindsight.insights import InsightStore
from hindsight.main import app
from hindsight.pattern_report import PatternNarrator
from hindsight.pattern_reports import PatternReport, PatternReportStore
from hindsight.patterns import MAX_OUTCOMES
from hindsight.pipeline import EntrySource
from hindsight.vocab import Emotion
from tests.conftest import SERVICE_TOKEN
from tests.fakes import FakeToolModel
from tests.test_pattern_report import reply

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
TODAY = date(2026, 10, 27)
URL = "/v1/users/1/pattern-report"
T0 = datetime(2026, 3, 2, 15, tzinfo=UTC)
BODIES = {
    1: "Nervous about NVDA but buying more.",
    2: "Anxious after the drop, adding again.",
    3: "Still uneasy. Bought another slice.",
}
ANXIOUS = EntryInsight(
    thesis=None,
    exit_condition=None,
    horizon=None,
    conviction=None,
    emotions=[Emotion.ANXIOUS],
    idea_source=None,
    tickers=["NVDA"],
    themes=[],
)
ITEMS: list[dict[str, Any]] = [
    {
        "index": 0,
        "observation": "Each time you wrote about feeling anxious, you bought soon after.",
        "question": "What did buying settle for you in those moments?",
        "quotes": [{"entry_id": 1, "quote": "Nervous about NVDA"}],
    }
]
OUTCOME = {"entryId": 1, "price": 141.2, "kind": "LATEST", "on": "2026-10-27"}


def at(days: float) -> datetime:
    return T0 + timedelta(days=days)


def trade(trade_id: int, when: datetime) -> dict[str, Any]:
    return {
        "tradeId": trade_id,
        "ticker": "NVDA",
        "side": "BUY",
        "shares": 3.0,
        "price": 120.5,
        "timestamp": when.isoformat(),
        "initial": False,
    }


# a buy a day after each anxious entry: one felt_vs_did pattern and nothing else
TRADE = trade(7, at(1))
REQUEST: dict[str, Any] = {
    "asOf": "2026-10-27",
    "outcomes": [OUTCOME],
    "trades": [TRADE, trade(8, at(11)), trade(9, at(21))],
    "holdingPeriods": [
        {"ticker": "NVDA", "openedAt": at(1).isoformat(), "closedAt": None, "pnl": 62.1}
    ],
    "autoEntryIds": [],
}


class Harness:
    def __init__(self, sessions: Sessions, limits: Limits | None = None) -> None:
        self.sessions = sessions
        self.store = PatternReportStore(sessions)
        self.insights = InsightStore(sessions)
        self.model = FakeToolModel(messages=iter([reply(ITEMS), reply(ITEMS)]))
        ledger = BudgetLedger(sessions, limits or Limits(-1, 1_000_000, 100))
        self.narrator: PatternNarrator | None = PatternNarrator(ledger, self.model, MODEL)

    async def add_entry(
        self, entry_id: int, user_id: int, body: str, when: datetime, anxious: bool = True
    ) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                text(
                    "insert into core_read.journal_entries"
                    " (id, user_id, entry_type, body, ticker, timestamp, price_snapshot, tags)"
                    " values (:id, :user_id, 'BUY', :body, 'NVDA', :when, 118.0, '{}')"
                ),
                {"id": entry_id, "user_id": user_id, "body": body, "when": when},
            )
        if anxious:
            record = Enrichment(entry_id, ANXIOUS, "a" * 64, MODEL, "entry-insight-v1")
            await self.insights.save(record, user_id)

    def sent(self) -> list[dict[str, Any]]:
        return json.loads(self.model.prompts[0][1].text)["patterns"]


@pytest.fixture
async def harness(sessions: Sessions) -> AsyncIterator[Harness]:
    harness = Harness(sessions)
    for entry_id, body in BODIES.items():
        await harness.add_entry(entry_id, 1, body, at((entry_id - 1) * 10))
    # another user's anxious entries at the same times
    for entry_id, body in BODIES.items():
        await harness.add_entry(50 + entry_id, 2, body, at((entry_id - 1) * 10))
    app.dependency_overrides[get_pattern_store] = lambda: harness.store
    app.dependency_overrides[get_store] = lambda: harness.insights
    app.dependency_overrides[get_source] = lambda: EntrySource(sessions)
    app.dependency_overrides[get_narrator] = lambda: harness.narrator
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


async def test_generates_saves_and_returns_camel_case(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 200
    report = response.json()
    assert report["generatedOn"] == "2026-10-27"
    assert report["asOf"] == "2026-10-27"
    assert report["entriesUpTo"] == "2026-03-22T15:00:00Z"
    assert report["saved"] is True
    assert report["patterns"] == [
        {
            "kind": "felt_vs_did",
            "subject": "anxious",
            "figures": [
                {"key": "entries", "value": 3, "unit": "count"},
                {"key": "followed_by_buy", "value": 3, "unit": "count"},
                {"key": "followed_by_sell", "value": 0, "unit": "count"},
            ],
            "observation": ITEMS[0]["observation"],
            "question": ITEMS[0]["question"],
            "citations": [{"entryId": 1, "quote": "Nervous about NVDA"}],
        }
    ]
    assert {n["kind"] for n in report["needs"]} == {
        "outcomes_by_state",
        "horizon_vs_holding",
        "gains_vs_losses",
        "silent_trades",
        "open_theses",
    }
    assert (await client.get(URL)).json() == {"report": report}


async def test_second_request_of_the_day_reuses_the_report(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    first = await client.post(URL, json=REQUEST)
    second = await client.post(URL, json={**REQUEST, "trades": []})

    assert second.json() == first.json()
    assert len(harness.model.calls) == 1


async def test_nothing_qualifying_returns_an_unsaved_report_without_spending(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(URL, json={**REQUEST, "trades": []})

    assert response.status_code == 200
    report = response.json()
    assert (report["saved"], report["patterns"]) == (False, [])
    assert {"kind": "felt_vs_did", "more": 1} in report["needs"]
    assert harness.model.calls == []
    assert (await client.get(URL)).json() == {"report": None}


async def test_ai_off_refuses_new_reports_but_serves_saved_ones(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    harness.narrator = None
    assert (await client.post(URL, json=REQUEST)).status_code == 503
    assert (await client.get(URL)).json() == {"report": None}

    yesterday = await saved_yesterday(harness)
    assert (await client.get(URL)).json() == {"report": yesterday}
    # an empty report needs no model, so it still works
    assert (await client.post(URL, json={**REQUEST, "trades": []})).status_code == 200


async def saved_yesterday(harness: Harness) -> dict[str, Any]:
    report = PatternReport(
        generated_on=TODAY - timedelta(days=1),
        as_of=TODAY - timedelta(days=1),
        entries_up_to=None,
        saved=True,
        patterns=[],
        needs=[],
    )
    await harness.store.save(1, report, MODEL, "pattern-report-v1")
    return report.model_dump(mode="json", by_alias=True)


async def test_get_reads_the_latest_report(client: httpx.AsyncClient, harness: Harness) -> None:
    yesterday = await saved_yesterday(harness)

    assert (await client.get(URL)).json() == {"report": yesterday}
    assert (await client.get("/v1/users/2/pattern-report")).json() == {"report": None}


async def test_budget_refusal_is_unavailable_and_saves_nothing(
    client: httpx.AsyncClient, harness: Harness, sessions: Sessions
) -> None:
    harness.narrator = Harness(sessions, Limits(-1, 1_000_000, 0)).narrator

    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 503
    assert await harness.store.for_day(1, TODAY) is None


@pytest.mark.parametrize("error", [TimeoutError("secret detail"), BudgetError("secret detail")])
async def test_transient_errors_are_unavailable_and_log_only_the_class(
    client: httpx.AsyncClient,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    async def fail(*_: Any) -> None:
        raise error

    assert harness.narrator is not None
    monkeypatch.setattr(harness.narrator, "narrate", fail)

    with caplog.at_level(logging.WARNING, logger="hindsight.api"):
        response = await client.post(URL, json=REQUEST)

    assert response.status_code == 503
    assert type(error).__name__ in caplog.text
    assert "secret detail" not in caplog.text
    assert await harness.store.for_day(1, TODAY) is None


async def test_narration_that_doesnt_check_out_is_not_saved(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    bad = [{**ITEMS[0], "observation": "You bought 3 times."}]
    harness.model.messages = iter([reply(bad), reply(ITEMS)])

    first = await client.post(URL, json=REQUEST)

    assert first.status_code == 200
    assert (first.json()["saved"], first.json()["patterns"]) == (False, [])
    assert await harness.store.for_day(1, TODAY) is None
    # nothing was used up, so a retry can still succeed today
    assert (await client.post(URL, json=REQUEST)).json()["saved"] is True


async def test_only_the_users_own_entries_are_read(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    # vestry's facts naming another user's entries change nothing
    request = {**REQUEST, "autoEntryIds": [51, 52, 53], "outcomes": [{**OUTCOME, "entryId": 51}]}

    await client.post(URL, json=request)

    [felt] = harness.sent()
    assert [e["id"] for e in felt["entries"]] == [3, 2, 1]


async def test_reads_only_the_newest_entries(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "MAX_ENTRIES", 3)
    await harness.add_entry(4, 1, "An older note.", at(-30), anxious=False)
    # trades from before the read entries would all look silent if they counted
    old_trades = [trade(20 + i, at(-60 + i * 10)) for i in range(3)]

    await client.post(URL, json={**REQUEST, "trades": REQUEST["trades"] + old_trades})

    assert [p["kind"] for p in harness.sent()] == ["felt_vs_did"]
    assert [e["id"] for e in harness.sent()[0]["entries"]] == [3, 2, 1]


async def test_requires_the_service_token(harness: Harness) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://hindsight"
    ) as anonymous:
        for method in ("GET", "POST"):
            response = await anonymous.request(method, URL, json=REQUEST)
            assert response.status_code == 401
    assert harness.model.calls == []


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
    client: httpx.AsyncClient, harness: Harness, change: dict[str, Any]
) -> None:
    response = await client.post(URL, json={**REQUEST, **change})

    assert response.status_code == 422
    assert harness.model.calls == []
