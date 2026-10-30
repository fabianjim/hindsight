import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
from langchain_core.outputs import ChatResult
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight import api
from hindsight.api import get_narrator, get_report_store, get_source, get_store, get_today
from hindsight.behavior_report import BehaviorNarrator
from hindsight.behavior_reports import Behavior, BehaviorReport, BehaviorReportStore, Figure
from hindsight.behaviors import MAX_TRADES
from hindsight.budget import BudgetError, BudgetLedger, Limits
from hindsight.insights import InsightStore
from hindsight.main import app
from hindsight.pipeline import EntrySource
from hindsight.vocab import BehaviorKind
from tests.conftest import SERVICE_TOKEN
from tests.fakes import FakeToolModel
from tests.test_behavior_report import MODEL, reply

pytestmark = pytest.mark.anyio

Sessions = async_sessionmaker[AsyncSession]

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


def buy(trade_id: int, month: int, ticker: str) -> dict[str, Any]:
    return {
        **BUY,
        "tradeId": trade_id,
        "ticker": ticker,
        "timestamp": f"2026-0{month}-03T15:00:00Z",
        "runUp": None,
        "runUpStatus": "UNAVAILABLE",
        "weightAtBuy": None,
        "outcomePrice": 90.0,
        "price": 100.0,
    }


# three buys with no entries around them: only undocumented qualifies
UNDOCUMENTED: dict[str, Any] = {
    "asOf": "2026-10-28",
    "trades": [buy(7, 3, "NVDA"), buy(8, 4, "AMD"), buy(9, 5, "TSLA")],
    "behaviorNotes": [
        {"kind": "undocumented", "subject": None, "entryId": 88},
        {"kind": "undocumented", "subject": None, "entryId": 89},
    ],
}
ITEMS: list[dict[str, Any]] = [
    {
        "index": 0,
        "observation": "None of these buys has a note written around it.",
        "question": "What was on your mind when you made them?",
        "quotes": [],
    }
]


class FailingModel(FakeToolModel):
    error: Exception

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        self.calls.append(kwargs)
        raise self.error


class Harness:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions
        self.store = BehaviorReportStore(sessions)
        self.limits = Limits(-1, 1_000_000, 100)
        self.model: FakeToolModel = FakeToolModel(messages=iter([reply(ITEMS)]))
        self.ai_on = True

    def narrator(self) -> BehaviorNarrator | None:
        if not self.ai_on:
            return None
        return BehaviorNarrator(BudgetLedger(self.sessions, self.limits), self.model, MODEL)

    def fail_with(self, error: Exception) -> None:
        self.model = FailingModel(messages=iter([]), error=error)

    async def add_entry(self, entry_id: int, user_id: int, body: str, when: str) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                text(
                    "insert into core_read.journal_entries"
                    " (id, user_id, entry_type, body, ticker, timestamp, price_snapshot, tags)"
                    " values (:id, :user_id, 'INSIGHT', :body, 'MSFT', :when, null, '{}')"
                ),
                {"id": entry_id, "user_id": user_id, "body": body, "when": when},
            )

    def sent(self) -> list[dict[str, Any]]:
        return json.loads(self.model.prompts[0][1].text)["behaviors"]

    async def save(
        self, day: date, trade_ids: list[int], user_id: int = 1, attempt: bool = True
    ) -> None:
        behavior = Behavior(
            kind=BehaviorKind.UNDOCUMENTED,
            subject=None,
            figures=[Figure(key="decisions", value=len(trade_ids), unit="count")],
            previous=None,
            observation="Earlier.",
            question="Earlier?",
            citations=[],
            trade_ids=trade_ids,
            entry_ids=[],
        )
        report = BehaviorReport(
            day=day,
            generated_at=datetime(2026, 10, 1, tzinfo=UTC),
            as_of=day,
            entries_up_to=None,
            saved=True,
            behaviors=[behavior],
            needs=[],
        )
        if attempt:
            await self.store.start_attempt(user_id, day)
        await self.store.save(user_id, report, MODEL, "behavior-report-v3")


@pytest.fixture
async def harness(sessions: Sessions) -> AsyncIterator[Harness]:
    harness = Harness(sessions)
    app.dependency_overrides[get_report_store] = lambda: harness.store
    app.dependency_overrides[get_source] = lambda: EntrySource(sessions)
    app.dependency_overrides[get_store] = lambda: InsightStore(sessions)
    app.dependency_overrides[get_narrator] = harness.narrator
    app.dependency_overrides[get_today] = lambda: TODAY
    yield harness
    app.dependency_overrides.clear()


@pytest.fixture
async def client(harness: Harness) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://hindsight",
        headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        yield client


async def last_attempt(client: httpx.AsyncClient) -> dict[str, Any] | None:
    return (await client.get(URL)).json()["lastAttempt"]


def with_trade(**changes: Any) -> dict[str, Any]:
    return {**REQUEST, "trades": [{**BUY, **changes}]}


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


async def test_generates_saves_and_returns_camel_case(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 200
    report = response.json()
    assert report.pop("generatedAt")
    assert report == {
        "day": "2026-10-28",
        "asOf": "2026-10-28",
        "entriesUpTo": None,
        "saved": True,
        "behaviors": [
            {
                "kind": "undocumented",
                "subject": None,
                "figures": [
                    {"key": "decisions", "value": 3.0, "unit": "count"},
                    {"key": "pnl", "value": -90.0, "unit": "money"},
                    {"key": "avg_pnl", "value": -30.0, "unit": "money"},
                    {"key": "avg_return", "value": -0.1, "unit": "fraction"},
                    {"key": "gains", "value": 0.0, "unit": "count"},
                ],
                "previous": None,
                "observation": ITEMS[0]["observation"],
                "question": ITEMS[0]["question"],
                "citations": [],
                "tradeIds": [9, 8, 7],
                "entryIds": [],
            }
        ],
        "needs": report["needs"],
    }
    assert report["needs"]
    assert await harness.store.for_day(1, TODAY) is not None
    attempt = await last_attempt(client)
    assert attempt is not None
    assert (attempt["day"], attempt["status"]) == ("2026-10-28", "SAVED")


async def test_a_saved_report_is_returned_without_a_model_call(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    first = await client.post(URL, json=UNDOCUMENTED)

    second = await client.post(URL, json=UNDOCUMENTED)

    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(harness.model.calls) == 1


async def test_nothing_qualifying_is_unsaved_and_records_no_attempt(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(URL, json=REQUEST)

    assert response.status_code == 200
    assert (response.json()["saved"], response.json()["behaviors"]) == (False, [])
    assert response.json()["needs"]
    assert harness.model.calls == []
    assert await last_attempt(client) is None


async def test_ai_off_is_unavailable_and_records_no_attempt(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    harness.ai_on = False

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 503
    assert await last_attempt(client) is None


async def test_ai_off_still_serves_a_saved_report(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await harness.save(TODAY, [9, 8, 7])
    harness.ai_on = False

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 200
    assert response.json()["saved"] is True


async def test_budget_refusal_is_unavailable_and_records_no_attempt(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    harness.limits = Limits(-1, 1_000_000, 0)

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 503
    assert harness.model.calls == []
    assert await last_attempt(client) is None
    # nothing counted, so a retry can still succeed today
    harness.limits = Limits(-1, 1_000_000, 100)
    assert (await client.post(URL, json=UNDOCUMENTED)).json()["saved"] is True


async def test_a_refusal_after_the_attempt_started_records_nothing(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(*_: Any, **__: Any) -> None:
        raise BudgetError("budget is blocked")

    # the middleware's own reservation is refused once the attempt is recorded
    monkeypatch.setattr(BudgetLedger, "reserve_call", refuse)

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 503
    assert await last_attempt(client) is None


async def test_narration_that_doesnt_check_out_fails_the_day(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    bad = [{**ITEMS[0], "observation": "You made 3 buys without a note."}]
    harness.model = FakeToolModel(messages=iter([reply(bad), reply(ITEMS)]))

    first = await client.post(URL, json=UNDOCUMENTED)

    assert first.status_code == 409
    assert first.json() == {"detail": first.json()["detail"], "code": "ATTEMPTED_TODAY"}
    assert await harness.store.for_day(1, TODAY) is None
    assert (await last_attempt(client) or {})["status"] == "FAILED"
    # the day is used
    second = await client.post(URL, json=UNDOCUMENTED)
    assert second.status_code == 409
    assert len(harness.model.calls) == 1


@pytest.mark.parametrize("error", [TimeoutError("secret detail"), ConnectionError("secret")])
async def test_provider_errors_fail_the_day_and_log_only_the_class(
    client: httpx.AsyncClient,
    harness: Harness,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    harness.fail_with(error)

    with caplog.at_level(logging.WARNING, logger="hindsight.api"):
        response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 409
    assert response.json()["code"] == "ATTEMPTED_TODAY"
    assert type(error).__name__ in caplog.text
    assert "secret" not in caplog.text
    assert (await last_attempt(client) or {})["status"] == "FAILED"


async def test_an_unexpected_error_fails_the_day(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    harness.fail_with(RuntimeError("bug"))

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 500
    assert (await last_attempt(client) or {})["status"] == "FAILED"
    assert (await client.post(URL, json=UNDOCUMENTED)).status_code == 409


async def test_an_attempt_left_by_a_crash_counts_as_failed(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await harness.store.start_attempt(1, TODAY)

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 409
    assert response.json()["code"] == "ATTEMPTED_TODAY"
    assert harness.model.calls == []
    assert (await last_attempt(client) or {})["status"] == "FAILED"


async def test_losing_the_attempt_race_returns_the_winners_report(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = harness.store.start_attempt

    # another request starts the day and saves between this one's checks and its model call
    async def raced(user_id: int, day: date) -> bool:
        await start(user_id, day)
        await harness.save(day, [1, 2, 3], user_id, attempt=False)
        return await start(user_id, day)

    monkeypatch.setattr(harness.store, "start_attempt", raced)

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 200
    assert response.json()["behaviors"][0]["tradeIds"] == [1, 2, 3]
    assert harness.model.calls == []


async def test_losing_the_attempt_race_to_a_running_one_is_attempted(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = harness.store.start_attempt

    async def raced(user_id: int, day: date) -> bool:
        await start(user_id, day)
        return await start(user_id, day)

    monkeypatch.setattr(harness.store, "start_attempt", raced)

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.status_code == 409
    assert harness.model.calls == []


async def test_the_users_own_notes_reach_the_model(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await harness.add_entry(88, 1, "I skip writing when I'm in a hurry.", "2026-09-01T12:00:00Z")
    await harness.add_entry(89, 2, "Someone else's note.", "2026-09-01T12:00:00Z")
    body = {
        **UNDOCUMENTED,
        "feedback": [
            {
                "kind": "undocumented",
                "subject": None,
                "reaction": "FITS",
                "note": "Guilty.",
                "reportDay": "2026-10-20",
                "at": "2026-10-20T18:02:00Z",
            }
        ],
    }

    await client.post(URL, json=body)

    [sent] = harness.sent()
    assert sent["notes"] == ["Guilty.", "I skip writing when I'm in a hurry."]


async def test_notes_older_than_the_entries_read_still_reach_the_model(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "MAX_ENTRIES", 1)
    await harness.add_entry(88, 1, "An old note.", "2025-01-01T12:00:00Z")
    await harness.add_entry(90, 1, "A newer entry.", "2026-01-01T12:00:00Z")

    await client.post(URL, json=UNDOCUMENTED)

    assert harness.sent()[0]["notes"] == ["An old note."]


async def test_previous_comes_from_the_newest_earlier_report(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await harness.save(date(2026, 10, 14), [8, 7])
    await harness.save(date(2026, 10, 21), [9, 8, 7, 6])

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.json()["behaviors"][0]["previous"] == {
        "day": "2026-10-21",
        "figures": [{"key": "decisions", "value": 4.0, "unit": "count"}],
    }


@pytest.mark.parametrize(("trade_ids", "shown"), [([9, 8, 7], False), ([8, 7], True)])
async def test_doesnt_fit_hides_a_behavior_until_its_trades_change(
    client: httpx.AsyncClient, harness: Harness, trade_ids: list[int], shown: bool
) -> None:
    await harness.save(date(2026, 10, 20), trade_ids)
    body = {
        **UNDOCUMENTED,
        "feedback": [
            {
                "kind": "undocumented",
                "subject": None,
                "reaction": "DOESNT_FIT",
                "note": None,
                "reportDay": "2026-10-20",
                "at": "2026-10-20T18:02:00Z",
            }
        ],
    }

    response = await client.post(URL, json=body)

    assert response.json()["saved"] is shown


async def test_another_users_reports_and_attempts_are_never_used(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    await harness.save(TODAY, [1, 2, 3], user_id=2)
    await harness.store.start_attempt(2, TODAY - timedelta(days=1))

    response = await client.post(URL, json=UNDOCUMENTED)

    assert response.json()["behaviors"][0]["tradeIds"] == [9, 8, 7]
    assert response.json()["behaviors"][0]["previous"] is None
    assert (await client.get(f"{URL}/2026-10-28")).json()["report"]["behaviors"][0]["tradeIds"] == [
        9,
        8,
        7,
    ]
    listing = (await client.get(URL)).json()
    assert [r["day"] for r in listing["reports"]] == ["2026-10-28"]


async def test_lists_summaries_newest_first_with_pages(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    for day in (date(2026, 10, 14), date(2026, 10, 21), date(2026, 10, 7)):
        await harness.save(day, [8, 7])

    first = (await client.get(URL, params={"limit": 2})).json()
    second = (await client.get(URL, params={"limit": 2, "before": "2026-10-14"})).json()

    assert [r["day"] for r in first["reports"]] == ["2026-10-21", "2026-10-14"]
    assert first["hasOlder"] is True
    assert first["reports"][0]["behaviors"] == [
        {
            "kind": "undocumented",
            "subject": None,
            "figures": [{"key": "decisions", "value": 2.0, "unit": "count"}],
        }
    ]
    assert first["lastAttempt"]["day"] == "2026-10-21"
    assert first["lastAttempt"]["status"] == "SAVED"
    assert [r["day"] for r in second["reports"]] == ["2026-10-07"]
    assert second["hasOlder"] is False


async def test_reads_a_day_without_generating(client: httpx.AsyncClient, harness: Harness) -> None:
    await harness.save(date(2026, 10, 21), [8, 7])
    harness.ai_on = False

    saved = (await client.get(f"{URL}/2026-10-21")).json()["report"]
    missing = (await client.get(f"{URL}/2026-10-22")).json()

    assert saved["saved"] is True and saved["behaviors"][0]["tradeIds"] == [8, 7]
    assert missing == {"report": None}
    assert harness.model.calls == []


@pytest.mark.parametrize("limit", [0, 51])
async def test_list_limit_is_bounded(client: httpx.AsyncClient, limit: int) -> None:
    response = await client.get(URL, params={"limit": limit})

    assert response.status_code == 422


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
