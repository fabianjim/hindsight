from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import pytest

from hindsight.behavior_reports import Figure, Need, Previous
from hindsight.behaviors import (
    BehaviorNote,
    BehaviorReportRequest,
    Feedback,
    Findings,
    FoundBehavior,
    Holding,
    HoldingPeriod,
    LedgerTrade,
    PastBehavior,
    PastReport,
    find_behaviors,
)
from hindsight.citations import Citation
from hindsight.enrichment import Claim, EntryInput, EntryInsight
from hindsight.vocab import (
    BehaviorKind,
    Emotion,
    EntryType,
    Horizon,
    IdeaSource,
    ReasoningBasis,
    ThesisOutcome,
)

NY = ZoneInfo("America/New_York")
# a monday, 10am in new york
T0 = datetime(2026, 3, 2, 15, tzinfo=UTC)
SECOND = timedelta(seconds=1)
MINUTE = timedelta(minutes=1)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
AS_OF = date(2026, 10, 27)
REPORT_DAY = date(2026, 10, 20)

NO_PLAN = BehaviorKind.NO_PLAN
UNDOCUMENTED = BehaviorKind.UNDOCUMENTED
EMOTIONAL = BehaviorKind.EMOTIONAL

Reaction = Literal["FITS", "KNEW", "DOESNT_FIT"]


# --- builders ---


def buy(
    trade_id: int,
    at: datetime = T0,
    *,
    ticker: str = "NVDA",
    outcome: float | None = None,
    run_up: float | None = None,
    pending: bool = False,
    weight: float | None = None,
    initial: bool = False,
) -> LedgerTrade:
    known = outcome is not None
    return LedgerTrade(
        trade_id=trade_id,
        ticker=ticker,
        side="BUY",
        shares=1,
        price=100,
        timestamp=at,
        initial=initial,
        run_up=run_up,
        run_up_status="KNOWN" if run_up is not None else "PENDING" if pending else "UNAVAILABLE",
        weight_at_buy=weight,
        outcome_price=outcome,
        outcome_kind="LATEST" if known else None,
        outcome_on=AS_OF if known else None,
    )


def sell(trade_id: int, at: datetime = T0, *, ticker: str = "NVDA") -> LedgerTrade:
    return LedgerTrade(
        trade_id=trade_id,
        ticker=ticker,
        side="SELL",
        shares=1,
        price=100,
        timestamp=at,
        initial=False,
    )


def entry(
    entry_id: int, at: datetime = T0, *, ticker: str | None = "NVDA", body: str = "Some words."
) -> EntryInput:
    return EntryInput(
        id=entry_id, entry_type=EntryType.INSIGHT, body=body, ticker=ticker, timestamp=at
    )


def insight(
    *emotions: Emotion,
    conviction: Literal[1, 2, 3, 4, 5] | None = None,
    horizon: Horizon | None = None,
    idea_source: IdeaSource | None = None,
    basis: ReasoningBasis | None = None,
    outcome: ThesisOutcome | None = None,
    thesis: bool = False,
    exit: bool = False,
    tickers: Sequence[str] = (),
) -> EntryInsight:
    claim = Claim(summary="A claim", citations=[Citation(entry_id=1, quote="words")])
    return EntryInsight(
        thesis=claim if thesis else None,
        exit_condition=claim if exit else None,
        horizon=horizon,
        conviction=conviction,
        emotions=list(emotions),
        idea_source=idea_source,
        tickers=list(tickers),
        themes=[],
        reasoning_basis=basis,
        thesis_outcome=outcome,
    )


def period(
    ticker: str, opened: datetime, closed: datetime | None, pnl: float | None, cost: float = 100
) -> HoldingPeriod:
    return HoldingPeriod(ticker=ticker, opened_at=opened, closed_at=closed, cost=cost, pnl=pnl)


def holding(
    ticker: str, weight: float, *, sector: str | None = None, loss: float | None = None
) -> Holding:
    return Holding(
        ticker=ticker,
        market_value=1000,
        weight=weight,
        sector=sector,
        possible_monthly_loss=loss,
    )


def feedback(
    kind: str,
    reaction: Reaction,
    *,
    subject: str | None = None,
    day: date = REPORT_DAY,
    at: datetime = T0,
    note: str | None = None,
) -> Feedback:
    return Feedback(kind=kind, subject=subject, reaction=reaction, note=note, report_day=day, at=at)


def note(entry_id: int, kind: str = "no_plan", subject: str | None = None) -> BehaviorNote:
    return BehaviorNote(kind=kind, subject=subject, entry_id=entry_id)


def past(day: date, *behaviors: PastBehavior) -> PastReport:
    return PastReport(day=day, behaviors=list(behaviors))


def request(
    trades: Sequence[LedgerTrade] = (),
    *,
    periods: Sequence[HoldingPeriod] = (),
    holdings: Sequence[Holding] = (),
    auto: Sequence[int] = (),
    feedback: Sequence[Feedback] = (),
    notes: Sequence[BehaviorNote] = (),
) -> BehaviorReportRequest:
    return BehaviorReportRequest(
        as_of=AS_OF,
        trades=list(trades),
        holding_periods=list(periods),
        holdings=list(holdings),
        auto_entry_ids=list(auto),
        feedback=list(feedback),
        behavior_notes=list(notes),
    )


# trades with the entries and insights the user wrote around them
@dataclass
class Log:
    trades: list[LedgerTrade] = field(default_factory=list[LedgerTrade])
    entries: list[EntryInput] = field(default_factory=list[EntryInput])
    insights: dict[int, EntryInsight] = field(default_factory=dict[int, EntryInsight])

    def __add__(self, other: Log) -> Log:
        return Log(
            [*self.trades, *other.trades],
            [*self.entries, *other.entries],
            {**self.insights, **other.insights},
        )


# trade n on its own ticker Tn at T0 + n hours, with entry 100 + n a minute before it.
# None means the entry isn't extracted yet
def decisions(
    *said: EntryInsight | None,
    start: int = 1,
    outcome: float | None = None,
    run_up: float | None = None,
    weight: float | None = None,
) -> Log:
    log = Log()
    for n, insight_ in enumerate(said, start):
        at = T0 + n * HOUR
        log.trades.append(buy(n, at, ticker=f"T{n}", outcome=outcome, run_up=run_up, weight=weight))
        log.entries.append(entry(100 + n, at - MINUTE, ticker=f"T{n}"))
        if insight_ is not None:
            log.insights[100 + n] = insight_
    return log


# trades nobody wrote about, laid out like decisions
def bare(
    *ids: int,
    side: Literal["BUY", "SELL"] = "BUY",
    outcome: float | None = None,
    run_up: float | None = None,
    pending: bool = False,
    weight: float | None = None,
) -> Log:
    return Log(
        [
            buy(n, T0 + n * HOUR, ticker=f"T{n}", outcome=outcome, run_up=run_up,
                pending=pending, weight=weight)
            if side == "BUY"
            else sell(n, T0 + n * HOUR, ticker=f"T{n}")
            for n in ids
        ]
    )  # fmt: skip


def find(
    log: Log,
    *,
    periods: Sequence[HoldingPeriod] = (),
    holdings: Sequence[Holding] = (),
    auto: Sequence[int] = (),
    feedback: Sequence[Feedback] = (),
    notes: Sequence[BehaviorNote] = (),
    history: Sequence[PastReport] = (),
    entries_from: datetime | None = None,
) -> Findings:
    facts = request(
        log.trades,
        periods=periods,
        holdings=holdings,
        auto=auto,
        feedback=feedback,
        notes=notes,
    )
    return find_behaviors(facts, log.entries, log.insights, history, entries_from)


def get(found: Findings, kind: BehaviorKind, subject: str | None = None) -> FoundBehavior:
    [behavior] = [b for b in found.behaviors if (b.kind, b.subject) == (kind, subject)]
    return behavior


def need(found: Findings, kind: BehaviorKind, subject: str | None = None) -> Need | None:
    return next((n for n in found.needs if (n.kind, n.subject) == (kind, subject)), None)


def identities(found: Findings) -> list[tuple[BehaviorKind, str | None]]:
    return [(b.kind, b.subject) for b in found.behaviors]


def figures(behavior: FoundBehavior) -> list[tuple[str, float, str]]:
    return [(f.key, f.value, f.unit) for f in behavior.figures]


def keys(behavior: FoundBehavior) -> list[str]:
    return [f.key for f in behavior.figures]


def standard(
    decisions: int, pnl: float, avg_pnl: float, avg_return: float, gains: int, prefix: str = ""
) -> list[tuple[str, float, str]]:
    return [
        (f"{prefix}decisions", decisions, "count"),
        (f"{prefix}pnl", pnl, "money"),
        (f"{prefix}avg_pnl", avg_pnl, "money"),
        (f"{prefix}avg_return", avg_return, "fraction"),
        (f"{prefix}gains", gains, "count"),
    ]


def baseline(
    decisions: int, pnl: float, avg_pnl: float, avg_return: float, gains: int
) -> list[tuple[str, float, str]]:
    return standard(decisions, pnl, avg_pnl, avg_return, gains, "baseline_")


PLANNED = [insight(thesis=True), insight(exit=True), insight(horizon=Horizon.WEEKS)]


def test_nothing_needs_every_kind() -> None:
    found = find(Log())

    assert found.behaviors == []
    assert [(n.kind, n.subject, n.more) for n in found.needs] == [
        (NO_PLAN, None, 3),
        (UNDOCUMENTED, None, 3),
        *((EMOTIONAL, e, 3) for e in ("anxious", "excited", "regret", "frustrated")),
        *((kind, None, 3) for kind in list(BehaviorKind)[3:]),
    ]
    assert found.entries_up_to is None


# --- kinds ---


def test_no_plan_compares_buys_without_a_plan_to_buys_with_one() -> None:
    log = (
        decisions(insight(), insight(Emotion.CALM), outcome=110)
        + decisions(insight(), start=3, outcome=90)
        + decisions(*PLANNED, start=4, outcome=120)
    )

    found = find(log)

    no_plan = get(found, NO_PLAN)
    assert figures(no_plan) == standard(3, 10, 3.33, 0.0333, 2) + baseline(3, 60, 20, 0.2, 3)
    assert no_plan.trade_ids == [3, 2, 1]
    assert no_plan.entry_ids == [103, 102, 101]


def test_no_plan_short_waits_for_entries() -> None:
    found = find(decisions(insight(), insight()))

    assert need(found, NO_PLAN) == Need(kind=NO_PLAN, subject=None, more=1, reason="ENTRIES")


def test_undocumented_compares_trades_without_entries_to_trades_with_them() -> None:
    log = (
        bare(1, 2, outcome=110)
        + bare(3, side="SELL")
        + decisions(insight(), insight(), insight(), start=4, outcome=90)
    )

    undocumented = get(find(log), UNDOCUMENTED)

    # the sell counts as a decision with an unknown outcome
    assert figures(undocumented) == standard(3, 20, 10, 0.1, 2) + baseline(3, -30, -10, -0.1, 0)
    assert undocumented.trade_ids == [3, 2, 1]
    assert undocumented.entry_ids == []


def test_undocumented_short_waits_for_trades() -> None:
    found = find(bare(1))

    assert need(found, UNDOCUMENTED) == Need(
        kind=UNDOCUMENTED, subject=None, more=2, reason="TRADES"
    )


def test_emotional_compares_trades_naming_the_emotion_to_trades_naming_none() -> None:
    log = (
        decisions(
            insight(Emotion.ANXIOUS),
            insight(Emotion.ANXIOUS, Emotion.CALM),
            insight(Emotion.ANXIOUS, Emotion.EXCITED),
            outcome=110,
        )
        + decisions(insight(Emotion.CALM), insight(), insight(Emotion.FOMO), start=4, outcome=90)
        # names excited, so it's in neither of anxious's groups
        + decisions(insight(Emotion.EXCITED), start=7, outcome=500)
    )

    found = find(log)

    anxious = get(found, EMOTIONAL, "anxious")
    assert figures(anxious) == standard(3, 30, 10, 0.1, 3) + baseline(3, -30, -10, -0.1, 0)
    assert need(found, EMOTIONAL, "excited") == Need(
        kind=EMOTIONAL, subject="excited", more=1, reason="ENTRIES"
    )
    assert need(found, EMOTIONAL, "regret") == Need(
        kind=EMOTIONAL, subject="regret", more=3, reason="ENTRIES"
    )


@pytest.mark.parametrize(
    "emotion", [Emotion.ANXIOUS, Emotion.EXCITED, Emotion.REGRET, Emotion.FRUSTRATED]
)
def test_emotional_has_one_group_per_emotion_and_counts_sells(emotion: Emotion) -> None:
    log = decisions(insight(emotion))
    log += Log(
        [sell(n, T0 + n * HOUR, ticker=f"T{n}") for n in (2, 3)],
        [entry(100 + n, T0 + n * HOUR, ticker=f"T{n}") for n in (2, 3)],
        {100 + n: insight(emotion) for n in (2, 3)},
    )

    behavior = get(find(log), EMOTIONAL, emotion.value)

    # sells carry no outcome
    assert figures(behavior) == [("decisions", 3, "count")]
    assert behavior.trade_ids == [3, 2, 1]


def test_fomo_compares_buys_naming_fomo_to_other_buys() -> None:
    log = decisions(
        insight(Emotion.FOMO, idea_source=IdeaSource.NEWS),
        insight(Emotion.FOMO, idea_source=IdeaSource.SOCIAL_MEDIA),
        insight(Emotion.FOMO, idea_source=IdeaSource.OWN_RESEARCH),
        outcome=90,
    ) + decisions(
        insight(),
        insight(Emotion.EXCITED),
        insight(idea_source=IdeaSource.NEWS),
        start=4,
        outcome=110,
    )

    fomo = get(find(log), BehaviorKind.FOMO)

    assert figures(fomo) == [
        *standard(3, -30, -10, -0.1, 0),
        *baseline(3, 30, 10, 0.1, 3),
        ("from_news_or_social", 2, "count"),
    ]


def test_fomo_short_waits_for_entries() -> None:
    found = find(decisions(insight(Emotion.FOMO)))

    assert need(found, BehaviorKind.FOMO) == Need(
        kind=BehaviorKind.FOMO, subject=None, more=2, reason="ENTRIES"
    )


def test_hype_needs_news_or_social_and_excitement_or_a_run_up() -> None:
    log = (
        decisions(insight(Emotion.EXCITED, idea_source=IdeaSource.NEWS), run_up=0.15)
        + decisions(insight(Emotion.FOMO, idea_source=IdeaSource.SOCIAL_MEDIA), start=2)
        + decisions(insight(idea_source=IdeaSource.NEWS), start=3, run_up=0.2)
        # news without excitement or a run-up, excitement without news
        + decisions(insight(idea_source=IdeaSource.NEWS), start=4, run_up=0.05)
        + decisions(insight(Emotion.EXCITED, idea_source=IdeaSource.OWN_RESEARCH), start=5)
        + decisions(insight(Emotion.EXCITED), start=6, run_up=0.3)
    )

    hype = get(find(log), BehaviorKind.HYPE)

    # the median covers members with a known run-up only
    assert figures(hype) == [
        ("decisions", 3, "count"),
        ("baseline_decisions", 3, "count"),
        ("median_run_up", 0.175, "fraction"),
    ]
    assert hype.trade_ids == [3, 2, 1]


def test_hype_short_with_unextracted_entries_waits_for_fields() -> None:
    log = decisions(insight(Emotion.EXCITED, idea_source=IdeaSource.NEWS), None)

    assert need(find(log), BehaviorKind.HYPE) == Need(
        kind=BehaviorKind.HYPE, subject=None, more=2, reason="FIELDS"
    )


def test_run_up_compares_buys_after_a_rise_to_calm_buys() -> None:
    log = (
        bare(1, run_up=0.1)
        + bare(2, run_up=0.2)
        + bare(3, run_up=0.3)
        + bare(4, run_up=0.0)
        + bare(5, run_up=0.05)
        + bare(6, run_up=0.09)
        + bare(7)
        + bare(8, side="SELL")
    )

    run_up = get(find(log), BehaviorKind.RUN_UP)

    assert figures(run_up) == [
        ("decisions", 3, "count"),
        ("baseline_decisions", 3, "count"),
        ("median_run_up", 0.2, "fraction"),
        ("baseline_median_run_up", 0.05, "fraction"),
    ]
    assert run_up.trade_ids == [3, 2, 1]


def test_run_up_short_waits_for_prices_while_one_is_pending() -> None:
    found = find(bare(1, 2, run_up=0.2) + bare(3, pending=True))

    assert need(found, BehaviorKind.RUN_UP) == Need(
        kind=BehaviorKind.RUN_UP, subject=None, more=1, reason="PRICES"
    )


def test_run_up_short_without_pending_prices_waits_for_trades() -> None:
    found = find(bare(1, 2, run_up=0.2) + bare(3))

    assert need(found, BehaviorKind.RUN_UP) == Need(
        kind=BehaviorKind.RUN_UP, subject=None, more=1, reason="TRADES"
    )


def test_overtrading_takes_burst_weeks_and_round_trips() -> None:
    week = 7 * DAY
    trades = [
        # four trades in the first week, one in each of the next three
        *(buy(n, T0 + n * HOUR, ticker=f"T{n}") for n in (1, 2, 3, 4)),
        *(buy(n, T0 + (n - 4) * week, ticker=f"T{n}") for n in (5, 6, 7)),
        # a round trip, under five days
        buy(8, T0 + 5 * week, ticker="RT"),
        sell(9, T0 + 5 * week + 3 * DAY, ticker="RT"),
        # exactly five days isn't a round trip
        buy(10, T0 + 6 * week, ticker="LONG"),
        sell(11, T0 + 6 * week + 5 * DAY, ticker="LONG"),
    ]
    periods = [
        period("RT", T0 + 5 * week, T0 + 5 * week + 3 * DAY, 5),
        period("LONG", T0 + 6 * week, T0 + 6 * week + 5 * DAY, 5),
    ]

    overtrading = get(find(Log(trades), periods=periods), BehaviorKind.OVERTRADING)

    # weeks hold 4, 1, 1, 1, 2 and 2 trades: the median is 1.5 and only the first is a burst
    assert figures(overtrading) == [
        ("decisions", 6, "count"),
        ("baseline_decisions", 5, "count"),
        ("burst_weeks", 1, "count"),
        ("round_trips", 1, "count"),
        ("median_trades_per_week", 1.5, "count"),
    ]
    assert overtrading.trade_ids == [9, 8, 4, 3, 2, 1]


def test_overtrading_short_waits_for_trades() -> None:
    found = find(bare(1, 2))

    assert need(found, BehaviorKind.OVERTRADING) == Need(
        kind=BehaviorKind.OVERTRADING, subject=None, more=3, reason="TRADES"
    )


def test_overconfidence_compares_high_conviction_buys_to_lower_ones() -> None:
    log = (
        decisions(insight(conviction=4), insight(conviction=5), insight(conviction=4), outcome=110)
        + decisions(
            insight(conviction=1), insight(conviction=2), insight(conviction=3), start=4, outcome=90
        )
        # no stated conviction: neither
        + decisions(insight(), start=7, outcome=500)
    )
    # looking back later on buys 1 and 2
    log += Log(
        entries=[entry(201, T0 + 10 * DAY, ticker="T1"), entry(202, T0 + 11 * DAY, ticker="T2")],
        insights={
            201: insight(outcome=ThesisOutcome.DID_NOT),
            202: insight(outcome=ThesisOutcome.PLAYED_OUT),
        },
    )

    overconfidence = get(find(log), BehaviorKind.OVERCONFIDENCE)

    assert figures(overconfidence) == [
        *standard(3, 30, 10, 0.1, 3),
        *baseline(3, -30, -10, -0.1, 0),
        ("right_for_wrong_reason", 1, "count"),
    ]
    assert overconfidence.entry_ids == [202, 201, 103, 102, 101]
    assert overconfidence.shown_entry_ids == [103, 102, 101, 202, 201]


def test_overconfidence_short_waits_for_entries() -> None:
    found = find(decisions(insight(conviction=5)))

    assert need(found, BehaviorKind.OVERCONFIDENCE) == Need(
        kind=BehaviorKind.OVERCONFIDENCE, subject=None, more=2, reason="ENTRIES"
    )


def test_a_look_back_belongs_to_the_most_recent_earlier_buy() -> None:
    trades = [
        buy(1, T0, outcome=110),
        buy(2, T0 + 20 * DAY, outcome=110),
        sell(3, T0 + 28 * DAY),
        buy(4, T0 + 29 * DAY, outcome=110, initial=True),
    ]
    entries = [
        entry(101, T0 - MINUTE),
        entry(102, T0 + 20 * DAY - MINUTE),
        entry(200, T0 + 25 * DAY),
        entry(201, T0 + 30 * DAY),
    ]
    insights = {
        101: insight(conviction=5),
        102: insight(conviction=5),
        200: insight(outcome=ThesisOutcome.PLAYED_OUT),
        201: insight(outcome=ThesisOutcome.DID_NOT),
    }
    log = Log(trades, entries, insights) + decisions(insight(conviction=5), start=5, outcome=110)

    overconfidence = get(find(log), BehaviorKind.OVERCONFIDENCE)

    # 201 is newer than 200 and both look back on buy 2, not buy 1, the sell or the initial buy
    assert figures(overconfidence)[-1] == ("right_for_wrong_reason", 1, "count")
    assert overconfidence.entry_ids == [201, 102, 105, 101]


def test_price_action_takes_the_newest_stated_reasoning_basis() -> None:
    log = (
        decisions(
            insight(basis=ReasoningBasis.PRICE_ACTION),
            insight(basis=ReasoningBasis.COMMENTARY),
            insight(basis=ReasoningBasis.PRICE_ACTION),
            outcome=110,
        )
        + decisions(
            insight(basis=ReasoningBasis.PRICE_ACTION),
            insight(basis=ReasoningBasis.FUNDAMENTALS),
            insight(basis=ReasoningBasis.FUNDAMENTALS),
            start=4,
            outcome=90,
        )
        # mixed or unstated: neither
        + decisions(insight(basis=ReasoningBasis.MIXED), insight(), start=7, outcome=500)
    )
    log += Log(
        # a newer entry on buy 3 that states no basis, and one on buy 4 that says fundamentals
        entries=[entry(203, T0 + 3 * HOUR, ticker="T3"), entry(204, T0 + 4 * HOUR, ticker="T4")],
        insights={203: insight(), 204: insight(basis=ReasoningBasis.FUNDAMENTALS)},
    )

    price_action = get(find(log), BehaviorKind.PRICE_ACTION)

    assert figures(price_action) == standard(3, 30, 10, 0.1, 3) + baseline(3, -30, -10, -0.1, 0)
    assert price_action.entry_ids == [203, 103, 102, 101]
    assert price_action.shown_entry_ids == [103, 102, 101, 203]


def test_price_action_short_waits_for_entries() -> None:
    found = find(decisions(insight(basis=ReasoningBasis.COMMENTARY)))

    assert need(found, BehaviorKind.PRICE_ACTION) == Need(
        kind=BehaviorKind.PRICE_ACTION, subject=None, more=2, reason="ENTRIES"
    )


def at_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 15, tzinfo=UTC)


def test_loss_aversion_compares_closed_losses_to_closed_gains() -> None:
    periods = [
        period("L1", T0, T0 + 2 * DAY, -10),
        period("L2", T0, T0 + 4 * DAY, -20),
        period("L3", T0, T0 + 6 * DAY, -30),
        period("G1", T0, T0 + 1 * DAY, 10),
        period("G2", T0, T0 + 3 * DAY, 5),
        period("G3", T0, T0 + 5 * DAY, 15),
        # open losses: older than the gains' median of three days, and exactly that old
        period("O1", at_day(AS_OF - 10 * DAY), None, -5),
        period("O2", at_day(AS_OF - 3 * DAY), None, -5),
        period("O3", at_day(AS_OF - 10 * DAY), None, 5),
    ]
    trades = [
        buy(1, T0, ticker="L1", initial=True),
        sell(2, T0 + 2 * DAY, ticker="L1"),
        buy(3, T0 + HOUR, ticker="L2"),
    ]

    loss_aversion = get(find(Log(trades), periods=periods), BehaviorKind.LOSS_AVERSION)

    assert figures(loss_aversion) == [
        *standard(3, -60, -20, -0.2, 0),
        *baseline(3, 30, 10, 0.1, 3),
        ("median_days", 4, "days"),
        ("baseline_median_days", 3, "days"),
        ("open_losses_held_longer", 1, "count"),
    ]
    # a period brings every trade inside it, the initial one too
    assert loss_aversion.trade_ids == [3, 2, 1]


def test_loss_aversion_compares_days_only_against_three_gains() -> None:
    periods = [
        period("L1", T0, T0 + 2 * DAY, -10),
        period("L2", T0, T0 + 4 * DAY, -20),
        period("L3", T0, T0 + 6 * DAY, -30),
        period("G1", T0, T0 + 1 * DAY, 10),
        period("G2", T0, T0 + 3 * DAY, 5),
    ]

    loss_aversion = get(find(Log(), periods=periods), BehaviorKind.LOSS_AVERSION)

    assert figures(loss_aversion) == [
        *standard(3, -60, -20, -0.2, 0),
        ("median_days", 4, "days"),
    ]


def test_loss_aversion_short_waits_for_trades() -> None:
    periods = [period("L1", T0, T0 + DAY, -10), period("L2", T0, None, -10)]

    assert need(find(Log(), periods=periods), BehaviorKind.LOSS_AVERSION) == Need(
        kind=BehaviorKind.LOSS_AVERSION, subject=None, more=2, reason="TRADES"
    )


def test_short_term_compares_buys_sold_within_their_horizon_to_ones_held_longer() -> None:
    plans: list[tuple[Horizon, int | None]] = [
        (Horizon.MONTHS, 30),
        (Horizon.MONTHS, 90),
        (Horizon.YEARS, 365),
        (Horizon.MONTHS, 91),
        (Horizon.YEARS, 366),
        (Horizon.MONTHS, None),
        # weeks have no planned length: neither
        (Horizon.WEEKS, 5),
    ]
    log = Log()
    periods: list[HoldingPeriod] = []
    for n, (horizon, kept) in enumerate(plans, 1):
        log += decisions(insight(horizon=horizon), start=n)
        at = T0 + n * HOUR
        periods.append(period(f"T{n}", at, None if kept is None else at + kept * DAY, 5))

    short_term = get(find(log, periods=periods), BehaviorKind.SHORT_TERM)

    assert figures(short_term) == [
        ("decisions", 3, "count"),
        ("baseline_decisions", 3, "count"),
        ("median_days", 90, "days"),
    ]
    assert short_term.trade_ids == [3, 2, 1]


def test_short_term_short_waits_for_fields_while_an_entry_is_pending() -> None:
    log = decisions(insight(horizon=Horizon.MONTHS), None)
    periods = [period("T1", T0 + HOUR, T0 + HOUR + DAY, 5), period("T2", T0, None, 5)]

    assert need(find(log, periods=periods), BehaviorKind.SHORT_TERM) == Need(
        kind=BehaviorKind.SHORT_TERM, subject=None, more=2, reason="FIELDS"
    )


def test_risk_concentration_takes_concentrated_holdings_without_an_exit() -> None:
    holdings = [
        holding("AAA", 0.3, loss=50),
        # tech sums to 0.45
        holding("BBB", 0.1, sector="Tech", loss=20),
        holding("CCC", 0.35, sector="Tech", loss=30),
        holding("DDD", 0.25, loss=10),
        holding("EEE", 0.05, sector="Energy", loss=5),
        holding("FFF", 0.1),
    ]
    periods = [
        period("AAA", T0, None, -10),
        period("BBB", T0, None, 20),
        period("CCC", T0, None, 40),
    ]
    log = Log(
        entries=[
            # an exit stated before the period's look-back window doesn't count
            entry(1, T0 - 7 * DAY - SECOND, ticker="AAA"),
            entry(2, T0 - 7 * DAY, ticker="AAA"),
            entry(3, T0, ticker="CCC"),
        ],
        insights={1: insight(exit=True), 2: insight(), 3: insight(exit=True)},
    )

    risk = get(find(log, periods=periods, holdings=holdings), BehaviorKind.RISK_CONCENTRATION)

    assert figures(risk) == [
        *standard(3, 10, 5, 0.05, 1),
        *baseline(3, 40, 40, 0.4, 1),
        ("possible_monthly_loss", 80, "money"),
        ("largest_weight", 0.3, "fraction"),
        ("sector_weight", 0.45, "fraction"),
    ]
    assert risk.entry_ids == [2]
    # the impact is the summed possible loss
    assert risk.score == 80 * 3 / 8


def test_risk_concentration_short_with_an_unknown_loss_waits_for_prices() -> None:
    holdings = [holding("AAA", 0.5), holding("BBB", 0.5, loss=10)]

    assert need(find(Log(), holdings=holdings), BehaviorKind.RISK_CONCENTRATION) == Need(
        kind=BehaviorKind.RISK_CONCENTRATION, subject=None, more=2, reason="PRICES"
    )


def test_risk_concentration_short_with_an_unextracted_entry_waits_for_fields() -> None:
    holdings = [holding("AAA", 0.5, loss=10), holding("BBB", 0.5, loss=10)]
    log = Log(entries=[entry(1, T0, ticker="AAA")])

    assert need(find(log, holdings=holdings), BehaviorKind.RISK_CONCENTRATION) == Need(
        kind=BehaviorKind.RISK_CONCENTRATION, subject=None, more=2, reason="FIELDS"
    )


def sized() -> Log:
    # the top quarter of nine buys is three: 0.5 and the earlier two of the three 0.4s
    return (
        decisions(insight(conviction=2), weight=0.5)
        + decisions(insight(conviction=5), start=2, weight=0.4)
        + decisions(insight(), start=3, weight=0.4)
        + bare(4, weight=0.4)
        + bare(5, 6, 7, 8, 9, weight=0.1)
        + bare(10)
    )


def test_position_sizing_takes_the_largest_quarter_of_buys() -> None:
    periods = [
        period("T1", T0, None, 5),
        period("T2", T0, None, 5),
        period("T3", T0, T0 + DAY, 5),
    ]
    holdings = [holding("T1", 0.1, loss=40), holding("T2", 0.1, loss=60)]

    sizing = get(find(sized(), periods=periods, holdings=holdings), BehaviorKind.POSITION_SIZING)

    assert figures(sizing) == [
        ("decisions", 3, "count"),
        ("baseline_decisions", 6, "count"),
        ("avg_weight", 0.4333, "fraction"),
        ("baseline_avg_weight", 0.15, "fraction"),
        ("large_low_conviction", 2, "count"),
        ("possible_monthly_loss", 100, "money"),
    ]
    assert sizing.trade_ids == [3, 2, 1]


def test_position_sizing_omits_the_possible_loss_when_one_is_unknown() -> None:
    periods = [period("T1", T0, None, 5), period("T2", T0, None, 5)]
    holdings = [holding("T1", 0.1, loss=40), holding("T2", 0.1)]

    sizing = get(find(sized(), periods=periods, holdings=holdings), BehaviorKind.POSITION_SIZING)

    assert "possible_monthly_loss" not in keys(sizing)


# an undocumented buy states no conviction; one still being extracted isn't counted yet
def test_position_sizing_counts_undocumented_buys_as_low_conviction() -> None:
    log = bare(1, 2, 3, weight=0.5) + bare(4, 5, 6, 7, 8, 9, weight=0.1)

    sizing = get(find(log), BehaviorKind.POSITION_SIZING)

    assert ("large_low_conviction", 3, "count") in figures(sizing)


def test_position_sizing_skips_large_buys_still_being_extracted() -> None:
    log = decisions(None, None, None, weight=0.5) + bare(4, 5, 6, 7, 8, 9, weight=0.1)

    sizing = get(find(log), BehaviorKind.POSITION_SIZING)

    assert ("large_low_conviction", 0, "count") in figures(sizing)


def test_position_sizing_short_waits_for_trades() -> None:
    found = find(bare(1, 2, 3, 4, weight=0.1))

    assert need(found, BehaviorKind.POSITION_SIZING) == Need(
        kind=BehaviorKind.POSITION_SIZING, subject=None, more=2, reason="TRADES"
    )


# --- shared rules ---


def test_unknown_outcomes_count_as_decisions_but_not_money() -> None:
    log = bare(1, outcome=110) + bare(2) + bare(3, side="SELL")

    undocumented = get(find(log), UNDOCUMENTED)

    assert figures(undocumented) == standard(3, 10, 10, 0.1, 1)


def test_all_unknown_outcomes_leave_only_the_count() -> None:
    log = bare(1, 2) + bare(3, side="SELL")

    assert figures(get(find(log), UNDOCUMENTED)) == [("decisions", 3, "count")]


def test_baseline_figures_need_three_baseline_decisions() -> None:
    log = decisions(insight(), insight(), insight(), outcome=110) + decisions(
        *PLANNED[:2], start=4, outcome=90
    )

    assert figures(get(find(log), NO_PLAN)) == standard(3, 30, 10, 0.1, 3)


def test_run_up_baseline_median_needs_three_calm_buys() -> None:
    log = bare(1, 2, 3, run_up=0.2) + bare(4, 5, run_up=0.0)

    assert keys(get(find(log), BehaviorKind.RUN_UP)) == ["decisions", "median_run_up"]


@pytest.mark.parametrize("unwritten", ["auto", "note", "blank"])
def test_auto_note_and_blank_entries_dont_document_a_trade(unwritten: str) -> None:
    log = bare(1, 2, 3)
    log.entries = [
        entry(100 + n, T0 + n * HOUR, ticker=f"T{n}", body="  \n" if unwritten == "blank" else "x")
        for n in (1, 2, 3)
    ]
    log.insights = {e.id: insight() for e in log.entries}
    auto = [101, 102, 103] if unwritten == "auto" else []
    notes = [note(i) for i in (101, 102, 103)] if unwritten == "note" else []

    found = find(log, auto=auto, notes=notes)

    assert figures(get(found, UNDOCUMENTED)) == [("decisions", 3, "count")]
    assert get(found, UNDOCUMENTED).entry_ids == []
    assert need(found, NO_PLAN) == Need(kind=NO_PLAN, subject=None, more=3, reason="ENTRIES")


@pytest.mark.parametrize(
    ("at", "linked"),
    [
        (T0 - 7 * DAY, True),
        (T0 - 7 * DAY - SECOND, False),
        # the last second of the new york day after the trade
        (datetime(2026, 3, 3, 23, 59, 59, tzinfo=NY), True),
        (datetime(2026, 3, 4, tzinfo=NY), False),
    ],
)
def test_linked_entries_run_from_a_week_before_to_the_end_of_the_next_day(
    at: datetime, linked: bool
) -> None:
    log = Log([buy(1, T0)], [entry(101, at)], {101: insight()})

    found = find(log)

    assert need(found, UNDOCUMENTED) == Need(
        kind=UNDOCUMENTED, subject=None, more=3 if linked else 2, reason="TRADES"
    )


@pytest.mark.parametrize(
    ("ticker", "named", "linked"),
    [(None, ["NVDA"], True), (None, ["AAPL"], False), ("AAPL", ["NVDA"], False)],
)
def test_an_entry_without_a_ticker_links_through_its_insight(
    ticker: str | None, named: list[str], linked: bool
) -> None:
    log = Log([buy(1, T0)], [entry(101, T0, ticker=ticker)], {101: insight(tickers=named)})

    found = find(log)

    assert need(found, UNDOCUMENTED) == Need(
        kind=UNDOCUMENTED, subject=None, more=3 if linked else 2, reason="TRADES"
    )


def test_a_trade_with_an_unextracted_entry_is_in_neither_group() -> None:
    log = (
        decisions(insight(), insight(), insight())
        + decisions(*PLANNED, start=4)
        + decisions(None, start=7)
        # one of its two entries is extracted and states a plan
        + decisions(insight(thesis=True), start=8)
        + Log(entries=[entry(208, T0 + 8 * HOUR, ticker="T8")])
    )

    found = find(log)

    assert figures(get(found, NO_PLAN)) == [
        ("decisions", 3, "count"),
        ("baseline_decisions", 3, "count"),
    ]
    # documented, just not read yet
    assert need(found, UNDOCUMENTED) == Need(
        kind=UNDOCUMENTED, subject=None, more=3, reason="TRADES"
    )


def test_a_short_group_with_an_unextracted_entry_waits_for_fields() -> None:
    found = find(decisions(insight(), insight(), None))

    assert need(found, NO_PLAN) == Need(kind=NO_PLAN, subject=None, more=1, reason="FIELDS")


def test_entries_up_to_is_the_newest_extracted_entry() -> None:
    log = Log(entries=[entry(1, T0), entry(2, T0 + DAY), entry(3, T0 + 2 * DAY)])
    log.insights = {1: insight(), 2: insight()}

    assert find(log).entries_up_to == T0 + DAY


def test_entries_from_drops_trades_whose_window_wasnt_read() -> None:
    log = bare(1, 2, 3)

    found = find(log, entries_from=T0 - 7 * DAY + 2 * HOUR)

    assert need(found, UNDOCUMENTED) == Need(
        kind=UNDOCUMENTED, subject=None, more=1, reason="TRADES"
    )


def test_shown_entries_put_the_ones_that_placed_a_decision_first() -> None:
    log = decisions(*[insight(Emotion.ANXIOUS)] * 3)
    log += Log(
        entries=[entry(200 + n, T0 + n * HOUR, ticker=f"T{n}") for n in (1, 2, 3)],
        insights={200 + n: insight(Emotion.CALM) for n in (1, 2, 3)},
    )

    anxious = get(find(log), EMOTIONAL, "anxious")

    assert anxious.entry_ids == [203, 103, 202, 102, 201, 101]
    assert anxious.shown_entry_ids == [103, 102, 101, 203, 202]


# --- ranking ---


def test_score_is_impact_times_evidence() -> None:
    def score(log: Log) -> float:
        return get(find(log), UNDOCUMENTED).score

    # no baseline: |pnl|, weighed by members out of eight
    assert score(bare(1, 2, 3, 4, outcome=110)) == 40 * 4 / 8
    assert score(bare(*range(1, 9), outcome=105)) == 40
    assert score(bare(*range(1, 11), outcome=104)) == 40
    # with a baseline: the gap in average pnl times members
    with_baseline = bare(1, 2, 3, outcome=110) + decisions(*[insight()] * 3, start=4, outcome=90)
    assert score(with_baseline) == 20 * 3 * 3 / 8


def test_behaviors_rank_by_score() -> None:
    # undocumented: (10 - -10) * 3 members; no_plan: |-30|
    log = bare(1, 2, 3, outcome=110) + decisions(*[insight()] * 3, start=4, outcome=90)

    found = find(log)

    assert identities(found) == [(UNDOCUMENTED, None), (NO_PLAN, None)]
    assert [b.score for b in found.behaviors] == [22.5, 11.25]


def close_call() -> Log:
    # no_plan scores 30 * 3/8 = 11.25; undocumented (21 - 10) * 3 * 3/8 = 12.375
    return decisions(*[insight()] * 3, outcome=110) + bare(4, 5, 6, outcome=121)


def test_close_call_ranks_undocumented_first() -> None:
    assert identities(find(close_call())) == [(UNDOCUMENTED, None), (NO_PLAN, None)]


def test_fits_raises_a_behavior() -> None:
    found = find(close_call(), feedback=[feedback("no_plan", "FITS")])

    assert identities(found) == [(NO_PLAN, None), (UNDOCUMENTED, None)]
    assert get(found, NO_PLAN).score == 11.25 * 1.25


def test_knew_lowers_a_behavior() -> None:
    found = find(close_call(), feedback=[feedback("undocumented", "KNEW")])

    assert identities(found) == [(NO_PLAN, None), (UNDOCUMENTED, None)]
    assert get(found, UNDOCUMENTED).score == 12.375 * 0.75


def no_plan_in(day: date, trade_ids: list[int]) -> PastReport:
    return past(day, PastBehavior("no_plan", None, [], trade_ids))


@pytest.mark.parametrize(
    ("history", "shown"),
    [
        ([no_plan_in(REPORT_DAY, [1, 2, 3])], False),
        ([no_plan_in(REPORT_DAY, [1, 2])], True),
        # the report the reaction was given on is missing, or doesn't hold the behavior
        ([], False),
        ([no_plan_in(REPORT_DAY - DAY, [1, 2])], False),
        ([past(REPORT_DAY, PastBehavior("undocumented", None, [], [1, 2]))], False),
    ],
)
def test_doesnt_fit_hides_a_behavior_until_its_trades_change(
    history: list[PastReport], shown: bool
) -> None:
    log = decisions(*[insight()] * 3, outcome=110)

    found = find(log, feedback=[feedback("no_plan", "DOESNT_FIT")], history=history)

    assert ((NO_PLAN, None) in identities(found)) == shown
    # hidden isn't short
    assert need(found, NO_PLAN) is None


def test_the_newest_feedback_wins() -> None:
    log = decisions(*[insight()] * 3, outcome=110)
    fits = feedback("no_plan", "FITS", at=T0 + 2 * DAY)
    hides = feedback("no_plan", "DOESNT_FIT", at=T0 + DAY)

    assert get(find(log, feedback=[hides, fits]), NO_PLAN).score == 11.25 * 1.25
    later = feedback("no_plan", "DOESNT_FIT", at=T0 + 3 * DAY)
    assert identities(find(log, feedback=[later, fits])) == []


def test_at_most_two_per_kind() -> None:
    log = (
        decisions(*[insight(Emotion.ANXIOUS)] * 3, outcome=110)
        + decisions(*[insight(Emotion.EXCITED)] * 3, start=4, outcome=120)
        + decisions(*[insight(Emotion.REGRET)] * 3, start=7, outcome=105)
    )

    found = find(log)

    assert [b.subject for b in found.behaviors if b.kind == EMOTIONAL] == ["excited", "anxious"]


def test_ties_keep_catalog_order_and_at_most_five_are_kept() -> None:
    # no outcomes, so every score is zero
    log = (
        decisions(*[insight(Emotion.ANXIOUS, conviction=5)] * 3)
        + decisions(*[insight(Emotion.EXCITED, conviction=5)] * 3, start=4)
        + decisions(*[insight(Emotion.REGRET, conviction=5)] * 3, start=7)
        + bare(10, 11, 12, run_up=0.2)
    )

    found = find(log)

    # regret is past the kind cap and overconfidence past the overall cap
    assert identities(found) == [
        (NO_PLAN, None),
        (UNDOCUMENTED, None),
        (EMOTIONAL, "anxious"),
        (EMOTIONAL, "excited"),
        (BehaviorKind.RUN_UP, None),
    ]


# --- history and notes ---


def test_previous_is_the_newest_report_holding_the_behavior() -> None:
    log = decisions(*[insight()] * 3) + bare(4, 5, 6)
    older = [Figure(key="decisions", value=4, unit="count")]
    newer = [Figure(key="decisions", value=5, unit="count")]
    history = [
        no_plan_in(date(2026, 10, 1), [1]),
        past(date(2026, 10, 20), PastBehavior("undocumented", None, [], [4])),
        past(date(2026, 10, 10), PastBehavior("no_plan", None, newer, [1])),
        past(date(2026, 9, 1), PastBehavior("no_plan", None, older, [1])),
    ]

    found = find(log, history=history)

    assert get(found, NO_PLAN).previous == Previous(day=date(2026, 10, 10), figures=newer)
    assert get(found, UNDOCUMENTED).previous == Previous(day=date(2026, 10, 20), figures=[])


def test_previous_is_none_without_the_behavior_in_history() -> None:
    log = decisions(*[insight()] * 3)
    history = [past(date(2026, 10, 1), PastBehavior("undocumented", None, [], [1]))]

    assert get(find(log, history=history), NO_PLAN).previous is None


def test_notes_hold_the_feedback_note_and_the_newest_written_notes() -> None:
    log = decisions(*[insight()] * 3)
    notes = [note(301), note(304), note(302), note(303), note(305, "emotional", "anxious")]

    with_note = get(
        find(log, feedback=[feedback("no_plan", "FITS", note="I rush.")], notes=notes), NO_PLAN
    )
    without = get(find(log, notes=notes), NO_PLAN)

    assert with_note.feedback_notes == ["I rush."]
    assert with_note.note_entry_ids == [304, 303]
    assert without.feedback_notes == []
    assert without.note_entry_ids == [304, 303, 302]
