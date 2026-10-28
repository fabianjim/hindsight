from datetime import UTC, date, datetime, timedelta
from typing import Literal

import pytest

from hindsight.citations import Citation
from hindsight.enrichment import Claim, EntryInput, EntryInsight
from hindsight.pattern_reports import Need
from hindsight.patterns import (
    FELT_WINDOW,
    MIN_SUPPORT,
    SILENT_WINDOW,
    Candidate,
    Candidates,
    HoldingPeriod,
    LedgerTrade,
    Outcome,
    PatternReportRequest,
    find_patterns,
)
from hindsight.vocab import Emotion, EntryType, Horizon, IdeaSource, PatternKind

# a monday, 10am in new york
T0 = datetime(2026, 3, 2, 15, tzinfo=UTC)
DAY = timedelta(days=1)
SECOND = timedelta(seconds=1)


def entry(
    entry_id: int,
    at: datetime = T0,
    *,
    entry_type: EntryType = EntryType.INSIGHT,
    ticker: str | None = "NVDA",
    body: str = "Some words.",
    snapshot: float | None = None,
) -> EntryInput:
    return EntryInput(
        id=entry_id,
        entry_type=entry_type,
        body=body,
        ticker=ticker,
        timestamp=at,
        price_snapshot=snapshot,
    )


def insight(
    *emotions: Emotion,
    conviction: Literal[1, 2, 3, 4, 5] | None = None,
    horizon: Horizon | None = None,
    idea_source: IdeaSource | None = None,
    thesis: bool = False,
) -> EntryInsight:
    claim = Claim(summary="A thesis", citations=[Citation(entry_id=1, quote="words")])
    return EntryInsight(
        thesis=claim if thesis else None,
        exit_condition=None,
        horizon=horizon,
        conviction=conviction,
        emotions=list(emotions),
        idea_source=idea_source,
        tickers=[],
        themes=[],
    )


def trade(
    trade_id: int,
    at: datetime,
    side: Literal["BUY", "SELL"] = "BUY",
    *,
    ticker: str = "NVDA",
    initial: bool = False,
) -> LedgerTrade:
    return LedgerTrade(
        trade_id=trade_id,
        ticker=ticker,
        side=side,
        shares=1,
        price=100,
        timestamp=at,
        initial=initial,
    )


def period(
    opened: datetime, closed: datetime | None, pnl: float | None, ticker: str = "NVDA"
) -> HoldingPeriod:
    return HoldingPeriod(ticker=ticker, opened_at=opened, closed_at=closed, pnl=pnl)


def find(
    entries: list[EntryInput],
    insights: dict[int, EntryInsight] | None = None,
    *,
    outcomes: list[Outcome] | None = None,
    trades: list[LedgerTrade] | None = None,
    periods: list[HoldingPeriod] | None = None,
    auto: list[int] | None = None,
) -> Candidates:
    facts = PatternReportRequest(
        as_of=date(2026, 10, 27),
        outcomes=outcomes or [],
        trades=trades or [],
        holding_periods=periods or [],
        auto_entry_ids=auto or [],
    )
    return find_patterns(facts, entries, insights or {})


def of_kind(found: Candidates, kind: PatternKind) -> list[Candidate]:
    return [c for c in found.patterns if c.kind == kind]


def figures(candidate: Candidate) -> dict[str, float]:
    return {f.key: f.value for f in candidate.figures}


def more(found: Candidates, kind: PatternKind) -> int | None:
    return next((n.more for n in found.needs if n.kind == kind), None)


def anxious_entries(count: int, ticker: str | None = "NVDA") -> list[EntryInput]:
    return [entry(i, T0 + i * 10 * DAY, ticker=ticker) for i in range(1, count + 1)]


def all_anxious(entries: list[EntryInput]) -> dict[int, EntryInsight]:
    return {e.id: insight(Emotion.ANXIOUS) for e in entries}


def test_a_user_with_nothing_needs_the_minimum_everywhere() -> None:
    found = find([])

    assert found.patterns == []
    # gains_vs_losses needs the minimum on each side
    assert found.needs == [
        Need(
            kind=kind, more=2 * MIN_SUPPORT if kind == PatternKind.GAINS_VS_LOSSES else MIN_SUPPORT
        )
        for kind in PatternKind
    ]
    assert found.entries_up_to is None


# felt_vs_did


def test_felt_vs_did_counts_entries_followed_by_trades() -> None:
    entries = anxious_entries(4)
    trades = [
        trade(1, entries[0].timestamp + DAY),
        trade(2, entries[0].timestamp + 2 * DAY),
        trade(3, entries[1].timestamp + DAY, "SELL"),
        trade(4, entries[2].timestamp + DAY, ticker="AAPL"),
    ]

    [candidate] = of_kind(
        find(entries, all_anxious(entries), trades=trades), PatternKind.FELT_VS_DID
    )

    assert candidate.subject == "anxious"
    assert candidate.support == 4
    assert figures(candidate) == {"entries": 4, "followed_by_buy": 1, "followed_by_sell": 1}
    assert candidate.entry_ids == [4, 3, 2, 1]


@pytest.mark.parametrize(
    ("after", "followed"),
    [(timedelta(0), False), (SECOND, True), (FELT_WINDOW, True), (FELT_WINDOW + SECOND, False)],
)
def test_felt_window_is_strictly_after_and_at_most_seven_days(
    after: timedelta, followed: bool
) -> None:
    entries = anxious_entries(3)
    trades = [trade(1, entries[0].timestamp + after)]

    found = find(entries, all_anxious(entries), trades=trades)

    assert bool(of_kind(found, PatternKind.FELT_VS_DID)) == followed


def test_an_entry_without_a_ticker_is_followed_by_any_trade() -> None:
    entries = anxious_entries(3, ticker=None)
    trades = [trade(1, entries[0].timestamp + DAY, ticker="AAPL")]

    [candidate] = of_kind(
        find(entries, all_anxious(entries), trades=trades), PatternKind.FELT_VS_DID
    )

    assert figures(candidate)["followed_by_buy"] == 1


def test_initial_trades_never_follow_a_feeling() -> None:
    entries = anxious_entries(3)
    trades = [trade(1, entries[0].timestamp + DAY, initial=True)]

    found = find(entries, all_anxious(entries), trades=trades)

    assert of_kind(found, PatternKind.FELT_VS_DID) == []
    assert more(found, PatternKind.FELT_VS_DID) == 1


# shared rules


def test_auto_entries_count_as_unwritten() -> None:
    entries = anxious_entries(3)
    # each trade lands on an auto entry, which neither precedes it as a feeling nor explains it
    trades = [trade(i, e.timestamp + SECOND) for i, e in enumerate(entries)]

    found = find(entries, all_anxious(entries), trades=trades, auto=[1, 2, 3])

    assert more(found, PatternKind.FELT_VS_DID) == MIN_SUPPORT
    [silent] = of_kind(found, PatternKind.SILENT_TRADES)
    assert figures(silent) == {"trades": 3, "without_entry": 3}


def test_blank_entries_count_as_unwritten() -> None:
    entries = [entry(i, T0 + i * DAY, body="  ") for i in range(1, 4)]

    found = find(entries, all_anxious(entries), trades=[trade(1, T0 + DAY + SECOND)])

    assert more(found, PatternKind.FELT_VS_DID) == MIN_SUPPORT


def test_entries_without_insights_still_count_for_ledger_kinds() -> None:
    entries = [entry(1, T0), entry(2, T0 + 10 * DAY), entry(3, T0 + 20 * DAY)]
    trades = [trade(i, e.timestamp + DAY) for i, e in enumerate(entries)]
    trades.append(trade(9, T0 + 40 * DAY))

    found = find(entries, {}, trades=trades)

    assert more(found, PatternKind.FELT_VS_DID) == MIN_SUPPORT
    [silent] = of_kind(found, PatternKind.SILENT_TRADES)
    assert figures(silent) == {"trades": 4, "without_entry": 1}


def test_ids_that_arent_the_users_entries_are_ignored() -> None:
    entries = [entry(i, T0 + i * DAY, snapshot=100) for i in range(1, 4)]
    outcomes = [Outcome(entry_id=99, price=200, kind="LATEST", on=date(2026, 10, 27))]

    found = find(entries, all_anxious(entries), outcomes=outcomes, auto=[99])

    assert of_kind(found, PatternKind.OUTCOMES_BY_STATE) == []
    assert more(found, PatternKind.OUTCOMES_BY_STATE) == MIN_SUPPORT


# outcomes_by_state


def latest(entry_id: int, price: float) -> Outcome:
    return Outcome(entry_id=entry_id, price=price, kind="LATEST", on=date(2026, 10, 27))


def test_outcomes_by_state_groups_by_emotion_conviction_and_idea_source() -> None:
    entries = [entry(i, T0 + i * DAY, snapshot=100) for i in range(1, 6)]
    entries.append(entry(6, T0 + 6 * DAY))  # no snapshot
    insights = {
        1: insight(Emotion.FOMO, conviction=5, idea_source=IdeaSource.SOCIAL_MEDIA),
        2: insight(Emotion.FOMO, conviction=4, idea_source=IdeaSource.SOCIAL_MEDIA),
        3: insight(Emotion.FOMO, conviction=3, idea_source=IdeaSource.SOCIAL_MEDIA),
        4: insight(conviction=4),
        5: insight(Emotion.FOMO, conviction=1),
        6: insight(Emotion.FOMO, conviction=5),
    }
    prices = {1: 90, 2: 80, 3: 120, 4: 70, 5: 100, 6: 50}
    outcomes = [latest(i, p) for i, p in prices.items()]

    found = find(entries, insights, outcomes=outcomes)

    by_subject = {c.subject: c for c in of_kind(found, PatternKind.OUTCOMES_BY_STATE)}
    # idea_source:social_media qualifies too, but ties with conviction:high and the kind cap is 2
    assert list(by_subject) == ["emotion:fomo", "conviction:high"]
    assert figures(by_subject["emotion:fomo"]) == {
        "entries": 4,
        "above_entry_price": 1,
        "below_entry_price": 2,
    }
    assert figures(by_subject["conviction:high"]) == {
        "entries": 3,
        "above_entry_price": 0,
        "below_entry_price": 3,
    }
    assert by_subject["conviction:high"].entry_ids == [4, 2, 1]


def test_an_even_split_is_not_a_pattern() -> None:
    entries = [entry(i, T0 + i * DAY, snapshot=100) for i in range(1, 5)]
    insights = {e.id: insight(Emotion.CALM) for e in entries}
    outcomes = [latest(1, 110), latest(2, 120), latest(3, 90), latest(4, 80)]

    found = find(entries, insights, outcomes=outcomes)

    assert of_kind(found, PatternKind.OUTCOMES_BY_STATE) == []
    assert more(found, PatternKind.OUTCOMES_BY_STATE) == 1


# horizon_vs_holding


def test_horizon_vs_holding_counts_buys_held_shorter_and_still_open() -> None:
    buys = [entry(i, T0 + i * DAY, entry_type=EntryType.BUY) for i in range(1, 5)]
    buys.append(entry(5, T0 + 5 * DAY, entry_type=EntryType.BUY, ticker="AAPL"))  # no period
    insights = {e.id: insight(horizon=Horizon.YEARS) for e in buys}
    periods = [
        period(T0, T0 + 3.5 * DAY, 10),  # holds buys 1 to 3, sold within days
        period(T0 + 4 * DAY, None, 5),  # buy 4, still open
    ]

    [candidate] = of_kind(find(buys, insights, periods=periods), PatternKind.HORIZON_VS_HOLDING)

    assert candidate.subject == "years"
    assert figures(candidate) == {"buys": 4, "held_shorter": 3, "still_open": 1}


@pytest.mark.parametrize(("held", "shorter"), [(14, True), (15, False)])
def test_weeks_are_held_shorter_at_fourteen_days_or_fewer(held: int, shorter: bool) -> None:
    buys = [entry(i, T0, entry_type=EntryType.BUY) for i in range(1, 4)]
    insights = {e.id: insight(horizon=Horizon.WEEKS) for e in buys}
    periods = [period(T0, T0 + held * DAY, 10)]

    found = find(buys, insights, periods=periods)

    assert bool(of_kind(found, PatternKind.HORIZON_VS_HOLDING)) == shorter


def test_nothing_is_held_shorter_than_days() -> None:
    buys = [entry(i, T0, entry_type=EntryType.BUY) for i in range(1, 4)]
    insights = {e.id: insight(horizon=Horizon.DAYS) for e in buys}

    found = find(buys, insights, periods=[period(T0, T0 + DAY, 10)])

    assert of_kind(found, PatternKind.HORIZON_VS_HOLDING) == []
    assert more(found, PatternKind.HORIZON_VS_HOLDING) == 1


def test_only_buy_entries_have_a_holding_horizon() -> None:
    entries = [entry(i, T0) for i in range(1, 4)]
    insights = {e.id: insight(horizon=Horizon.YEARS) for e in entries}

    found = find(entries, insights, periods=[period(T0, T0 + DAY, 10)])

    assert more(found, PatternKind.HORIZON_VS_HOLDING) == MIN_SUPPORT


# gains_vs_losses


def closed(start: int, held: int, pnl: float, ticker: str = "NVDA") -> HoldingPeriod:
    return period(T0 + start * DAY, T0 + (start + held) * DAY, pnl, ticker)


def test_gains_vs_losses_compares_median_days_and_written_entries() -> None:
    periods = [
        closed(0, 2, 10),
        closed(10, 3, 10),
        closed(20, 4, 10, "AAPL"),
        closed(30, 10, -5),
        closed(50, 20, -5),
        closed(80, 30, -5),
        closed(120, 40, 0),  # neither side
        period(T0, None, -5, "MSFT"),  # still open
    ]
    entries = [entry(1, T0 + DAY), entry(2, T0 + 21 * DAY), entry(3, T0 + 31 * DAY)]

    [candidate] = of_kind(find(entries, periods=periods), PatternKind.GAINS_VS_LOSSES)

    assert candidate.subject is None
    assert candidate.support == 6
    assert figures(candidate) == {
        "gains": 3,
        "losses": 3,
        "gains_median_days": 3,
        "losses_median_days": 20,
        "gains_with_entry": 1,
        "losses_with_entry": 1,
    }
    assert candidate.entry_ids == [3, 1]


@pytest.mark.parametrize(
    ("gains", "losses", "needed"), [(2, 1, 3), (3, 2, 1), (0, 5, 3), (0, 0, 6)]
)
def test_gains_vs_losses_needs_the_minimum_on_both_sides(
    gains: int, losses: int, needed: int
) -> None:
    periods = [closed(i * 10, 2, 10) for i in range(gains)]
    periods += [closed(100 + i * 10, 20, -10) for i in range(losses)]

    found = find([], periods=periods)

    assert of_kind(found, PatternKind.GAINS_VS_LOSSES) == []
    assert more(found, PatternKind.GAINS_VS_LOSSES) == needed


def test_equal_medians_are_not_a_pattern() -> None:
    periods = [closed(i * 10, 5, 10 if i < 3 else -10) for i in range(6)]

    found = find([], periods=periods)

    assert more(found, PatternKind.GAINS_VS_LOSSES) == 1


def test_days_held_are_whole_new_york_days() -> None:
    # 3pm new york to 5am the next day is one day; a fractional median is kept
    evening = datetime(2026, 3, 2, 20, tzinfo=UTC)
    periods = [period(evening, evening + timedelta(hours=14), 10) for _ in range(2)]
    periods += [period(evening, evening + 2 * DAY, 10) for _ in range(2)]
    periods += [closed(100 + i * 10, 20, -10) for i in range(3)]

    [candidate] = of_kind(find([], periods=periods), PatternKind.GAINS_VS_LOSSES)

    assert figures(candidate)["gains_median_days"] == 1.5


# silent_trades


@pytest.mark.parametrize(
    ("offset", "silent"),
    [(-SILENT_WINDOW, False), (SILENT_WINDOW, False), (SILENT_WINDOW + SECOND, True)],
)
def test_a_trade_is_silent_without_an_entry_within_a_day(offset: timedelta, silent: bool) -> None:
    trades = [trade(1, T0), trade(2, T0 + 10 * DAY), trade(3, T0 + 20 * DAY)]
    entries = [entry(1, T0 + offset), entry(2, T0 + 10 * DAY), entry(3, T0 + 20 * DAY)]

    found = find(entries, trades=trades)

    assert bool(of_kind(found, PatternKind.SILENT_TRADES)) == silent


def test_silent_trades_ignore_initial_trades_and_other_tickers() -> None:
    trades = [trade(i, T0 + i * DAY, initial=True) for i in range(1, 5)]
    trades += [trade(i, T0 + i * 10 * DAY) for i in range(5, 8)]
    entries = [entry(1, T0 + 50 * DAY, ticker="AAPL")]

    [candidate] = of_kind(find(entries, trades=trades), PatternKind.SILENT_TRADES)

    assert figures(candidate) == {"trades": 3, "without_entry": 3}
    assert candidate.entry_ids == []


# open_theses


def test_open_theses_counts_buys_without_a_later_reflection() -> None:
    buys = [entry(i, T0 + i * 10 * DAY, entry_type=EntryType.BUY) for i in range(1, 4)]
    buys.append(entry(4, T0, entry_type=EntryType.BUY, ticker="AAPL"))
    buys.append(entry(5, T0, entry_type=EntryType.BUY, ticker=None))
    reflections = [
        entry(10, T0 + 15 * DAY, entry_type=EntryType.REFLECTION),  # revisits buy 1 only
        entry(11, T0 + 50 * DAY, entry_type=EntryType.REFLECTION, ticker="MSFT"),
    ]
    insights = {e.id: insight(thesis=True) for e in buys}

    [candidate] = of_kind(find(buys + reflections, insights), PatternKind.OPEN_THESES)

    assert figures(candidate) == {"buys_with_thesis": 4, "not_revisited": 3}
    assert candidate.entry_ids == [3, 2, 1, 4]


def test_a_reflection_vestry_wrote_doesnt_revisit() -> None:
    buys = [entry(i, T0, entry_type=EntryType.BUY) for i in range(1, 4)]
    reflection = entry(10, T0 + DAY, entry_type=EntryType.REFLECTION)
    insights = {e.id: insight(thesis=True) for e in buys}

    found = find([*buys, reflection], insights, auto=[10])

    [candidate] = of_kind(found, PatternKind.OPEN_THESES)
    assert figures(candidate)["not_revisited"] == 3


# ranking and needs


def test_ranks_by_support_then_kind_order_with_a_per_kind_cap() -> None:
    # four emotions, each with entries followed by a buy, plus three silent trades
    entries: list[EntryInput] = []
    insights: dict[int, EntryInsight] = {}
    trades: list[LedgerTrade] = []
    sizes = {Emotion.CALM: 4, Emotion.ANXIOUS: 5, Emotion.FOMO: 4, Emotion.REGRET: 3}
    for emotion, size in sizes.items():
        for _ in range(size):
            entries.append(entry(len(entries) + 1, T0 + len(entries) * 10 * DAY))
            insights[len(entries)] = insight(emotion)
        trades.append(trade(len(trades) + 1, entries[-1].timestamp + DAY))
    trades += [trade(100 + i, T0 + (500 + i * 10) * DAY) for i in range(5)]

    found = find(entries, insights, trades=trades)

    assert [(c.kind, c.subject, c.support) for c in found.patterns] == [
        (PatternKind.SILENT_TRADES, None, 9),
        (PatternKind.FELT_VS_DID, "anxious", 5),
        # calm and fomo tie; calm comes first in the emotion enum
        (PatternKind.FELT_VS_DID, "calm", 4),
    ]


def test_keeps_at_most_five_patterns() -> None:
    entries: list[EntryInput] = []
    insights: dict[int, EntryInsight] = {}
    outcomes: list[Outcome] = []
    for i, emotion in enumerate(Emotion):
        for _ in range(3):
            at = T0 + len(entries) * 10 * DAY
            entries.append(entry(len(entries) + 1, at, entry_type=EntryType.BUY, snapshot=100))
            insights[len(entries)] = insight(
                emotion, idea_source=list(IdeaSource)[i % 5], thesis=True
            )
            outcomes.append(latest(len(entries), 90))
    trades = [trade(i, e.timestamp + DAY) for i, e in enumerate(entries)]
    trades += [trade(100 + i, T0 + (1000 + i * 10) * DAY) for i in range(3)]

    found = find(entries, insights, outcomes=outcomes, trades=trades)

    # six qualify after the per-kind cap: silent, open theses, two of each other kind
    assert [c.kind for c in found.patterns] == [
        PatternKind.SILENT_TRADES,
        PatternKind.OPEN_THESES,
        PatternKind.OUTCOMES_BY_STATE,
        PatternKind.OUTCOMES_BY_STATE,
        PatternKind.FELT_VS_DID,
    ]


def test_needs_report_the_smallest_shortfall_across_subjects() -> None:
    entries = [entry(1, T0), entry(2, T0 + 10 * DAY), entry(3, T0 + 20 * DAY)]
    insights = {1: insight(Emotion.CALM), 2: insight(Emotion.CALM), 3: insight(Emotion.FOMO)}

    found = find(entries, insights)

    assert more(found, PatternKind.FELT_VS_DID) == 1


def test_qualifying_kinds_have_no_need() -> None:
    entries = anxious_entries(3)
    trades = [trade(1, entries[0].timestamp + DAY)]

    found = find(entries, all_anxious(entries), trades=trades)

    assert more(found, PatternKind.FELT_VS_DID) is None
    assert all(n.more >= 1 for n in found.needs)


def test_supporting_entries_are_the_newest_five() -> None:
    entries = anxious_entries(8)
    trades = [trade(1, entries[0].timestamp + DAY)]

    [candidate] = of_kind(
        find(entries, all_anxious(entries), trades=trades), PatternKind.FELT_VS_DID
    )

    assert candidate.entry_ids == [8, 7, 6, 5, 4]


def test_entries_up_to_is_the_newest_entry_with_an_insight() -> None:
    entries = [entry(1, T0), entry(2, T0 + DAY), entry(3, T0 + 2 * DAY)]

    found = find(entries, {1: insight(), 2: insight()})

    assert found.entries_up_to == T0 + DAY
