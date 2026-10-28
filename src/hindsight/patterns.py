from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import median
from typing import Literal

from pydantic import Field

from hindsight.budget import budget_day
from hindsight.enrichment import EntryInput, EntryInsight, Ticker
from hindsight.pattern_reports import MAX_PATTERNS, Figure, Need
from hindsight.vocab import Emotion, EntryType, Horizon, IdeaSource, PatternKind
from hindsight.wire import WireModel

MAX_OUTCOMES = 500
MAX_TRADES = 1000
MAX_HOLDING_PERIODS = 1000
MAX_AUTO_ENTRY_IDS = 1000

# newest entries read for a report; no more than vestry can flag as auto entries
MAX_ENTRIES = MAX_AUTO_ENTRY_IDS

MIN_SUPPORT = 3
MAX_PER_KIND = 2
MAX_ENTRIES_PER_PATTERN = 5
# a trade follows an entry when it comes after it, at most this much later
FELT_WINDOW = timedelta(days=7)
# a trade is silent without a written entry this close to it on either side
SILENT_WINDOW = timedelta(days=1)
# a buy is held shorter than its horizon when it's sold within this many days;
# nothing is shorter than days
SHORTER_THAN = {Horizon.WEEKS: 14, Horizon.MONTHS: 90, Horizon.YEARS: 365}
# outcomes_by_state subjects, in ranking order for ties
STATES = [
    *(f"emotion:{emotion}" for emotion in Emotion),
    "conviction:high",
    "conviction:low",
    *(f"idea_source:{source}" for source in IdeaSource),
]


# vestry's market and ledger facts; hindsight trusts the numbers
class Outcome(WireModel):
    entry_id: int
    price: float = Field(gt=0)
    # SOLD: the closing sale of the entry's holding period; LATEST: the latest stored price
    kind: Literal["SOLD", "LATEST"]
    on: date


class LedgerTrade(WireModel):
    trade_id: int
    ticker: Ticker
    side: Literal["BUY", "SELL"]
    shares: float = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime
    # recorded at portfolio creation
    initial: bool


class HoldingPeriod(WireModel):
    ticker: Ticker
    opened_at: datetime
    closed_at: datetime | None
    # realized if closed, unrealized at the latest price if open, none without a price
    pnl: float | None


class PatternReportRequest(WireModel):
    as_of: date
    outcomes: list[Outcome] = Field(default=[], max_length=MAX_OUTCOMES)
    trades: list[LedgerTrade] = Field(default=[], max_length=MAX_TRADES)
    holding_periods: list[HoldingPeriod] = Field(default=[], max_length=MAX_HOLDING_PERIODS)
    # entries vestry wrote itself; they count as unwritten
    auto_entry_ids: list[int] = Field(default=[], max_length=MAX_AUTO_ENTRY_IDS)


@dataclass(frozen=True)
class Candidate:
    kind: PatternKind
    subject: str | None
    figures: list[Figure]
    support: int
    # the only entries the model sees and may cite, newest first
    entry_ids: list[int]


@dataclass(frozen=True)
class Candidates:
    patterns: list[Candidate]
    needs: list[Need]
    # the newest entry with extracted fields
    entries_up_to: datetime | None


# one kind and subject: a candidate if it qualifies, otherwise how many more examples it needs
@dataclass(frozen=True)
class Group:
    kind: PatternKind
    more: int
    candidate: Candidate | None = None


# entries must be the user's own; facts about anyone else's entries are ignored.
# entries_from is set when older entries weren't read: the kinds that compare the
# whole ledger with the journal then only look at what happened from there on
def find_patterns(
    facts: PatternReportRequest,
    entries: Iterable[EntryInput],
    insights: Mapping[int, EntryInsight],
    entries_from: datetime | None = None,
) -> Candidates:
    entries = list(entries)
    auto = set(facts.auto_entry_ids)
    written = sorted(
        (e for e in entries if e.id not in auto and e.body.strip()), key=lambda e: e.timestamp
    )
    # the kinds that need extracted fields leave out entries that don't have them yet
    read = [(e, insights[e.id]) for e in written if e.id in insights]
    moves = [t for t in facts.trades if not t.initial]
    outcomes = {o.entry_id: o for o in facts.outcomes}
    closed = [p for p in facts.holding_periods if p.closed_at is not None and p.pnl is not None]
    if entries_from is not None:
        # a trade needs its whole window read before it can count as silent
        silent_from = entries_from + SILENT_WINDOW
        audited = [t for t in moves if t.timestamp >= silent_from]
        closed = [p for p in closed if p.opened_at >= entries_from]
    else:
        audited = moves

    groups = [
        *felt_vs_did(read, moves),
        *outcomes_by_state(read, outcomes),
        *horizon_vs_holding(read, facts.holding_periods),
        gains_vs_losses(written, closed),
        silent_trades(written, audited),
        open_theses(read, written),
    ]
    return Candidates(
        patterns=rank(groups),
        needs=needs(groups),
        entries_up_to=max((e.timestamp for e in entries if e.id in insights), default=None),
    )


def felt_vs_did(
    read: list[tuple[EntryInput, EntryInsight]], moves: list[LedgerTrade]
) -> list[Group]:
    groups: list[Group] = []
    for emotion in Emotion:
        felt = [e for e, insight in read if emotion in insight.emotions]
        if not felt:
            continue
        sides = [{t.side for t in moves if follows(t, e)} for e in felt]
        buys = sum("BUY" in s for s in sides)
        sells = sum("SELL" in s for s in sides)
        figures = [count("entries", len(felt)), count("followed_by_buy", buys)]
        figures.append(count("followed_by_sell", sells))
        groups.append(
            group(PatternKind.FELT_VS_DID, emotion, figures, len(felt), buys + sells > 0, felt)
        )
    return groups


# strictly after, so an entry stamped with its own trade isn't followed by it
def follows(trade: LedgerTrade, entry: EntryInput) -> bool:
    same_ticker = entry.ticker is None or trade.ticker == entry.ticker
    return same_ticker and timedelta(0) < trade.timestamp - entry.timestamp <= FELT_WINDOW


def outcomes_by_state(
    read: list[tuple[EntryInput, EntryInsight]], outcomes: Mapping[int, Outcome]
) -> list[Group]:
    by_state: dict[str, list[tuple[EntryInput, float]]] = {}
    for entry, insight in read:
        outcome = outcomes.get(entry.id)
        if outcome is None or entry.price_snapshot is None:
            continue
        for state in states(insight):
            by_state.setdefault(state, []).append((entry, outcome.price - entry.price_snapshot))

    groups: list[Group] = []
    for state in STATES:
        scored = by_state.get(state)
        if not scored:
            continue
        above = sum(change > 0 for _, change in scored)
        below = sum(change < 0 for _, change in scored)
        figures = [count("entries", len(scored)), count("above_entry_price", above)]
        figures.append(count("below_entry_price", below))
        entries = [e for e, _ in scored]
        groups.append(
            group(
                PatternKind.OUTCOMES_BY_STATE, state, figures, len(scored), above != below, entries
            )
        )
    return groups


def states(insight: EntryInsight) -> list[str]:
    found = [f"emotion:{emotion}" for emotion in insight.emotions]
    if insight.conviction is not None and insight.conviction >= 4:
        found.append("conviction:high")
    if insight.conviction is not None and insight.conviction <= 2:
        found.append("conviction:low")
    if insight.idea_source is not None:
        found.append(f"idea_source:{insight.idea_source}")
    return found


def horizon_vs_holding(
    read: list[tuple[EntryInput, EntryInsight]], periods: Sequence[HoldingPeriod]
) -> list[Group]:
    by_horizon: dict[Horizon, list[tuple[EntryInput, HoldingPeriod]]] = {}
    for entry, insight in read:
        if entry.entry_type != EntryType.BUY or insight.horizon is None:
            continue
        period = holding(entry, periods)
        if period is not None:
            by_horizon.setdefault(insight.horizon, []).append((entry, period))

    groups: list[Group] = []
    for horizon in Horizon:
        buys = by_horizon.get(horizon)
        if not buys:
            continue
        limit = SHORTER_THAN.get(horizon)
        shorter = sum(
            p.closed_at is not None
            and limit is not None
            and days_held(e.timestamp, p.closed_at) <= limit
            for e, p in buys
        )
        still_open = sum(p.closed_at is None for _, p in buys)
        figures = [count("buys", len(buys)), count("held_shorter", shorter)]
        figures.append(count("still_open", still_open))
        entries = [e for e, _ in buys]
        groups.append(
            group(PatternKind.HORIZON_VS_HOLDING, horizon, figures, len(buys), shorter > 0, entries)
        )
    return groups


# the period on the entry's ticker that contains it
def holding(entry: EntryInput, periods: Sequence[HoldingPeriod]) -> HoldingPeriod | None:
    for period in periods:
        if (
            period.ticker == entry.ticker
            and period.opened_at <= entry.timestamp
            and (period.closed_at is None or entry.timestamp <= period.closed_at)
        ):
            return period
    return None


def gains_vs_losses(written: list[EntryInput], closed: list[HoldingPeriod]) -> Group:
    gains = [p for p in closed if p.pnl is not None and p.pnl > 0]
    losses = [p for p in closed if p.pnl is not None and p.pnl < 0]
    # both sides need the minimum; needs reports the side that's further off
    more = max(MIN_SUPPORT - len(gains), MIN_SUPPORT - len(losses), 0)
    if more:
        return Group(PatternKind.GAINS_VS_LOSSES, more)

    gains_entries = [during(written, p) for p in gains]
    losses_entries = [during(written, p) for p in losses]
    gains_days = median(days_held(p.opened_at, p.closed_at) for p in gains if p.closed_at)
    losses_days = median(days_held(p.opened_at, p.closed_at) for p in losses if p.closed_at)
    figures = [
        count("gains", len(gains)),
        count("losses", len(losses)),
        Figure(key="gains_median_days", value=gains_days, unit="days"),
        Figure(key="losses_median_days", value=losses_days, unit="days"),
        count("gains_with_entry", sum(bool(e) for e in gains_entries)),
        count("losses_with_entry", sum(bool(e) for e in losses_entries)),
    ]
    entries = [e for found in gains_entries + losses_entries for e in found]
    return group(
        PatternKind.GAINS_VS_LOSSES,
        None,
        figures,
        len(gains) + len(losses),
        gains_days != losses_days,
        entries,
    )


# written entries on the period's ticker while it was held
def during(written: list[EntryInput], period: HoldingPeriod) -> list[EntryInput]:
    return [
        e
        for e in written
        if e.ticker == period.ticker
        and period.opened_at <= e.timestamp
        and (period.closed_at is None or e.timestamp <= period.closed_at)
    ]


def silent_trades(written: list[EntryInput], moves: list[LedgerTrade]) -> Group:
    silent = sum(
        not any(
            e.ticker == t.ticker and abs(e.timestamp - t.timestamp) <= SILENT_WINDOW
            for e in written
        )
        for t in moves
    )
    figures = [count("trades", len(moves)), count("without_entry", silent)]
    # the subject is the missing entry, so there's nothing to cite
    return group(PatternKind.SILENT_TRADES, None, figures, len(moves), silent > 0, [])


def open_theses(read: list[tuple[EntryInput, EntryInsight]], written: list[EntryInput]) -> Group:
    buys = [
        e
        for e, insight in read
        if e.entry_type == EntryType.BUY
        and e.ticker is not None
        and (insight.thesis is not None or insight.exit_condition is not None)
    ]
    reflections = [e for e in written if e.entry_type == EntryType.REFLECTION]
    not_revisited = sum(
        not any(r.ticker == b.ticker and r.timestamp > b.timestamp for r in reflections)
        for b in buys
    )
    figures = [count("buys_with_thesis", len(buys)), count("not_revisited", not_revisited)]
    return group(PatternKind.OPEN_THESES, None, figures, len(buys), not_revisited > 0, buys)


def group(
    kind: PatternKind,
    subject: str | None,
    figures: list[Figure],
    support: int,
    has_gap: bool,
    entries: list[EntryInput],
) -> Group:
    if support < MIN_SUPPORT:
        return Group(kind, MIN_SUPPORT - support)
    # no "none of them" patterns; one more example might open a gap
    if not has_gap:
        return Group(kind, 1)
    candidate = Candidate(kind, subject, figures, support, newest(entries))
    return Group(kind, 0, candidate)


def newest(entries: list[EntryInput]) -> list[int]:
    ordered = sorted({e.id: e for e in entries}.values(), key=lambda e: (e.timestamp, e.id))
    return [e.id for e in reversed(ordered)][:MAX_ENTRIES_PER_PATTERN]


# groups arrive in kind order and then subject order, which the stable sort keeps for ties
def rank(groups: list[Group]) -> list[Candidate]:
    ranked = sorted((g.candidate for g in groups if g.candidate), key=lambda c: -c.support)
    kept: list[Candidate] = []
    for candidate in ranked:
        if sum(c.kind == candidate.kind for c in kept) < MAX_PER_KIND:
            kept.append(candidate)
    return kept[:MAX_PATTERNS]


# the smallest shortfall for each kind with no qualifying candidate
def needs(groups: list[Group]) -> list[Need]:
    found: list[Need] = []
    for kind in PatternKind:
        of_kind = [g for g in groups if g.kind == kind]
        if any(g.candidate for g in of_kind):
            continue
        found.append(Need(kind=kind, more=min((g.more for g in of_kind), default=MIN_SUPPORT)))
    return found


def count(key: str, value: int) -> Figure:
    return Figure(key=key, value=value, unit="count")


# whole new york days, so a buy at 3pm sold the next morning is held one day
def days_held(start: datetime, end: datetime) -> int:
    return (budget_day(end) - budget_day(start)).days
