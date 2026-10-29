from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from math import ceil
from statistics import fmean, median
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from hindsight.behavior_reports import MAX_BEHAVIORS, Figure, Need, Previous
from hindsight.budget import BUDGET_ZONE, budget_day
from hindsight.enrichment import EntryInput, EntryInsight, Ticker
from hindsight.vocab import (
    BehaviorKind,
    Emotion,
    Horizon,
    IdeaSource,
    ReasoningBasis,
    ThesisOutcome,
)
from hindsight.wire import WireModel

MAX_TRADES = 1000
MAX_HOLDING_PERIODS = 1000
MAX_HOLDINGS = 200
MAX_AUTO_ENTRY_IDS = 1000
MAX_FEEDBACK = 200
MAX_BEHAVIOR_NOTES = 500
MAX_NOTE_LENGTH = 2000

Fraction = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Money = Annotated[float, Field(allow_inf_nan=False)]
# a behavior's identity; kinds hindsight doesn't know (or no longer knows) are ignored
Kind = Annotated[str, Field(max_length=40)]
Subject = Annotated[str, Field(max_length=40)]


# vestry's market and ledger facts; hindsight trusts the numbers
class LedgerTrade(WireModel):
    trade_id: int
    ticker: Ticker
    side: Literal["BUY", "SELL"]
    shares: Positive
    price: Positive
    timestamp: datetime
    # recorded at portfolio creation
    initial: bool
    # buys only: the change over the 20 trading days before the buy's New York day
    run_up: Fraction | None = None
    run_up_status: Literal["KNOWN", "PENDING", "UNAVAILABLE"] = "UNAVAILABLE"
    # buys only: the trade's value over open cost basis just before it plus the trade's value
    weight_at_buy: NonNegative | None = None
    # buys only. SOLD: the sale that closed this buy's holding period; LATEST: the latest
    # stored price while it's open. all three are none when unknown
    outcome_price: Positive | None = None
    outcome_kind: Literal["SOLD", "LATEST"] | None = None
    outcome_on: date | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.run_up is not None) != (self.run_up_status == "KNOWN"):
            raise ValueError("runUp is set exactly when runUpStatus is KNOWN")
        outcome = (self.outcome_price, self.outcome_kind, self.outcome_on)
        if any(o is None for o in outcome) and any(o is not None for o in outcome):
            raise ValueError("outcomePrice, outcomeKind and outcomeOn are all set or all null")
        return self


class HoldingPeriod(WireModel):
    ticker: Ticker
    opened_at: datetime
    closed_at: datetime | None
    # the period's total buy cost
    cost: Positive
    # realized if closed, realized plus unrealized if open, none without a price
    pnl: Money | None


class Holding(WireModel):
    ticker: Ticker
    market_value: Positive
    # share of portfolio_value
    weight: NonNegative
    sector: str | None = Field(default=None, max_length=100)
    # a bad month, about 1 in 20; none with fewer than 30 daily returns
    possible_monthly_loss: NonNegative | None = None


# the user's current reaction to one behavior identity
class Feedback(WireModel):
    kind: Kind
    subject: Subject | None
    reaction: Literal["FITS", "KNEW", "DOESNT_FIT"]
    note: str | None = Field(default=None, max_length=MAX_NOTE_LENGTH)
    # the report the reaction was given on
    report_day: date
    at: datetime


# an INSIGHT entry the user wrote from a behavior's Explore view; its text comes from core_read
class BehaviorNote(WireModel):
    kind: Kind
    subject: Subject | None
    entry_id: int


class BehaviorReportRequest(WireModel):
    # the New York day vestry computed the facts
    as_of: date
    # market value of open positions; none without prices
    portfolio_value: NonNegative | None = None
    # chronological, newest kept at the cap
    trades: list[LedgerTrade] = Field(default=[], max_length=MAX_TRADES)
    holding_periods: list[HoldingPeriod] = Field(default=[], max_length=MAX_HOLDING_PERIODS)
    holdings: list[Holding] = Field(default=[], max_length=MAX_HOLDINGS)
    # entries vestry wrote itself; they count as unwritten
    auto_entry_ids: list[int] = Field(default=[], max_length=MAX_AUTO_ENTRY_IDS)
    feedback: list[Feedback] = Field(default=[], max_length=MAX_FEEDBACK)
    behavior_notes: list[BehaviorNote] = Field(default=[], max_length=MAX_BEHAVIOR_NOTES)


# --- finding behaviors: pure, no i/o ---

MIN_DECISIONS = 3
MAX_PER_KIND = 2
# what the model sees per behavior
MAX_SHOWN_ENTRIES = 5
MAX_NOTES = 3
# a decision's linked entries run from this long before it to the end of the new york day after
LINK_BEFORE = timedelta(days=7)
RUN_UP = 0.10
ROUND_TRIP = timedelta(days=5)
BURST_TRADES = 3
BURST_FACTOR = 2
CONCENTRATED_WEIGHT = 0.20
CONCENTRATED_SECTOR = 0.40
HIGH_CONVICTION = 4
LOW_CONVICTION = 2
# a buy is sold sooner than planned when it's sold within this many days of its stated horizon
PLANNED_DAYS = {Horizon.MONTHS: 90, Horizon.YEARS: 365}
# decisions at which evidence counts in full
FULL_EVIDENCE = 8
FEEDBACK_FACTOR = {"FITS": 1.25, "KNEW": 0.75}
EMOTIONS = [Emotion.ANXIOUS, Emotion.EXCITED, Emotion.REGRET, Emotion.FRUSTRATED]
HYPE_SOURCES = {IdeaSource.NEWS, IdeaSource.SOCIAL_MEDIA}
HYPE_EMOTIONS = {Emotion.EXCITED, Emotion.FOMO}
CHASED_BASES = {ReasoningBasis.PRICE_ACTION, ReasoningBasis.COMMENTARY}

Reason = Literal["ENTRIES", "TRADES", "PRICES", "FIELDS"]


# a behavior identity in an earlier saved report
@dataclass(frozen=True)
class PastBehavior:
    kind: str
    subject: str | None
    figures: list[Figure]
    trade_ids: list[int]


@dataclass(frozen=True)
class PastReport:
    day: date
    behaviors: list[PastBehavior]


@dataclass(frozen=True)
class FoundBehavior:
    kind: BehaviorKind
    subject: str | None
    figures: list[Figure]
    previous: Previous | None
    # every decision and linked entry in the group, newest first
    trade_ids: list[int]
    entry_ids: list[int]
    # the only entries the model sees and may cite: the ones that placed a decision in the
    # group first, then the newest
    shown_entry_ids: list[int]
    # context for the wording, never for the figures; at most MAX_NOTES together
    feedback_notes: list[str]
    note_entry_ids: list[int]
    score: float


@dataclass(frozen=True)
class Findings:
    behaviors: list[FoundBehavior]
    needs: list[Need]
    # the newest entry with extracted fields
    entries_up_to: datetime | None


# one unit a behavior counts: a buy, a trade, a holding period or a holding
@dataclass(frozen=True)
class Decision:
    trade_ids: tuple[int, ...]
    # none when the outcome is unknown; such decisions still count toward decisions
    pnl: float | None
    ret: float | None
    entries: tuple[EntryInput, ...] = ()


@dataclass(frozen=True)
class Group:
    kind: BehaviorKind
    subject: str | None
    members: list[Decision]
    baseline: list[Decision]
    # what the group waits for when it's short
    reason: Reason
    extras: list[Figure] = field(default_factory=list[Figure])
    # entries whose fields put a decision in the group
    relevant: frozenset[int] = frozenset()
    # replaces the default impact (risk_concentration)
    impact: float | None = None


@dataclass(frozen=True)
class Book:
    facts: BehaviorReportRequest
    insights: Mapping[int, EntryInsight]
    # non-initial trades, chronological
    trades: list[LedgerTrade]
    # trade id -> written entries linked to it, newest first
    linked: dict[int, list[EntryInput]]
    # written entries by ticker, chronological
    by_ticker: dict[str, list[EntryInput]]


# entries must be the user's own. history is the user's earlier saved reports, newest first.
# entries_from is set when older entries weren't read: undocumented then only judges trades
# whose whole window was read
def find_behaviors(
    facts: BehaviorReportRequest,
    entries: Iterable[EntryInput],
    insights: Mapping[int, EntryInsight],
    history: Sequence[PastReport] = (),
    entries_from: datetime | None = None,
) -> Findings:
    entries = list(entries)
    book = build(facts, entries, insights)
    groups = [
        no_plan(book),
        undocumented(book, entries_from),
        *(emotional(book, emotion) for emotion in EMOTIONS),
        fomo(book),
        hype(book),
        run_up(book),
        overtrading(book),
        overconfidence(book),
        price_action(book),
        loss_aversion(book),
        short_term(book),
        risk_concentration(book),
        position_sizing(book),
    ]
    found = [behavior(g, book, history) for g in groups if len(g.members) >= MIN_DECISIONS]
    return Findings(
        behaviors=rank([b for b in found if b is not None]),
        needs=[
            Need(
                kind=g.kind, subject=g.subject, more=MIN_DECISIONS - len(g.members), reason=g.reason
            )
            for g in groups
            if len(g.members) < MIN_DECISIONS
        ],
        entries_up_to=max((e.timestamp for e in entries if e.id in insights), default=None),
    )


def build(
    facts: BehaviorReportRequest, entries: list[EntryInput], insights: Mapping[int, EntryInsight]
) -> Book:
    unwritten = {*facts.auto_entry_ids, *(n.entry_id for n in facts.behavior_notes)}
    by_ticker: dict[str, list[EntryInput]] = {}
    for entry in sorted(entries, key=lambda e: (e.timestamp, e.id)):
        if entry.id in unwritten or not entry.body.strip():
            continue
        for ticker in tickers_of(entry, insights.get(entry.id)):
            by_ticker.setdefault(ticker, []).append(entry)
    trades = sorted((t for t in facts.trades if not t.initial), key=lambda t: t.timestamp)
    linked = {t.trade_id: linked_to(t, by_ticker.get(t.ticker, [])) for t in trades}
    return Book(facts, insights, trades, linked, by_ticker)


def linked_to(trade: LedgerTrade, entries: list[EntryInput]) -> list[EntryInput]:
    start, end = trade.timestamp - LINK_BEFORE, link_end(trade.timestamp)
    return [e for e in reversed(entries) if start <= e.timestamp < end]


# an entry's own ticker, or the ones its text names when it has none
def tickers_of(entry: EntryInput, insight: EntryInsight | None) -> set[str]:
    if entry.ticker is not None:
        return {entry.ticker}
    return set(insight.tickers) if insight else set()


def link_end(at: datetime) -> datetime:
    return datetime.combine(budget_day(at) + timedelta(days=2), time(), tzinfo=BUDGET_ZONE)


# --- decisions ---


def buy_decision(book: Book, trade: LedgerTrade, extra: Iterable[EntryInput] = ()) -> Decision:
    entries = (*book.linked.get(trade.trade_id, []), *extra)
    if trade.side == "SELL" or trade.outcome_price is None:
        # the contract carries outcomes for buys only
        return Decision((trade.trade_id,), None, None, entries)
    pnl = trade.shares * (trade.outcome_price - trade.price)
    return Decision((trade.trade_id,), pnl, trade.outcome_price / trade.price - 1, entries)


def period_trades(book: Book, period: HoldingPeriod) -> list[LedgerTrade]:
    return [
        t for t in book.facts.trades if t.ticker == period.ticker and inside(t.timestamp, period)
    ]


def inside(at: datetime, period: HoldingPeriod) -> bool:
    return period.opened_at <= at and (period.closed_at is None or at <= period.closed_at)


def period_decision(book: Book, period: HoldingPeriod, entries: Iterable[EntryInput]) -> Decision:
    trade_ids = tuple(t.trade_id for t in period_trades(book, period))
    if period.pnl is None:
        return Decision(trade_ids, None, None, tuple(entries))
    return Decision(trade_ids, period.pnl, period.pnl / period.cost, tuple(entries))


def period_entries(book: Book, period: HoldingPeriod) -> list[EntryInput]:
    found = {e.id: e for t in period_trades(book, period) for e in book.linked.get(t.trade_id, [])}
    return list(found.values())


# what a documented decision's linked entries say, newest first; none while any lacks fields
def read(book: Book, trade: LedgerTrade) -> list[tuple[EntryInput, EntryInsight]] | None:
    found = [(e, book.insights.get(e.id)) for e in book.linked.get(trade.trade_id, [])]
    if any(i is None for _, i in found):
        return None
    return [(e, i) for e, i in found if i is not None]


@dataclass(frozen=True)
class Documented:
    trades: list[tuple[LedgerTrade, list[tuple[EntryInput, EntryInsight]]]]
    # documented decisions whose entries aren't extracted yet
    pending: int

    @property
    def reason(self) -> Reason:
        return "FIELDS" if self.pending else "ENTRIES"


def documented(book: Book, trades: Iterable[LedgerTrade]) -> Documented:
    found: list[tuple[LedgerTrade, list[tuple[EntryInput, EntryInsight]]]] = []
    pending = 0
    for trade in trades:
        if not book.linked.get(trade.trade_id):
            continue
        said = read(book, trade)
        if said is None:
            pending += 1
        else:
            found.append((trade, said))
    return Documented(found, pending)


def buys(book: Book) -> list[LedgerTrade]:
    return [t for t in book.trades if t.side == "BUY"]


# the newest linked entry that states the field
def newest[T](
    said: list[tuple[EntryInput, EntryInsight]], get: Callable[[EntryInsight], T | None]
) -> tuple[EntryInput, T] | None:
    for entry, insight in said:
        value = get(insight)
        if value is not None:
            return entry, value
    return None


def planned(insight: EntryInsight) -> bool:
    return any(f is not None for f in (insight.thesis, insight.exit_condition, insight.horizon))


# --- kinds ---


def no_plan(book: Book) -> Group:
    docs = documented(book, buys(book))
    members = [
        buy_decision(book, t) for t, said in docs.trades if not any(planned(i) for _, i in said)
    ]
    baseline = [
        buy_decision(book, t) for t, said in docs.trades if any(planned(i) for _, i in said)
    ]
    return Group(BehaviorKind.NO_PLAN, None, members, baseline, docs.reason)


def undocumented(book: Book, entries_from: datetime | None) -> Group:
    trades = book.trades
    if entries_from is not None:
        trades = [t for t in trades if t.timestamp - LINK_BEFORE >= entries_from]
    members = [buy_decision(book, t) for t in trades if not book.linked[t.trade_id]]
    baseline = [buy_decision(book, t) for t in trades if book.linked[t.trade_id]]
    return Group(BehaviorKind.UNDOCUMENTED, None, members, baseline, "TRADES")


def emotional(book: Book, emotion: Emotion) -> Group:
    docs = documented(book, book.trades)
    members: list[Decision] = []
    baseline: list[Decision] = []
    relevant: set[int] = set()
    for trade, said in docs.trades:
        named = {e for _, i in said for e in i.emotions}
        if emotion in named:
            members.append(buy_decision(book, trade))
            relevant.update(e.id for e, i in said if emotion in i.emotions)
        elif not named & set(EMOTIONS):
            baseline.append(buy_decision(book, trade))
    return Group(
        BehaviorKind.EMOTIONAL,
        emotion.value,
        members,
        baseline,
        docs.reason,
        relevant=frozenset(relevant),
    )


def fomo(book: Book) -> Group:
    docs = documented(book, buys(book))
    members: list[Decision] = []
    baseline: list[Decision] = []
    relevant: set[int] = set()
    from_news_or_social = 0
    for trade, said in docs.trades:
        if any(Emotion.FOMO in i.emotions for _, i in said):
            members.append(buy_decision(book, trade))
            relevant.update(e.id for e, i in said if Emotion.FOMO in i.emotions)
            from_news_or_social += any(i.idea_source in HYPE_SOURCES for _, i in said)
        else:
            baseline.append(buy_decision(book, trade))
    extras = [count("from_news_or_social", from_news_or_social)]
    return Group(
        BehaviorKind.FOMO, None, members, baseline, docs.reason, extras, frozenset(relevant)
    )


def hype(book: Book) -> Group:
    docs = documented(book, buys(book))
    members: list[Decision] = []
    baseline: list[Decision] = []
    run_ups: list[float] = []
    relevant: set[int] = set()
    for trade, said in docs.trades:
        sourced = [e.id for e, i in said if i.idea_source in HYPE_SOURCES]
        excited = any(set(i.emotions) & HYPE_EMOTIONS for _, i in said)
        rose = trade.run_up is not None and trade.run_up >= RUN_UP
        if sourced and (excited or rose):
            members.append(buy_decision(book, trade))
            relevant.update(sourced)
            if trade.run_up is not None:
                run_ups.append(trade.run_up)
        else:
            baseline.append(buy_decision(book, trade))
    extras = [fraction("median_run_up", median(run_ups))] if run_ups else []
    return Group(
        BehaviorKind.HYPE, None, members, baseline, docs.reason, extras, frozenset(relevant)
    )


def run_up(book: Book) -> Group:
    known = [t for t in buys(book) if t.run_up is not None]
    pending = any(t.run_up_status == "PENDING" for t in buys(book))
    rose = [t for t in known if t.run_up is not None and t.run_up >= RUN_UP]
    calm = [t for t in known if t.run_up is not None and t.run_up < RUN_UP]
    extras = (
        [fraction("median_run_up", median(t.run_up for t in rose if t.run_up is not None))]
        if rose
        else []
    )
    if len(calm) >= MIN_DECISIONS:
        extras.append(
            fraction(
                "baseline_median_run_up", median(t.run_up for t in calm if t.run_up is not None)
            )
        )
    return Group(
        BehaviorKind.RUN_UP,
        None,
        [buy_decision(book, t) for t in rose],
        [buy_decision(book, t) for t in calm],
        "PRICES" if pending else "TRADES",
        extras,
    )


def overtrading(book: Book) -> Group:
    weeks = Counter(week_of(t.timestamp) for t in book.trades)
    typical = median(weeks.values()) if weeks else 0
    bursts = {w for w, n in weeks.items() if n >= BURST_TRADES and n >= BURST_FACTOR * typical}
    trips = [
        p
        for p in book.facts.holding_periods
        if p.closed_at is not None and p.closed_at - p.opened_at < ROUND_TRIP
    ]
    members: list[Decision] = []
    baseline: list[Decision] = []
    for trade in book.trades:
        busy = week_of(trade.timestamp) in bursts
        tripped = any(p.ticker == trade.ticker and inside(trade.timestamp, p) for p in trips)
        (members if busy or tripped else baseline).append(buy_decision(book, trade))
    extras = [
        count("burst_weeks", len(bursts)),
        count("round_trips", len(trips)),
        Figure(key="median_trades_per_week", value=typical, unit="count"),
    ]
    return Group(BehaviorKind.OVERTRADING, None, members, baseline, "TRADES", extras)


# the monday of the trade's new york week
def week_of(at: datetime) -> date:
    day = budget_day(at)
    return day - timedelta(days=day.weekday())


def overconfidence(book: Book) -> Group:
    docs = documented(book, buys(book))
    outcomes = thesis_outcomes(book)
    members: list[Decision] = []
    baseline: list[Decision] = []
    relevant: set[int] = set()
    wrong_reason = 0
    for trade, said in docs.trades:
        stated = newest(said, lambda i: i.conviction)
        if stated is None:
            continue
        entry, conviction = stated
        if conviction >= HIGH_CONVICTION:
            looked_back = outcomes.get(trade.trade_id)
            decision = buy_decision(book, trade, [looked_back[0]] if looked_back else [])
            members.append(decision)
            relevant.add(entry.id)
            gained = decision.pnl is not None and decision.pnl > 0
            wrong_reason += (
                gained and looked_back is not None and looked_back[1] == ThesisOutcome.DID_NOT
            )
        else:
            baseline.append(buy_decision(book, trade))
    extras = [count("right_for_wrong_reason", wrong_reason)]
    return Group(
        BehaviorKind.OVERCONFIDENCE,
        None,
        members,
        baseline,
        docs.reason,
        extras,
        frozenset(relevant),
    )


# a look-back entry belongs to the most recent earlier buy on its ticker; the newest one wins
def thesis_outcomes(book: Book) -> dict[int, tuple[EntryInput, ThesisOutcome]]:
    found: dict[int, tuple[EntryInput, ThesisOutcome]] = {}
    for ticker, entries in book.by_ticker.items():
        earlier = [t for t in buys(book) if t.ticker == ticker]
        for entry in entries:
            insight = book.insights.get(entry.id)
            if insight is None or insight.thesis_outcome is None:
                continue
            before = [t for t in earlier if t.timestamp < entry.timestamp]
            if before:
                found[before[-1].trade_id] = (entry, insight.thesis_outcome)
    return found


def price_action(book: Book) -> Group:
    docs = documented(book, buys(book))
    members: list[Decision] = []
    baseline: list[Decision] = []
    relevant: set[int] = set()
    for trade, said in docs.trades:
        stated = newest(said, lambda i: i.reasoning_basis)
        if stated is None:
            continue
        entry, basis = stated
        if basis in CHASED_BASES:
            members.append(buy_decision(book, trade))
            relevant.add(entry.id)
        elif basis == ReasoningBasis.FUNDAMENTALS:
            baseline.append(buy_decision(book, trade))
    return Group(
        BehaviorKind.PRICE_ACTION,
        None,
        members,
        baseline,
        docs.reason,
        relevant=frozenset(relevant),
    )


def loss_aversion(book: Book) -> Group:
    closed = [
        p for p in book.facts.holding_periods if p.closed_at is not None and p.pnl is not None
    ]
    losses = [p for p in closed if p.pnl is not None and p.pnl < 0]
    gains = [p for p in closed if p.pnl is not None and p.pnl > 0]
    extras: list[Figure] = []
    if losses:
        extras.append(days("median_days", median(held(p) for p in losses)))
    if len(gains) >= MIN_DECISIONS:
        gain_days = median(held(p) for p in gains)
        extras.append(days("baseline_median_days", gain_days))
        still_open = [
            p
            for p in book.facts.holding_periods
            if p.closed_at is None
            and p.pnl is not None
            and p.pnl < 0
            and (book.facts.as_of - budget_day(p.opened_at)).days > gain_days
        ]
        extras.append(count("open_losses_held_longer", len(still_open)))
    return Group(
        BehaviorKind.LOSS_AVERSION,
        None,
        [period_decision(book, p, period_entries(book, p)) for p in losses],
        [period_decision(book, p, period_entries(book, p)) for p in gains],
        "TRADES",
        extras,
    )


# whole new york days, so a buy at 3pm sold the next morning is held one day
def held(period: HoldingPeriod) -> int:
    assert period.closed_at is not None
    return days_between(period.opened_at, period.closed_at)


def days_between(start: datetime, end: datetime) -> int:
    return (budget_day(end) - budget_day(start)).days


def short_term(book: Book) -> Group:
    docs = documented(book, buys(book))
    members: list[Decision] = []
    baseline: list[Decision] = []
    relevant: set[int] = set()
    sooner: list[int] = []
    for trade, said in docs.trades:
        stated = newest(said, lambda i: i.horizon)
        period = holding_of(book, trade)
        if stated is None or period is None or stated[1] not in PLANNED_DAYS:
            continue
        entry, horizon = stated
        kept = None if period.closed_at is None else days_between(trade.timestamp, period.closed_at)
        if kept is not None and kept <= PLANNED_DAYS[horizon]:
            members.append(buy_decision(book, trade))
            relevant.add(entry.id)
            sooner.append(kept)
        else:
            baseline.append(buy_decision(book, trade))
    extras = [days("median_days", median(sooner))] if sooner else []
    return Group(
        BehaviorKind.SHORT_TERM, None, members, baseline, docs.reason, extras, frozenset(relevant)
    )


# the holding period on the trade's ticker that contains it
def holding_of(book: Book, trade: LedgerTrade) -> HoldingPeriod | None:
    for period in book.facts.holding_periods:
        if period.ticker == trade.ticker and inside(trade.timestamp, period):
            return period
    return None


def open_period(book: Book, ticker: str) -> HoldingPeriod | None:
    found = [p for p in book.facts.holding_periods if p.ticker == ticker and p.closed_at is None]
    return max(found, key=lambda p: p.opened_at, default=None)


def risk_concentration(book: Book) -> Group:
    holdings = book.facts.holdings
    sectors: dict[str, float] = {}
    for h in holdings:
        if h.sector is not None:
            sectors[h.sector] = sectors.get(h.sector, 0) + h.weight
    members: list[tuple[Holding, Decision]] = []
    baseline: list[Decision] = []
    pending_prices = pending_fields = False
    for h in holdings:
        period = open_period(book, h.ticker)
        since = period.opened_at - LINK_BEFORE if period else None
        entries = [
            e for e in book.by_ticker.get(h.ticker, []) if since is None or e.timestamp >= since
        ]
        decision = (
            period_decision(book, period, entries)
            if period
            else Decision((), None, None, tuple(entries))
        )
        concentrated = h.weight > CONCENTRATED_WEIGHT or (
            h.sector is not None and sectors[h.sector] > CONCENTRATED_SECTOR
        )
        insights = [book.insights.get(e.id) for e in entries]
        unplanned = not any(i and i.exit_condition for i in insights)
        # a concentrated holding waiting on fields or prices is in neither group yet
        if concentrated and any(i is None for i in insights):
            pending_fields = True
        elif concentrated and unplanned and h.possible_monthly_loss is None:
            pending_prices = True
        elif concentrated and unplanned:
            members.append((h, decision))
        else:
            baseline.append(decision)
    loss = sum(h.possible_monthly_loss or 0 for h, _ in members)
    extras: list[Figure] = []
    if members:
        extras.append(money("possible_monthly_loss", loss))
        extras.append(fraction("largest_weight", max(h.weight for h, _ in members)))
        held_sectors = [sectors[h.sector] for h, _ in members if h.sector is not None]
        if held_sectors:
            extras.append(fraction("sector_weight", max(held_sectors)))
    reason: Reason = "PRICES" if pending_prices else "FIELDS" if pending_fields else "TRADES"
    return Group(
        BehaviorKind.RISK_CONCENTRATION,
        None,
        [d for _, d in members],
        baseline,
        reason,
        extras,
        impact=loss,
    )


def position_sizing(book: Book) -> Group:
    sized = sorted(
        (t for t in buys(book) if t.weight_at_buy is not None),
        key=lambda t: (-(t.weight_at_buy or 0), t.timestamp, t.trade_id),
    )
    top = sized[: ceil(len(sized) / 4)]
    rest = sized[len(top) :]
    extras: list[Figure] = []
    if top:
        extras.append(fraction("avg_weight", fmean(t.weight_at_buy or 0 for t in top)))
    if len(rest) >= MIN_DECISIONS:
        extras.append(fraction("baseline_avg_weight", fmean(t.weight_at_buy or 0 for t in rest)))
    if top:
        low = 0
        for trade in top:
            said = read(book, trade)
            stated = newest(said, lambda i: i.conviction) if said else None
            # undocumented buys state none; ones still being extracted aren't counted
            if said is not None and (stated is None or stated[1] <= LOW_CONVICTION):
                low += 1
        extras.append(count("large_low_conviction", low))
        held_now = {h.ticker: h for h in book.facts.holdings}
        still_open = {
            t.ticker
            for t in top
            if t.ticker in held_now
            and (p := holding_of(book, t)) is not None
            and p.closed_at is None
        }
        losses = [held_now[ticker].possible_monthly_loss for ticker in sorted(still_open)]
        if still_open and all(loss is not None for loss in losses):
            extras.append(money("possible_monthly_loss", sum(loss or 0 for loss in losses)))
    return Group(
        BehaviorKind.POSITION_SIZING,
        None,
        [buy_decision(book, t) for t in top],
        [buy_decision(book, t) for t in rest],
        "TRADES",
        extras,
    )


# --- figures, ranking, history ---


def standard(prefix: str, decisions: list[Decision]) -> list[Figure]:
    figures = [count(f"{prefix}decisions", len(decisions))]
    known = [d for d in decisions if d.pnl is not None and d.ret is not None]
    if known:
        pnl = sum(d.pnl or 0 for d in known)
        figures += [
            money(f"{prefix}pnl", pnl),
            money(f"{prefix}avg_pnl", pnl / len(known)),
            fraction(f"{prefix}avg_return", fmean(d.ret or 0 for d in known)),
            count(f"{prefix}gains", sum((d.pnl or 0) > 0 for d in known)),
        ]
    return figures


def figures_of(group: Group) -> list[Figure]:
    figures = standard("", group.members)
    if len(group.baseline) >= MIN_DECISIONS:
        figures += standard("baseline_", group.baseline)
    return figures + group.extras


def impact(group: Group, figures: list[Figure]) -> float:
    if group.impact is not None:
        return group.impact
    values = {f.key: f.value for f in figures}
    if "avg_pnl" in values and "baseline_avg_pnl" in values:
        return abs(values["avg_pnl"] - values["baseline_avg_pnl"]) * len(group.members)
    return abs(values.get("pnl", 0))


def behavior(group: Group, book: Book, history: Sequence[PastReport]) -> FoundBehavior | None:
    identity = (group.kind.value, group.subject)
    trade_ids = sorted({i for d in group.members for i in d.trade_ids}, reverse=True)
    feedback = latest_feedback(book.facts.feedback, identity)
    factor = 1.0
    if feedback is not None and feedback.reaction == "DOESNT_FIT":
        # hidden until the decisions behind it change
        if not changed_since(history, feedback.report_day, identity, trade_ids):
            return None
    elif feedback is not None:
        factor = FEEDBACK_FACTOR[feedback.reaction]

    figures = figures_of(group)
    entries = sorted(
        {e.id: e for d in group.members for e in d.entries}.values(),
        key=lambda e: (e.timestamp, e.id),
        reverse=True,
    )
    shown = sorted(entries, key=lambda e: e.id not in group.relevant)[:MAX_SHOWN_ENTRIES]
    notes = [feedback.note] if feedback is not None and feedback.note else []
    note_ids = sorted(
        (n.entry_id for n in book.facts.behavior_notes if (n.kind, n.subject) == identity),
        reverse=True,
    )
    return FoundBehavior(
        kind=group.kind,
        subject=group.subject,
        figures=figures,
        previous=previous(history, identity),
        trade_ids=trade_ids,
        entry_ids=[e.id for e in entries],
        shown_entry_ids=[e.id for e in shown],
        feedback_notes=notes,
        note_entry_ids=note_ids[: MAX_NOTES - len(notes)],
        score=impact(group, figures) * min(1, len(group.members) / FULL_EVIDENCE) * factor,
    )


def latest_feedback(
    feedback: Iterable[Feedback], identity: tuple[str, str | None]
) -> Feedback | None:
    return max(
        (f for f in feedback if (f.kind, f.subject) == identity), key=lambda f: f.at, default=None
    )


# without the report the reaction was given on there's nothing to compare, so it stays hidden
def changed_since(
    history: Sequence[PastReport], day: date, identity: tuple[str, str | None], trade_ids: list[int]
) -> bool:
    for report in history:
        if report.day == day:
            for past in report.behaviors:
                if (past.kind, past.subject) == identity:
                    return sorted(past.trade_ids) != sorted(trade_ids)
    return False


def previous(history: Sequence[PastReport], identity: tuple[str, str | None]) -> Previous | None:
    for report in sorted(history, key=lambda r: r.day, reverse=True):
        for past in report.behaviors:
            if (past.kind, past.subject) == identity:
                return Previous(day=report.day, figures=past.figures)
    return None


# highest score first; ties keep catalog order. at most two per kind
def rank(found: list[FoundBehavior]) -> list[FoundBehavior]:
    kept: list[FoundBehavior] = []
    for candidate in sorted(found, key=lambda b: -b.score):
        if sum(b.kind == candidate.kind for b in kept) < MAX_PER_KIND:
            kept.append(candidate)
    return kept[:MAX_BEHAVIORS]


def count(key: str, value: float) -> Figure:
    return Figure(key=key, value=value, unit="count")


def money(key: str, value: float) -> Figure:
    return Figure(key=key, value=round(value, 2), unit="money")


def fraction(key: str, value: float) -> Figure:
    return Figure(key=key, value=round(value, 4), unit="fraction")


def days(key: str, value: float) -> Figure:
    return Figure(key=key, value=value, unit="days")
