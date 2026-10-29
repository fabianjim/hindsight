from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from hindsight.enrichment import Ticker
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
