from datetime import date, datetime
from typing import Literal

from pydantic import Field

from hindsight.enrichment import Ticker
from hindsight.wire import WireModel

MAX_PATTERNS = 5
MAX_OUTCOMES = 500
MAX_TRADES = 1000
MAX_HOLDING_PERIODS = 1000
MAX_AUTO_ENTRY_IDS = 1000


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
