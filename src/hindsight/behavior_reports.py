from datetime import date, datetime
from typing import Literal

from pydantic import Field

from hindsight.citations import Citation
from hindsight.vocab import BehaviorKind
from hindsight.wire import WireModel

MAX_BEHAVIORS = 5
MAX_LISTED = 50


# vestry words the figures and skips keys it doesn't know; fractions show as percents
class Figure(WireModel):
    key: str
    value: float = Field(allow_inf_nan=False)
    unit: Literal["count", "money", "fraction", "days"]


# the same identity in the newest earlier saved report
class Previous(WireModel):
    day: date
    figures: list[Figure]


class Behavior(WireModel):
    kind: BehaviorKind
    subject: str | None
    figures: list[Figure]
    previous: Previous | None
    # no digits or number words; the figures carry the numbers
    observation: str
    question: str
    # at least one, except for undocumented, whose subject is a missing entry
    citations: list[Citation]
    # every decision and linked entry in the group
    trade_ids: list[int]
    entry_ids: list[int]


# a kind (and subject) that didn't qualify and what it waits for
class Need(WireModel):
    kind: BehaviorKind
    subject: str | None
    more: int = Field(ge=1)
    reason: Literal["ENTRIES", "TRADES", "PRICES", "FIELDS"]


class BehaviorReport(WireModel):
    # the New York day the report belongs to
    day: date
    generated_at: datetime
    as_of: date
    # the newest entry with extracted fields when the report was built
    entries_up_to: datetime | None
    # false only for an empty report that wasn't stored
    saved: bool
    behaviors: list[Behavior] = Field(max_length=MAX_BEHAVIORS)
    needs: list[Need]


class SavedBehaviorReport(WireModel):
    report: BehaviorReport | None


class BehaviorSummary(WireModel):
    kind: BehaviorKind
    subject: str | None
    figures: list[Figure]


class BehaviorReportSummary(WireModel):
    day: date
    generated_at: datetime
    behaviors: list[BehaviorSummary]


# once a model call starts the day is used; a crashed attempt reads as FAILED
class Attempt(WireModel):
    day: date
    status: Literal["SAVED", "FAILED"]
    at: datetime


class BehaviorReportList(WireModel):
    # newest first
    reports: list[BehaviorReportSummary]
    has_older: bool
    # the user's newest attempt
    last_attempt: Attempt | None
