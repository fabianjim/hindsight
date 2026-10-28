from datetime import date, datetime
from typing import Literal

from pydantic import Field

from hindsight.citations import Citation
from hindsight.vocab import PatternKind
from hindsight.wire import WireModel

MAX_PATTERNS = 5


# vestry words the figures; vestry ignores keys it doesn't know
class Figure(WireModel):
    key: str
    value: float
    unit: Literal["count", "days"]


class Pattern(WireModel):
    kind: PatternKind
    subject: str | None
    figures: list[Figure]
    # no digits; the figures carry the numbers
    observation: str
    question: str
    # at least one, except for silent_trades, whose subject is a missing entry
    citations: list[Citation]


# a kind that didn't qualify and how many more examples it needs
class Need(WireModel):
    kind: PatternKind
    more: int = Field(ge=1)


class PatternReport(WireModel):
    generated_on: date
    as_of: date
    # the newest entry with extracted fields when the report was built
    entries_up_to: datetime | None
    # false only for an empty report that wasn't stored
    saved: bool
    patterns: list[Pattern] = Field(max_length=MAX_PATTERNS)
    needs: list[Need]
