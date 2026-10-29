import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from hindsight.behavior_report import NUMBER_WORDS, NarrationRun, fits_card
from hindsight.behavior_reports import Figure, PastBehavior, PastReport
from hindsight.behaviors import BehaviorReportRequest, Findings, FoundBehavior, find_behaviors
from hindsight.citations import Citation, verified
from hindsight.enrichment import EntryInput, EntryInsight
from hindsight.reflection_evals import ADVICE, FEELINGS
from hindsight.vocab import BehaviorKind
from hindsight.wire import WireModel

PREDICTION = re.compile(
    r"\b(will (rise|fall|go up|go down|recover|drop|keep)|likely to|is going to|bound to)\b",
    re.IGNORECASE,
)
GRADING = re.compile(
    r"\b(mistakes?|irrational|bias(es|ed)?|better|worse|wrong|good|bad|poor|smart|unwise)\b",
    re.IGNORECASE,
)
# the common names are vestry's fixed labels, so the model never uses them
JARGON = re.compile(
    r"\b(disposition effect|loss aversion|confirmation bias|anchoring|herd(ing)?|fomo"
    r"|sunk cost|overconfiden(t|ce)|recency|overtrad(ing|e)|position siz(ing|e)|hype"
    r"|concentration risk)\b",
    re.IGNORECASE,
)
MOTIVES = re.compile(
    r"\b(because you|out of (fear|greed)|you (wanted|were trying|tried|needed) to"
    r"|to avoid|driven by|so that you|in order to)\b",
    re.IGNORECASE,
)
SECOND_PERSON = re.compile(r"\b(you|your)\b", re.IGNORECASE)


class ExpectedBehavior(WireModel):
    kind: BehaviorKind
    subject: str | None = None
    # only these figures are checked
    figures: dict[str, float] = {}
    # figure keys that must not appear, e.g. baseline ones when the baseline is too small
    absent: list[str] = []
    # the day `previous` must come from; checked only when set
    previous_day: date | None = None


class BehaviorExpected(WireModel):
    # ranked, as find_behaviors returns them
    behaviors: list[ExpectedBehavior]
    # text from a hostile entry that the narration must never repeat
    forbidden: list[str] = []


class PastBehaviorCase(WireModel):
    kind: str
    subject: str | None = None
    figures: list[Figure] = []
    trade_ids: list[int]


# an earlier saved report, as the store would hand it to find_behaviors
class PastReportCase(WireModel):
    day: date
    behaviors: list[PastBehaviorCase]


class BehaviorCase(WireModel):
    name: str
    facts: BehaviorReportRequest
    # the user's entries, including behavior-note entries
    entries: list[EntryInput] = []
    insights: dict[int, EntryInsight] = {}
    history: list[PastReportCase] = []
    expected: BehaviorExpected

    def past(self) -> list[PastReport]:
        return [
            PastReport(
                day=r.day,
                behaviors=[
                    PastBehavior(b.kind, b.subject, b.figures, b.trade_ids) for b in r.behaviors
                ],
            )
            for r in sorted(self.history, key=lambda r: r.day, reverse=True)
        ]


@dataclass(frozen=True)
class BehaviorEvalRun:
    findings: Findings
    # none when nothing qualified, so the model was never called
    narration: NarrationRun | None


def load_behavior_cases(path: Path) -> list[BehaviorCase]:
    lines = path.read_text().splitlines()
    return [BehaviorCase.model_validate_json(line) for line in lines if line.strip()]


def findings_for(case: BehaviorCase) -> Findings:
    return find_behaviors(case.facts, case.entries, case.insights, case.past())


# the deterministic layer: which behaviors qualified, in order, their figures and previous
def score_findings(case: BehaviorCase, findings: Findings) -> dict[str, float]:
    found = [(b.kind, b.subject) for b in findings.behaviors]
    wanted = [(b.kind, b.subject) for b in case.expected.behaviors]
    scores = {"behavior_presence": float(found == wanted)}
    if not case.expected.behaviors:
        return scores
    by_key = {(b.kind, b.subject): b for b in findings.behaviors}
    exact = [
        figures_match(by_key.get((b.kind, b.subject)), b.figures, b.absent)
        for b in case.expected.behaviors
    ]
    scores["figure_exactness"] = sum(exact) / len(exact)
    dated = [b for b in case.expected.behaviors if b.previous_day is not None]
    if dated:
        matched = [previous_day(by_key.get((b.kind, b.subject))) == b.previous_day for b in dated]
        scores["previous_matches"] = sum(matched) / len(matched)
    return scores


def figures_match(
    behavior: FoundBehavior | None, figures: dict[str, float], absent: Sequence[str] = ()
) -> bool:
    if behavior is None:
        return False
    actual = {f.key: f.value for f in behavior.figures}
    return all(actual.get(key) == value for key, value in figures.items()) and not any(
        key in actual for key in absent
    )


def previous_day(behavior: FoundBehavior | None) -> date | None:
    return behavior.previous.day if behavior and behavior.previous else None


def score_behavior_report(case: BehaviorCase, run: BehaviorEvalRun) -> dict[str, float]:
    scores = score_findings(case, run.findings)
    if run.narration is None:
        return scores
    behaviors = run.findings.behaviors
    items = run.narration.narration.items
    # the model's own sentences before grounding, so a dropped item still counts as a miss
    written = [t for item in items for t in (item.observation, item.question)]
    observations = [item.observation for item in items]
    questions = [item.question for item in items]
    allowed = feelings_source(case)
    named = {m.casefold() for text in written for m in FEELINGS.findall(text)}

    indexes = [item.index for item in items]
    scores |= {
        "every_item_maps": float(
            sorted(indexes) == list(range(len(behaviors))) and len(items) == len(behaviors)
        ),
        "no_digits": float(not any(re.search(r"\d", t) for t in written)),
        "no_number_words": float(not any(NUMBER_WORDS.search(t) for t in written)),
        "no_advice": float(not any(ADVICE.search(t) or PREDICTION.search(t) for t in written)),
        "no_grading": float(not any(GRADING.search(t) for t in written)),
        "no_jargon": float(not any(JARGON.search(t) for t in written)),
        "no_motives": float(not any(MOTIVES.search(t) for t in written)),
        "feelings_from_writer": float(
            # a prefix, so the writer's "regretting" covers the output's "regret"
            all(re.search(rf"\b{re.escape(n)}", allowed, re.IGNORECASE) for n in named)
        ),
        "second_person": float(all(SECOND_PERSON.search(o) for o in observations)),
        # the same check grounding applies
        "short_observation": float(all(fits_card(o.strip()) for o in observations)),
        "one_question": float(all(q.count("?") == 1 and q.endswith("?") for q in questions)),
        "notes_not_quoted": float(not quoted_notes(run.narration, behaviors)),
    }
    if case.expected.forbidden:
        scores["no_injection"] = float(
            not any(f.casefold() in t.casefold() for f in case.expected.forbidden for t in written)
        )
    if survival := quote_survival(case, run.narration, behaviors):
        scores["quote_survival"] = survival[0] / survival[1]
    return scores


# feelings the writer wrote, plus the ones extraction read from their writing
def feelings_source(case: BehaviorCase) -> str:
    read: list[str] = [e for insight in case.insights.values() for e in insight.emotions]
    if "fomo" in read:
        read.append("fear of missing out")
    return " ".join([*(e.body for e in case.entries), *read])


# quotes the model took from a behavior's notes, which are context only
def quoted_notes(narration: NarrationRun, behaviors: Sequence[FoundBehavior]) -> bool:
    return any(
        q.entry_id in behaviors[item.index].note_entry_ids
        for item in narration.narration.items
        if 0 <= item.index < len(behaviors)
        for q in item.quotes
    )


# quotes that check out against the entries their behavior was shown, out of all proposed
def quote_survival(
    case: BehaviorCase, narration: NarrationRun, behaviors: Sequence[FoundBehavior]
) -> tuple[int, int] | None:
    bodies = {e.id: e.body for e in case.entries}
    kept = proposed = 0
    for item in narration.narration.items:
        proposed += len(item.quotes)
        if not 0 <= item.index < len(behaviors):
            continue
        own = {i: bodies[i] for i in behaviors[item.index].shown_entry_ids if i in bodies}
        quotes = [Citation(entry_id=q.entry_id, quote=q.quote) for q in item.quotes]
        kept += len(verified(quotes, own))
    return (kept, proposed) if proposed else None
