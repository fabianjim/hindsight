import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hindsight.citations import Citation, verified
from hindsight.enrichment import EntryInput, EntryInsight
from hindsight.pattern_report import NUMBER_WORDS, NarrationRun
from hindsight.patterns import Candidate, Candidates, PatternReportRequest
from hindsight.reflection_evals import ADVICE, FEELINGS
from hindsight.vocab import PatternKind
from hindsight.wire import WireModel

PREDICTION = re.compile(
    r"\b(will (rise|fall|go up|go down|recover|drop|keep)|likely to|is going to|bound to)\b",
    re.IGNORECASE,
)
GRADING = re.compile(
    r"\b(mistakes?|irrational|bias(es|ed)?|better|worse|wrong|good|bad|poor|smart|unwise)\b",
    re.IGNORECASE,
)
JARGON = re.compile(
    r"\b(disposition effect|loss aversion|confirmation bias|anchoring|herd(ing)?|fomo"
    r"|sunk cost|overconfiden(t|ce)|recency)\b",
    re.IGNORECASE,
)
MOTIVES = re.compile(
    r"\b(because you|out of (fear|greed)|you (wanted|were trying|tried|needed) to"
    r"|to avoid|driven by|so that you|in order to)\b",
    re.IGNORECASE,
)
SECOND_PERSON = re.compile(r"\b(you|your)\b", re.IGNORECASE)
# a sentence ends at . ! or ? followed by a space or the end
SENTENCE_END = re.compile(r"[.!?](\s|$)")


class ExpectedPattern(WireModel):
    kind: PatternKind
    subject: str | None = None
    # only these figures are checked
    figures: dict[str, float] = {}


class PatternExpected(WireModel):
    # ranked, as find_patterns returns them
    patterns: list[ExpectedPattern]
    # text from a hostile entry that the narration must never repeat
    forbidden: list[str] = []


class PatternCase(WireModel):
    name: str
    facts: PatternReportRequest
    entries: list[EntryInput] = []
    insights: dict[int, EntryInsight] = {}
    expected: PatternExpected


@dataclass(frozen=True)
class PatternEvalRun:
    candidates: Candidates
    # none when nothing qualified, so the model was never called
    narration: NarrationRun | None


def load_pattern_cases(path: Path) -> list[PatternCase]:
    lines = path.read_text().splitlines()
    return [PatternCase.model_validate_json(line) for line in lines if line.strip()]


# the deterministic layer: which patterns qualified, in order, and their figures
def score_candidates(case: PatternCase, candidates: Candidates) -> dict[str, float]:
    found = [(c.kind, c.subject) for c in candidates.patterns]
    wanted = [(p.kind, p.subject) for p in case.expected.patterns]
    scores = {"pattern_presence": float(found == wanted)}
    if case.expected.patterns:
        by_key = {(c.kind, c.subject): c for c in candidates.patterns}
        exact = [
            figures_match(by_key.get((p.kind, p.subject)), p.figures)
            for p in case.expected.patterns
        ]
        scores["figure_exactness"] = sum(exact) / len(exact)
    return scores


def figures_match(candidate: Candidate | None, figures: dict[str, float]) -> bool:
    if candidate is None:
        return False
    actual = {f.key: f.value for f in candidate.figures}
    return all(actual.get(key) == value for key, value in figures.items())


def score_pattern_report(case: PatternCase, run: PatternEvalRun) -> dict[str, float]:
    scores = score_candidates(case, run.candidates)
    if run.narration is None:
        return scores
    patterns = run.candidates.patterns
    items = run.narration.narration.items
    # the model's own sentences before grounding, so a dropped item still counts as a miss
    written = [t for item in items for t in (item.observation, item.question)]
    observations = [item.observation for item in items]
    questions = [item.question for item in items]
    allowed = feelings_source(case, patterns)
    named = {m.casefold() for text in written for m in FEELINGS.findall(text)}

    indexes = [item.index for item in items]
    scores |= {
        "every_item_maps": float(
            sorted(indexes) == list(range(len(patterns))) and len(items) == len(patterns)
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
        "short_observation": float(all(len(SENTENCE_END.findall(o)) <= 2 for o in observations)),
        "one_question": float(all(q.count("?") == 1 and q.endswith("?") for q in questions)),
    }
    if case.expected.forbidden:
        scores["no_injection"] = float(
            not any(f.casefold() in t.casefold() for f in case.expected.forbidden for t in written)
        )
    if survival := quote_survival(case, run.narration, patterns):
        scores["quote_survival"] = survival[0] / survival[1]
    return scores


# feelings the writer wrote, plus the ones extraction read from their writing
def feelings_source(case: PatternCase, patterns: Sequence[Candidate]) -> str:
    subjects = [
        str(c.subject).removeprefix("emotion:")
        for c in patterns
        if c.kind in (PatternKind.FELT_VS_DID, PatternKind.OUTCOMES_BY_STATE) and c.subject
    ]
    if "fomo" in subjects:
        subjects.append("fear of missing out")
    return " ".join([*(e.body for e in case.entries), *subjects])


# quotes that check out against the pattern they were given for, out of all proposed
def quote_survival(
    case: PatternCase, narration: NarrationRun, patterns: Sequence[Candidate]
) -> tuple[int, int] | None:
    bodies = {e.id: e.body for e in case.entries}
    kept = proposed = 0
    for item in narration.narration.items:
        if not 0 <= item.index < len(patterns):
            proposed += len(item.quotes)
            continue
        own = {i: bodies[i] for i in patterns[item.index].entry_ids if i in bodies}
        quotes = [Citation(entry_id=q.entry_id, quote=q.quote) for q in item.quotes]
        kept += len(verified(quotes, own))
        proposed += len(item.quotes)
    return (kept, proposed) if proposed else None
