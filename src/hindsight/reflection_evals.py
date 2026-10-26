import re
from pathlib import Path

from hindsight.enrichment import EntryInput
from hindsight.reflection import ReflectionRun, Since
from hindsight.wire import WireModel

# advice in any form; asking about their own plans ("what would make you sell?") is allowed
ADVICE = re.compile(
    r"\b(you should|should you|you might want to|consider (buying|selling|trimming|adding)"
    r"|(good|right) time to|i recommend|recommend)\b",
    re.IGNORECASE,
)
# feelings the output may name only when the writer did
FEELINGS = re.compile(
    r"\b(fomo|fear of missing out|greed|greedy|panic|panicked|anxious|anxiety|euphoria"
    r"|euphoric|regret|regretted)\b",
    re.IGNORECASE,
)


class ReflectionExpected(WireModel):
    has_recap: bool
    # null when either answer is reasonable, so it isn't scored
    has_questions: bool | None


class ReflectionCase(WireModel):
    name: str
    entry: EntryInput
    later: list[EntryInput] = []
    since: Since = Since()
    expected: ReflectionExpected


def load_reflection_cases(path: Path) -> list[ReflectionCase]:
    lines = path.read_text().splitlines()
    return [ReflectionCase.model_validate_json(line) for line in lines if line.strip()]


def score_reflection(case: ReflectionCase, run: ReflectionRun) -> dict[str, float]:
    aid = run.aid
    # quotes are the writer's words, so only the model's own sentences are checked
    written = [claim.summary for claim in aid.recap] + aid.questions
    source_text = " ".join(e.body for e in [case.entry, *case.later])
    named = {m.casefold() for text in written for m in FEELINGS.findall(text)}
    scores = {
        "recap_presence": float(bool(aid.recap) == case.expected.has_recap),
        "no_numbers": float(not any(re.search(r"\d", text) for text in written)),
        "no_advice": float(not any(ADVICE.search(text) for text in written)),
        "feelings_from_writer": float(
            # a prefix, so the writer's "regretting" covers the output's "regret"
            all(re.search(rf"\b{re.escape(n)}", source_text, re.IGNORECASE) for n in named)
        ),
        "questions_are_questions": float(all(q.endswith("?") for q in aid.questions)),
    }
    if case.expected.has_questions is not None:
        scores["question_presence"] = float(bool(aid.questions) == case.expected.has_questions)
    # recap points the model proposed that survived the quote check
    if run.draft.recap:
        scores["citation_survival"] = len(aid.recap) / len(run.draft.recap)
    return scores
