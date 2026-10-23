from pathlib import Path
from typing import Literal

from hindsight.enrichment import Enrichment, EntryInput
from hindsight.vocab import Emotion, Horizon, IdeaSource
from hindsight.wire import WireModel


class Expected(WireModel):
    has_thesis: bool
    has_exit: bool
    horizon: Horizon | None
    conviction: Literal[1, 2, 3, 4, 5] | None
    idea_source: IdeaSource | None
    emotions: list[Emotion]
    tickers: list[str]


class EvalCase(WireModel):
    name: str
    entry: EntryInput
    expected: Expected


def load_cases(path: Path) -> list[EvalCase]:
    lines = path.read_text().splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]


def overlap(found: set[str], wanted: set[str]) -> float:
    return 1.0 if not found and not wanted else len(found & wanted) / len(found | wanted)


def score(expected: Expected, result: Enrichment) -> dict[str, float]:
    insight = result.insight
    scores = {
        "thesis_presence": float((insight.thesis is not None) == expected.has_thesis),
        "exit_presence": float((insight.exit_condition is not None) == expected.has_exit),
        "horizon": float(insight.horizon == expected.horizon),
        "idea_source": float(insight.idea_source == expected.idea_source),
        "conviction": float(insight.conviction == expected.conviction),
        "conviction_within_1": float(
            insight.conviction == expected.conviction
            if insight.conviction is None or expected.conviction is None
            else abs(insight.conviction - expected.conviction) <= 1
        ),
        "emotions": overlap(set(insight.emotions), set(expected.emotions)),
        "tickers": overlap(set(insight.tickers), set(expected.tickers)),
    }
    # claims the model made that survived the quote check; the clearest signal of invention
    raw = result.extraction
    proposed = [c for c in (raw.thesis, raw.exit_condition) if c] if raw else []
    if proposed:
        kept = [c for c in (insight.thesis, insight.exit_condition) if c]
        scores["citation_survival"] = len(kept) / len(proposed)
    return scores


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for scores in results for name in scores})
    return {
        name: sum(values) / len(values)
        for name in names
        if (values := [scores[name] for scores in results if name in scores])
    }
