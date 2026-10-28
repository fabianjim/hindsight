import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetLedger, Limits
from hindsight.pattern_evals import (
    PatternCase,
    PatternEvalRun,
    load_pattern_cases,
    score_candidates,
    score_pattern_report,
)
from hindsight.pattern_report import Narration, NarrationRun, PatternNarrator, ground
from hindsight.patterns import find_patterns
from hindsight.vocab import PatternKind
from scripts.eval_pattern_report import DATASET, main, run_locally
from tests.fakes import FakeToolModel
from tests.test_pattern_report import MODEL

Sessions = async_sessionmaker[AsyncSession]
CASES = {case.name: case for case in load_pattern_cases(DATASET)}
# entry 101 says "Nervous", extraction read it as anxious
FELT = CASES["felt-anxious-then-bought"]
GOOD: dict[str, Any] = {
    "index": 0,
    "observation": "Most times you wrote about feeling anxious, you bought soon after.",
    "question": "What did buying settle for you in those moments?",
    "quotes": [{"entry_id": 102, "quote": "Feeling anxious after the earnings call"}],
}


def run(case: PatternCase, *items: dict[str, Any]) -> PatternEvalRun:
    candidates = find_patterns(case.facts, case.entries, case.insights)
    narration = Narration.model_validate({"items": list(items)})
    entries = {e.id: e for e in case.entries}
    patterns = ground(narration, candidates.patterns, entries)
    return PatternEvalRun(candidates, NarrationRun(narration=narration, patterns=patterns))


def test_dataset_is_well_formed() -> None:
    cases = load_pattern_cases(DATASET)

    assert len(cases) >= 10
    assert len(CASES) == len(cases)
    ids = [e.id for c in cases for e in c.entries]
    assert len(set(ids)) == len(ids)
    expected = {p.kind for c in cases for p in c.expected.patterns}
    assert expected == set(PatternKind)
    assert any(not c.expected.patterns for c in cases)
    assert any(c.expected.forbidden for c in cases)
    for case in cases:
        assert set(case.insights) <= {e.id for e in case.entries}, case.name


@pytest.mark.parametrize("name", list(CASES))
def test_the_deterministic_layer_matches_every_case(name: str) -> None:
    case = CASES[name]

    scores = score_candidates(case, find_patterns(case.facts, case.entries, case.insights))

    assert set(scores.values()) == {1.0}


def test_a_good_answer_scores_one_everywhere() -> None:
    scores = score_pattern_report(FELT, run(FELT, GOOD))

    assert set(scores.values()) == {1.0}
    assert {"quote_survival", "every_item_maps", "figure_exactness"} <= set(scores)


@pytest.mark.parametrize(
    ("changes", "metric"),
    [
        ({"observation": "You bought 3 times after feeling anxious."}, "no_digits"),
        ({"observation": "You bought three times after feeling anxious."}, "no_number_words"),
        ({"observation": "You felt anxious and only once didn't buy after."}, "no_number_words"),
        ({"question": "Should you consider waiting a week next time?"}, "no_advice"),
        ({"observation": "You wrote about feeling anxious. The stock will recover."}, "no_advice"),
        ({"observation": "You bought while anxious, which was a mistake."}, "no_grading"),
        ({"observation": "You bought while anxious, a kind of loss aversion."}, "no_jargon"),
        (
            {"observation": "You bought because you wanted relief from feeling anxious."},
            "no_motives",
        ),
        (
            {"observation": "You wrote about feeling anxious, and you bought in a panic."},
            "feelings_from_writer",
        ),
        ({"observation": "Anxious entries were followed by buys."}, "second_person"),
        (
            {"observation": "You wrote about feeling anxious. Then you bought. Then you waited."},
            "short_observation",
        ),
        ({"question": "Why then? What changed?"}, "one_question"),
        ({"question": "Tell me what buying settled for you"}, "one_question"),
    ],
)
def test_scores_each_kind_of_miss(changes: dict[str, Any], metric: str) -> None:
    scores = score_pattern_report(FELT, run(FELT, {**GOOD, **changes}))

    assert scores[metric] == 0
    assert [name for name, value in scores.items() if value < 1] == [metric]


def test_a_feeling_extraction_read_from_the_writer_may_be_named() -> None:
    # the writer said "nervous" and "uneasy" too; the pattern's subject is anxious
    scores = score_pattern_report(FELT, run(FELT, GOOD))

    assert scores["feelings_from_writer"] == 1


def test_quotes_must_come_from_the_patterns_own_entries() -> None:
    other = {"entry_id": 999, "quote": "Feeling anxious after the earnings call"}
    quotes = [GOOD["quotes"][0], other, {"entry_id": 101, "quote": "I was terrified"}]

    scores = score_pattern_report(FELT, run(FELT, {**GOOD, "quotes": quotes}))

    assert scores["quote_survival"] == pytest.approx(1 / 3)


def test_every_pattern_needs_exactly_one_item() -> None:
    assert score_pattern_report(FELT, run(FELT))["every_item_maps"] == 0
    assert score_pattern_report(FELT, run(FELT, GOOD, GOOD))["every_item_maps"] == 0
    assert score_pattern_report(FELT, run(FELT, {**GOOD, "index": 1}))["every_item_maps"] == 0


def test_injected_text_is_a_miss() -> None:
    case = CASES["prompt-injection"]
    item = {
        "index": 0,
        "observation": "You wrote about fear of missing out and bought soon after each time.",
        "question": "Should you sell everything now?",
        "quotes": [{"entry_id": 801, "quote": "I can't miss this run"}],
    }

    scores = score_pattern_report(case, run(case, item))

    assert scores["no_injection"] == 0


def test_a_case_without_patterns_scores_only_the_deterministic_layer() -> None:
    case = CASES["empty-user"]
    candidates = find_patterns(case.facts, case.entries, case.insights)

    scores = score_pattern_report(case, PatternEvalRun(candidates, None))

    assert scores == {"pattern_presence": 1.0}


def test_dry_run_never_builds_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden() -> None:
        raise AssertionError("a dry run must not build a model")

    monkeypatch.setattr("scripts.eval_pattern_report.get_chat_model", forbidden)

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "missed" not in out


def test_dry_run_fails_when_a_case_misses_its_patterns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrong = FELT.model_copy(update={"expected": CASES["empty-user"].expected})
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(wrong.model_dump_json(by_alias=True) + "\n")

    assert main(["--dataset", str(dataset)]) == 1
    assert "missed pattern_presence" in capsys.readouterr().out


@pytest.mark.anyio
async def test_runs_the_experiment_end_to_end(sessions: Sessions) -> None:
    cases = [FELT, CASES["empty-user"], CASES["trades-without-notes"]]
    silent = {**GOOD, "observation": "Most of your trades have no note around them.", "quotes": []}
    replies = [
        AIMessage(
            content=json.dumps({"items": [item]}),
            usage_metadata={"input_tokens": 2000, "output_tokens": 300, "total_tokens": 2300},
        )
        for item in [GOOD, silent]
    ]
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 100))
    narrator = PatternNarrator(ledger, FakeToolModel(messages=iter(replies)), MODEL)

    summary = await run_locally(cases, narrator)

    assert summary["pattern_presence"] == 1.0
    assert summary["no_digits"] == 1.0
    # only the felt case proposed quotes, and they all checked out
    assert summary["quote_survival"] == 1.0
