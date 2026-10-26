import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetLedger, Limits
from hindsight.citations import Citation
from hindsight.enrichment import Claim
from hindsight.reflection import Draft, ReflectionAidGenerator, ReflectionRun
from hindsight.reflection_aids import ReflectionAid
from hindsight.reflection_evals import (
    ReflectionCase,
    ReflectionExpected,
    load_reflection_cases,
    score_reflection,
)
from scripts.eval_reflection_aid import DATASET, main, run_locally
from tests.fakes import FakeToolModel
from tests.test_reflection import ENTRY, LATER, MODEL

Sessions = async_sessionmaker[AsyncSession]
CASE = ReflectionCase(
    name="case",
    entry=ENTRY,
    later=[LATER],
    expected=ReflectionExpected(has_recap=True, has_questions=True),
)
POINT = {"summary": "You later called the dip overdone", "entry_id": 52, "quotes": ["overdone"]}
CLAIM = Claim(
    summary="You later called the dip overdone",
    citations=[Citation(entry_id=52, quote="overdone")],
)


def run(
    recap: list[Claim], questions: list[str], proposed: list[dict[str, Any]] | None = None
) -> ReflectionRun:
    draft = Draft.model_validate({"recap": proposed or [], "questions": questions})
    return ReflectionRun(draft=draft, aid=ReflectionAid(recap=recap, questions=questions))


def test_dataset_is_well_formed() -> None:
    cases = load_reflection_cases(DATASET)

    assert len(cases) >= 8
    assert len({c.name for c in cases}) == len(cases)
    ids = [e.id for c in cases for e in [c.entry, *c.later]]
    assert len(set(ids)) == len(ids)
    for case in cases:
        assert case.entry.body.strip(), case.name
        for later in case.later:
            assert later.timestamp > case.entry.timestamp, case.name
            assert later.ticker == case.entry.ticker, case.name
        # a recap needs later entries to come from
        assert case.later or not case.expected.has_recap, case.name


def test_a_good_answer_scores_one_everywhere() -> None:
    scores = score_reflection(
        CASE, run([CLAIM], ["What made it look like a dip to you?"], proposed=[POINT])
    )

    assert set(scores.values()) == {1.0}
    assert {"question_presence", "citation_survival"} <= set(scores)


@pytest.mark.parametrize(
    ("question", "metric"),
    [
        ("It is up 20% since. How does that feel?", "no_numbers"),
        ("Should you consider selling some now?", "no_advice"),
        ("Was this FOMO at the time?", "feelings_from_writer"),
        ("Tell me what you were thinking", "questions_are_questions"),
    ],
)
def test_scores_each_kind_of_miss(question: str, metric: str) -> None:
    scores = score_reflection(CASE, run([CLAIM], [question], proposed=[POINT]))

    assert scores[metric] == 0
    assert sum(v < 1 for v in scores.values()) == 1


def test_asking_about_their_own_exit_plan_is_not_advice() -> None:
    scores = score_reflection(CASE, run([CLAIM], ["What would make you sell?"], proposed=[POINT]))

    assert scores["no_advice"] == 1


def test_a_feeling_the_writer_named_may_be_mentioned() -> None:
    case = CASE.model_copy(update={"entry": ENTRY.model_copy(update={"body": "Regretting it."})})

    scores = score_reflection(case, run([], ["How does the regret look now?"]))

    assert scores["feelings_from_writer"] == 1


def test_scores_presence_and_dropped_quotes() -> None:
    scores = score_reflection(CASE, run([], [], proposed=[POINT, POINT]))

    assert (scores["recap_presence"], scores["question_presence"]) == (0, 0)
    assert scores["citation_survival"] == 0


def test_skips_what_does_not_apply() -> None:
    case = CASE.model_copy(
        update={"expected": ReflectionExpected(has_recap=False, has_questions=None)}
    )

    scores = score_reflection(case, run([], []))

    assert "question_presence" not in scores
    assert "citation_survival" not in scores
    assert set(scores.values()) == {1.0}


def test_dry_run_never_builds_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden() -> None:
        raise AssertionError("a dry run must not build a model")

    monkeypatch.setattr("scripts.eval_reflection_aid.get_chat_model", forbidden)

    assert main([]) == 0
    assert "dry run" in capsys.readouterr().out


@pytest.mark.anyio
async def test_runs_the_experiment_end_to_end(sessions: Sessions) -> None:
    cases = load_reflection_cases(DATASET)[:2]
    replies = [
        AIMessage(
            content=json.dumps({"recap": [], "questions": []}),
            usage_metadata={"input_tokens": 900, "output_tokens": 50, "total_tokens": 950},
        )
        for _ in cases
    ]
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 100))
    generator = ReflectionAidGenerator(ledger, FakeToolModel(messages=iter(replies)), MODEL)

    summary = await run_locally(cases, generator)

    # the first two cases are placeholders, so an empty answer is right
    assert summary["question_presence"] == 1.0
    assert summary["no_numbers"] == 1.0
