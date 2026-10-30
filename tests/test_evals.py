import json
import re
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.budget import BudgetLedger, Limits
from hindsight.citations import Citation
from hindsight.enrichment import Claim, Enrichment, EntryEnricher, EntryInsight, Extraction
from hindsight.evals import Expected, load_cases, score, summarize
from hindsight.vocab import Emotion, ReasoningBasis
from scripts.eval_entry_insight import DATASET, main, run_locally
from tests.fakes import FakeToolModel

Sessions = async_sessionmaker[AsyncSession]
MODEL = "openai:gpt-4.1-mini-2025-04-14"
CLAIM = Claim(summary="Demand lasts", citations=[Citation(entry_id=1, quote="demand lasts")])
EXPECTED = Expected(
    has_thesis=True,
    has_exit=False,
    horizon=None,
    conviction=4,
    idea_source=None,
    emotions=[Emotion.FOMO],
    tickers=["NVDA"],
    reasoning_basis=ReasoningBasis.PRICE_ACTION,
    thesis_outcome=None,
)
NOTHING: dict[str, Any] = {
    "thesis": None,
    "exit_condition": None,
    "horizon": None,
    "conviction": None,
    "emotions": [],
    "idea_source": None,
    "tickers": [],
    "themes": [],
    "reasoning_basis": None,
    "thesis_outcome": None,
}


def result(insight: dict[str, Any], raw: dict[str, Any] | None = None) -> Enrichment:
    return Enrichment(
        entry_id=1,
        insight=EntryInsight.model_validate({**NOTHING, **insight}),
        input_hash="a" * 64,
        model=MODEL,
        prompt_version="v",
        extraction=Extraction.model_validate({**NOTHING, **raw}) if raw is not None else None,
    )


def test_dataset_is_well_formed() -> None:
    cases = load_cases(DATASET)

    assert len(cases) >= 20
    assert len({c.name for c in cases}) == len(cases)
    assert len({c.entry.id for c in cases}) == len(cases)
    for case in cases:
        assert case.entry.body.strip(), case.name
        for ticker in case.expected.tickers:
            assert re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker), case.name


def test_a_perfect_answer_scores_one_everywhere() -> None:
    scores = score(
        EXPECTED,
        result(
            {
                "thesis": CLAIM,
                "conviction": 4,
                "emotions": ["fomo"],
                "tickers": ["NVDA"],
                "reasoning_basis": "price_action",
            },
            raw={"thesis": {"summary": "Demand lasts", "quotes": ["demand lasts"]}},
        ),
    )

    assert set(scores.values()) == {1.0}
    assert "citation_survival" in scores


def test_scores_each_kind_of_miss() -> None:
    scores = score(
        EXPECTED,
        result(
            {
                "conviction": 3,
                "emotions": ["fomo", "anxious"],
                "tickers": [],
                "reasoning_basis": "commentary",
                "thesis_outcome": "played_out",
            },
            raw={
                "thesis": {"summary": "Invented", "quotes": ["not in the entry"]},
                "exit_condition": {"summary": "Invented", "quotes": ["nor this"]},
            },
        ),
    )

    assert scores["thesis_presence"] == 0
    assert scores["exit_presence"] == 1
    assert (scores["conviction"], scores["conviction_within_1"]) == (0, 1)
    assert scores["emotions"] == 0.5
    assert scores["tickers"] == 0
    assert scores["citation_survival"] == 0
    assert (scores["reasoning_basis"], scores["thesis_outcome"]) == (0, 0)


def test_citation_survival_is_skipped_when_nothing_was_claimed() -> None:
    assert "citation_survival" not in score(EXPECTED, result({}, raw={}))


def test_summary_averages_only_where_a_metric_applies() -> None:
    summary = summarize([{"a": 1.0, "b": 0.0}, {"a": 0.0}])

    assert summary == {"a": 0.5, "b": 0.0}


def test_dry_run_never_builds_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden() -> None:
        raise AssertionError("a dry run must not build a model")

    monkeypatch.setattr("scripts.eval_entry_insight.get_chat_model", forbidden)

    assert main([]) == 0
    assert "dry run" in capsys.readouterr().out


@pytest.mark.anyio
async def test_runs_the_experiment_end_to_end(sessions: Sessions) -> None:
    cases = load_cases(DATASET)[:2]
    replies = [
        AIMessage(
            content=json.dumps({**NOTHING, "tickers": c.expected.tickers}),
            usage_metadata={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550},
        )
        for c in cases
    ]
    ledger = BudgetLedger(sessions, Limits(-1, 1_000_000, 100))
    enricher = EntryEnricher(ledger, FakeToolModel(messages=iter(replies)), MODEL)

    summary = await run_locally(cases, enricher)

    assert summary["tickers"] == 1.0
    assert summary["thesis_presence"] < 1.0
