import argparse
import asyncio
from pathlib import Path
from typing import Any

from langfuse import Evaluation, Langfuse
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from hindsight.behavior_evals import (
    BehaviorCase,
    BehaviorEvalRun,
    findings_for,
    load_behavior_cases,
    score_behavior_report,
    score_findings,
)
from hindsight.behavior_report import PROMPT_VERSION, BehaviorNarrator, allowance_for, render
from hindsight.budget import BudgetLedger, Limits
from hindsight.config import get_settings
from hindsight.evals import summarize
from hindsight.llm import get_chat_model
from hindsight.pricing import price_of
from hindsight.tracing import capture_content, init_tracing

DATASET = Path(__file__).parent.parent / "evals" / "behavior_report.jsonl"
# synthetic entries belong to no vestry user
EVAL_USER_ID = 0


async def run_case(narrator: BehaviorNarrator, case: BehaviorCase) -> BehaviorEvalRun:
    findings = findings_for(case)
    if not findings.behaviors:
        return BehaviorEvalRun(findings=findings, narration=None)
    entries = {e.id: e for e in case.entries}
    narration = await narrator.run(findings.behaviors, entries, EVAL_USER_ID)
    return BehaviorEvalRun(findings=findings, narration=narration)


# scores alone can't show tone, so every answer is printed for a read-through
def show(case: BehaviorCase, run: BehaviorEvalRun) -> None:
    print(f"  {case.name}")
    if run.narration is None:
        print("    no behaviors, no model call")
        return
    for item in run.narration.narration.items:
        print(f"    [{item.index}] {item.observation}")
        print(f"        q: {item.question}")
        for quote in item.quotes:
            print(f"        quote {quote.entry_id}: {quote.quote}")
    kept = [b.kind.value for b in run.narration.behaviors]
    print(f"    kept: {', '.join(kept) or 'none'}")


async def run_locally(cases: list[BehaviorCase], narrator: BehaviorNarrator) -> dict[str, float]:
    results: list[dict[str, float]] = []
    for case in cases:
        try:
            run = await run_case(narrator, case)
        except Exception as error:
            print(f"  {case.name}: failed ({type(error).__name__})")
            continue
        scores = score_behavior_report(case, run)
        show(case, run)
        misses = [name for name, value in scores.items() if value < 1]
        print(f"    {'ok' if not misses else 'missed ' + ', '.join(misses)}")
        results.append(scores)
    summary = summarize(results)
    print(f"\n{len(results)}/{len(cases)} cases ran")
    for name, value in summary.items():
        print(f"  {name:24} {value:.2f}")
    return summary


def run_in_langfuse(
    langfuse: Langfuse, cases: list[BehaviorCase], narrator: BehaviorNarrator
) -> None:
    by_name = {case.name: case for case in cases}

    async def task(*, item: Any, **_: Any) -> BehaviorEvalRun:
        metadata = item["metadata"] if isinstance(item, dict) else item.metadata
        case = by_name[metadata["name"]]
        run = await run_case(narrator, case)
        show(case, run)
        return run

    def evaluate(*, output: Any, metadata: Any, **_: Any) -> list[Evaluation]:
        if not isinstance(output, BehaviorEvalRun):
            return []
        scores = score_behavior_report(by_name[metadata["name"]], output)
        return [Evaluation(name=n, value=v) for n, v in scores.items()]

    settings = get_settings()
    result = langfuse.run_experiment(
        name="behavior-report",
        run_name=f"{PROMPT_VERSION} {settings.chat_model}",
        data=[
            {
                "input": case.model_dump(mode="json", by_alias=True, exclude={"expected"}),
                "expected_output": case.expected.model_dump(mode="json", by_alias=True),
                "metadata": {"name": case.name},
            }
            for case in cases
        ],
        task=task,
        evaluators=[evaluate],
        max_concurrency=4,
    )
    print(result.format())


# the deterministic layer is free, so the dry run checks it and shows what would be sent
def plan(cases: list[BehaviorCase]) -> int:
    misses = 0
    for case in cases:
        findings = findings_for(case)
        scores = score_findings(case, findings)
        kinds = ", ".join(f"{b.kind.value}:{b.subject}" for b in findings.behaviors)
        missed = [name for name, value in scores.items() if value < 1]
        misses += bool(missed)
        status = "ok" if not missed else "missed " + ", ".join(missed)
        print(f"  {case.name}: {kinds or 'no behaviors'} ({status})")
    return misses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score behavior reports.")
    parser.add_argument("--paid", action="store_true", help="call the real model (costs money)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_behavior_cases(args.dataset)
    price = price_of(settings.chat_model)
    narrated = [(c, findings_for(c)) for c in cases]
    narrated = [(c, found) for c, found in narrated if found.behaviors]
    # worst case if every call used its whole reservation; typical assumes ~2.5k in, 500 out
    worst = sum(
        allowance_for(price, render(found.behaviors, {e.id: e for e in c.entries}))
        for c, found in narrated
    )
    typical = len(narrated) * price.cost(2_500, 500)
    print(
        f"{len(cases)} cases, {len(narrated)} model calls, {settings.chat_model}, {PROMPT_VERSION}"
    )
    misses = plan(cases)
    print(f"cost: about ${typical / 1_000_000:.3f}, at most ${worst / 1_000_000:.2f}")
    if misses:
        print(f"{misses} cases miss their expected behaviors; fix those before paying")
        return 1
    if not args.paid:
        print("dry run: pass --paid to call the model")
        return 0

    # no pool, so no connection outlives the event loop that opened it
    engine = create_async_engine(settings.database_url.get_secret_value(), poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    ledger = BudgetLedger(sessions, Limits.from_settings(settings))
    narrator = BehaviorNarrator(ledger, get_chat_model(), settings.chat_model)
    langfuse = init_tracing(settings)
    # synthetic data, so traces may keep content
    with capture_content():
        if langfuse:
            run_in_langfuse(langfuse, cases, narrator)
            langfuse.shutdown()
        else:
            asyncio.run(run_locally(cases, narrator))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
