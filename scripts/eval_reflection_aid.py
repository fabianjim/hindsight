import argparse
import asyncio
from pathlib import Path
from typing import Any

from langfuse import Evaluation, Langfuse
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from hindsight.budget import BudgetLedger, Limits, budget_day, utcnow
from hindsight.config import get_settings
from hindsight.evals import summarize
from hindsight.llm import get_chat_model
from hindsight.pricing import price_of
from hindsight.reflection import (
    PROMPT_VERSION,
    ReflectionAidGenerator,
    ReflectionRun,
    allowance_for,
    newest_later,
    render,
)
from hindsight.reflection_evals import ReflectionCase, load_reflection_cases, score_reflection
from hindsight.tracing import capture_content, init_tracing

DATASET = Path(__file__).parent.parent / "evals" / "reflection_aid.jsonl"
# synthetic entries belong to no vestry user
EVAL_USER_ID = 0


async def run_case(generator: ReflectionAidGenerator, case: ReflectionCase) -> ReflectionRun:
    return await generator.run(case.entry, case.later, case.since, EVAL_USER_ID)


# scores alone can't show tone, so every answer is printed for a read-through
def show(case: ReflectionCase, run: ReflectionRun) -> None:
    print(f"  {case.name}")
    for claim in run.aid.recap:
        print(f"    recap: {claim.summary}")
    for question in run.aid.questions:
        print(f"    q: {question}")


async def run_locally(
    cases: list[ReflectionCase], generator: ReflectionAidGenerator
) -> dict[str, float]:
    results: list[dict[str, float]] = []
    for case in cases:
        try:
            run = await run_case(generator, case)
        except Exception as error:
            print(f"  {case.name}: failed ({type(error).__name__})")
            continue
        scores = score_reflection(case, run)
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
    langfuse: Langfuse, cases: list[ReflectionCase], generator: ReflectionAidGenerator
) -> None:
    by_name = {case.name: case for case in cases}

    async def task(*, item: Any, **_: Any) -> ReflectionRun:
        metadata = item["metadata"] if isinstance(item, dict) else item.metadata
        case = by_name[metadata["name"]]
        run = await run_case(generator, case)
        show(case, run)
        return run

    def evaluate(*, output: Any, metadata: Any, **_: Any) -> list[Evaluation]:
        if not isinstance(output, ReflectionRun):
            return []
        scores = score_reflection(by_name[metadata["name"]], output)
        return [Evaluation(name=n, value=v) for n, v in scores.items()]

    settings = get_settings()
    result = langfuse.run_experiment(
        name="reflection-aid",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score reflection aids.")
    parser.add_argument("--paid", action="store_true", help="call the real model (costs money)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_reflection_cases(args.dataset)
    price = price_of(settings.chat_model)
    today = budget_day(utcnow())
    # worst case if every call used its whole reservation; typical assumes ~1.5k in, 250 out
    worst = sum(
        allowance_for(price, render(c.entry, newest_later(c.entry, c.later), c.since, today))
        for c in cases
    )
    typical = len(cases) * price.cost(1_500, 250)
    print(f"{len(cases)} cases, {settings.chat_model}, {PROMPT_VERSION}")
    print(f"cost: about ${typical / 1_000_000:.3f}, at most ${worst / 1_000_000:.2f}")
    if not args.paid:
        print("dry run: pass --paid to call the model")
        return 0

    # no pool, so no connection outlives the event loop that opened it
    engine = create_async_engine(settings.database_url.get_secret_value(), poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    ledger = BudgetLedger(sessions, Limits.from_settings(settings))
    generator = ReflectionAidGenerator(ledger, get_chat_model(), settings.chat_model)
    langfuse = init_tracing(settings)
    # synthetic data, so traces may keep content
    with capture_content():
        if langfuse:
            run_in_langfuse(langfuse, cases, generator)
            langfuse.shutdown()
        else:
            asyncio.run(run_locally(cases, generator))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
