import argparse
import asyncio
from pathlib import Path
from typing import Any

from langfuse import Evaluation, Langfuse
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from hindsight.budget import BudgetLedger, Limits
from hindsight.config import get_settings
from hindsight.enrichment import ALLOWANCE_MICROS, PROMPT_VERSION, Enrichment, EntryEnricher
from hindsight.evals import EvalCase, load_cases, score, summarize
from hindsight.llm import get_chat_model
from hindsight.pricing import price_of
from hindsight.tracing import capture_content, init_tracing

DATASET = Path(__file__).parent.parent / "evals" / "entry_insight.jsonl"
# synthetic entries belong to no vestry user
EVAL_USER_ID = 0


async def run_locally(cases: list[EvalCase], enricher: EntryEnricher) -> dict[str, float]:
    results: list[dict[str, float]] = []
    for case in cases:
        try:
            scores = score(case.expected, await enricher.enrich(case.entry, EVAL_USER_ID))
        except Exception as error:
            print(f"  {case.name}: failed ({type(error).__name__})")
            continue
        misses = [name for name, value in scores.items() if value < 1]
        print(f"  {case.name}: {'ok' if not misses else 'missed ' + ', '.join(misses)}")
        results.append(scores)
    summary = summarize(results)
    print(f"\n{len(results)}/{len(cases)} cases ran")
    for name, value in summary.items():
        print(f"  {name:20} {value:.2f}")
    return summary


def run_in_langfuse(langfuse: Langfuse, cases: list[EvalCase], enricher: EntryEnricher) -> None:
    by_name = {case.name: case for case in cases}

    async def task(*, item: Any, **_: Any) -> Enrichment:
        metadata = item["metadata"] if isinstance(item, dict) else item.metadata
        return await enricher.enrich(by_name[metadata["name"]].entry, EVAL_USER_ID)

    def evaluate(*, output: Any, metadata: Any, **_: Any) -> list[Evaluation]:
        if not isinstance(output, Enrichment):
            return []
        expected = by_name[metadata["name"]].expected
        return [Evaluation(name=n, value=v) for n, v in score(expected, output).items()]

    settings = get_settings()
    result = langfuse.run_experiment(
        name="entry-insight",
        run_name=f"{PROMPT_VERSION} {settings.chat_model}",
        data=[
            {
                "input": case.entry.model_dump(mode="json", by_alias=True),
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
    parser = argparse.ArgumentParser(description="Score entry insight extraction.")
    parser.add_argument("--paid", action="store_true", help="call the real model (costs money)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_cases(args.dataset)
    # worst case if every call used its whole reservation; typical assumes ~1k in, 200 out
    worst = len(cases) * ALLOWANCE_MICROS / 1_000_000
    typical = len(cases) * price_of(settings.chat_model).cost(1_000, 200) / 1_000_000
    print(f"{len(cases)} cases, {settings.chat_model}, {PROMPT_VERSION}")
    print(f"cost: about ${typical:.3f}, at most ${worst:.2f}")
    if not args.paid:
        print("dry run: pass --paid to call the model")
        return 0

    # no pool, so no connection outlives the event loop that opened it
    engine = create_async_engine(settings.database_url.get_secret_value(), poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    ledger = BudgetLedger(sessions, Limits.from_settings(settings))
    enricher = EntryEnricher(ledger, get_chat_model(), settings.chat_model)
    langfuse = init_tracing(settings)
    # synthetic data, so traces may keep content
    with capture_content():
        if langfuse:
            run_in_langfuse(langfuse, cases, enricher)
            langfuse.shutdown()
        else:
            asyncio.run(run_locally(cases, enricher))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
