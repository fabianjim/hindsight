import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from hindsight.behavior_report import has_number
from hindsight.budget import BudgetLedger
from hindsight.citations import Citation, verified
from hindsight.enrichment import EntryInput
from hindsight.llm import UserContext
from hindsight.pattern_reports import MAX_PATTERNS, Pattern
from hindsight.patterns import Candidate
from hindsight.pricing import Price, price_of
from hindsight.spend import REQUEST_OVERHEAD_TOKENS, BudgetMiddleware
from hindsight.tracing import trace_callbacks
from hindsight.vocab import PatternKind

PROMPT_VERSION = "pattern-report-v2"
PROMPT = (Path(__file__).parent / "prompts" / "pattern_report.md").read_text()
MAX_OUTPUT_TOKENS = 1200


# openai strict mode requires additionalProperties: false on every object
class NarratedQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: int
    quote: str


class NarratedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    observation: str
    question: str
    quotes: list[NarratedQuote] = Field(max_length=3)


class Narration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[NarratedItem] = Field(max_length=MAX_PATTERNS)


@dataclass(frozen=True)
class NarrationRun:
    narration: Narration
    patterns: list[Pattern]


RESPONSE_FORMAT = ProviderStrategy(Narration, strict=True)
SCHEMA = json.dumps(RESPONSE_FORMAT.schema_spec.json_schema)


class PatternNarrator:
    def __init__(self, ledger: BudgetLedger, model: BaseChatModel, model_name: str) -> None:
        self._ledger = ledger
        self._model_name = model_name
        self._price = price_of(model_name)
        self._agent = create_agent(
            model,
            system_prompt=PROMPT,
            response_format=RESPONSE_FORMAT,
            context_schema=UserContext,
            middleware=[BudgetMiddleware(ledger, model_name, "narrate", MAX_OUTPUT_TOKENS)],
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def narrate(
        self, candidates: Sequence[Candidate], entries: Mapping[int, EntryInput], user_id: int
    ) -> list[Pattern]:
        return (await self.run(candidates, entries, user_id)).patterns

    # the model's narration alongside the grounded patterns, for evals
    async def run(
        self, candidates: Sequence[Candidate], entries: Mapping[int, EntryInput], user_id: int
    ) -> NarrationRun:
        if not candidates:
            raise ValueError("no patterns to narrate")
        if len(candidates) > MAX_PATTERNS:
            raise ValueError("too many patterns to narrate")
        message = render(candidates, entries)
        # a fresh key per attempt: the ledger never replays a key, and a retry must stay possible
        job = await self._ledger.start(secrets.token_hex(32), allowance_for(self._price, message))
        success = False
        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(message)]},
                context=UserContext(user_id=user_id, generation_id=job.id),
                config={"callbacks": trace_callbacks(), "run_name": "pattern-report"},
            )
            narration = result["structured_response"]
            success = True
        finally:
            await self._ledger.finish(job.id, success)
        assert isinstance(narration, Narration)
        return NarrationRun(narration=narration, patterns=ground(narration, candidates, entries))


# the middleware re-encodes the request as json, which at most doubles each byte
def allowance_for(price: Price, message: str) -> int:
    request_bytes = len((PROMPT + message + SCHEMA).encode())
    return price.cost(2 * request_bytes + REQUEST_OVERHEAD_TOKENS, MAX_OUTPUT_TOKENS)


def render(candidates: Sequence[Candidate], entries: Mapping[int, EntryInput]) -> str:
    # json keeps each body a quoted string, so it can't pose as part of the prompt
    fields = {
        "patterns": [
            {
                "index": index,
                "kind": c.kind,
                "subject": c.subject,
                "figures": [
                    {"key": f.key, "value": figure_value(f.value), "unit": f.unit}
                    for f in c.figures
                ],
                "entries": [entry_fields(entries[i]) for i in c.entry_ids if i in entries],
            }
            for index, c in enumerate(candidates)
        ]
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


def figure_value(value: float) -> float | int:
    return int(value) if value.is_integer() else value


def entry_fields(entry: EntryInput) -> dict[str, Any]:
    return {
        "id": entry.id,
        "type": entry.entry_type,
        "ticker": entry.ticker,
        "date": entry.timestamp.date().isoformat(),
        "body": entry.body,
    }


# keeps the patterns, in ranked order, whose narration checks out
def ground(
    narration: Narration, candidates: Sequence[Candidate], entries: Mapping[int, EntryInput]
) -> list[Pattern]:
    items: dict[int, NarratedItem] = {}
    for item in narration.items:
        items.setdefault(item.index, item)

    patterns: list[Pattern] = []
    for index, candidate in enumerate(candidates):
        item = items.get(index)
        if item is None:
            continue
        observation, question = item.observation.strip(), item.question.strip()
        # the figures carry the numbers; vestry words them
        if not observation or not question or has_number(f"{observation} {question}"):
            continue
        # quotes may only come from the entries this pattern was shown
        bodies = {i: entries[i].body for i in candidate.entry_ids if i in entries}
        quotes = [Citation(entry_id=q.entry_id, quote=q.quote) for q in item.quotes]
        citations = verified(quotes, bodies)
        # silent_trades is about the missing entry, so it has nothing to cite
        if not citations and candidate.kind != PatternKind.SILENT_TRADES:
            continue
        patterns.append(
            Pattern(
                kind=candidate.kind,
                subject=candidate.subject,
                figures=candidate.figures,
                observation=observation,
                question=question,
                citations=citations,
            )
        )
    return patterns
