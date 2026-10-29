import json
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from hindsight.behavior_reports import MAX_BEHAVIORS, Behavior
from hindsight.behaviors import FoundBehavior
from hindsight.budget import BudgetLedger
from hindsight.citations import Citation, verified
from hindsight.enrichment import EntryInput
from hindsight.llm import UserContext
from hindsight.pricing import Price, price_of
from hindsight.spend import REQUEST_OVERHEAD_TOKENS, BudgetMiddleware
from hindsight.tracing import trace_callbacks
from hindsight.vocab import BehaviorKind

PROMPT_VERSION = "behavior-report-v4"
PROMPT = (Path(__file__).parent / "prompts" / "behavior_report.md").read_text()
MAX_OUTPUT_TOKENS = 1200
# the observation has to fit a card that can't scroll
MAX_SENTENCES = 2
MAX_SENTENCE_WORDS = 24
# counts spelled out; proportions like "most" or "about half" are what the prompt asks for.
# "every one" and "each one" mean all of them, so they aren't counts
NUMBER_WORDS = re.compile(
    r"\b((?<!\bevery )(?<!\beach )one|two|three|four|five|six|seven|eight|nine|ten|eleven"
    r"|twelve|twenty|hundred|once|twice|dozen)\b",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


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

    items: list[NarratedItem] = Field(max_length=MAX_BEHAVIORS)


@dataclass(frozen=True)
class NarrationRun:
    narration: Narration
    behaviors: list[Behavior]


RESPONSE_FORMAT = ProviderStrategy(Narration, strict=True)
SCHEMA = json.dumps(RESPONSE_FORMAT.schema_spec.json_schema)


class BehaviorNarrator:
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

    # entries holds each behavior's shown entries and its note entries. on_start runs once the
    # reservation succeeds, right before the model call; if it raises, nothing is charged
    async def narrate(
        self,
        behaviors: Sequence[FoundBehavior],
        entries: Mapping[int, EntryInput],
        user_id: int,
        on_start: Callable[[], Awaitable[None]] | None = None,
    ) -> list[Behavior]:
        return (await self.run(behaviors, entries, user_id, on_start)).behaviors

    # the model's narration alongside the grounded behaviors, for evals
    async def run(
        self,
        behaviors: Sequence[FoundBehavior],
        entries: Mapping[int, EntryInput],
        user_id: int,
        on_start: Callable[[], Awaitable[None]] | None = None,
    ) -> NarrationRun:
        if not behaviors:
            raise ValueError("no behaviors to narrate")
        if len(behaviors) > MAX_BEHAVIORS:
            raise ValueError("too many behaviors to narrate")
        message = render(behaviors, entries)
        # a fresh key per attempt: the ledger never replays a key, and a retry must stay possible
        job = await self._ledger.start(secrets.token_hex(32), allowance_for(self._price, message))
        success = False
        try:
            if on_start is not None:
                await on_start()
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(message)]},
                context=UserContext(user_id=user_id, generation_id=job.id),
                config={"callbacks": trace_callbacks(), "run_name": "behavior-report"},
            )
            narration = result["structured_response"]
            success = True
        finally:
            await self._ledger.finish(job.id, success)
        assert isinstance(narration, Narration)
        return NarrationRun(narration=narration, behaviors=ground(narration, behaviors, entries))


# the middleware re-encodes the request as json, which at most doubles each byte
def allowance_for(price: Price, message: str) -> int:
    request_bytes = len((PROMPT + message + SCHEMA).encode())
    return price.cost(2 * request_bytes + REQUEST_OVERHEAD_TOKENS, MAX_OUTPUT_TOKENS)


def render(behaviors: Sequence[FoundBehavior], entries: Mapping[int, EntryInput]) -> str:
    # json keeps each body a quoted string, so it can't pose as part of the prompt
    fields = {
        "behaviors": [
            {
                "index": index,
                "kind": b.kind,
                "subject": b.subject,
                "figures": [
                    {"key": f.key, "value": figure_value(f.value), "unit": f.unit}
                    for f in b.figures
                ],
                "entries": [entry_fields(entries[i]) for i in b.shown_entry_ids if i in entries],
                "notes": notes_of(b, entries),
            }
            for index, b in enumerate(behaviors)
        ]
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


# context for the wording only; never citable
def notes_of(behavior: FoundBehavior, entries: Mapping[int, EntryInput]) -> list[str]:
    notes = [n for n in behavior.feedback_notes if n.strip()]
    notes += [entries[i].body for i in behavior.note_entry_ids if i in entries]
    return notes


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


# keeps the behaviors, in ranked order, whose narration checks out
def ground(
    narration: Narration, behaviors: Sequence[FoundBehavior], entries: Mapping[int, EntryInput]
) -> list[Behavior]:
    items: dict[int, NarratedItem] = {}
    for item in narration.items:
        items.setdefault(item.index, item)

    grounded: list[Behavior] = []
    for index, found in enumerate(behaviors):
        item = items.get(index)
        if item is None:
            continue
        observation, question = item.observation.strip(), item.question.strip()
        if not observation or not question or not fits_card(observation):
            continue
        # the figures carry the numbers; vestry words them
        if has_number(f"{observation} {question}"):
            continue
        # quotes may only come from the entries this behavior was shown, never its notes
        bodies = {i: entries[i].body for i in found.shown_entry_ids if i in entries}
        quotes = [Citation(entry_id=q.entry_id, quote=q.quote) for q in item.quotes]
        citations = verified(quotes, bodies)
        # undocumented is about the missing entry, so it has nothing to cite
        if not citations and found.kind != BehaviorKind.UNDOCUMENTED:
            continue
        grounded.append(
            Behavior(
                kind=found.kind,
                subject=found.subject,
                figures=found.figures,
                previous=found.previous,
                observation=observation,
                question=question,
                citations=citations,
                trade_ids=found.trade_ids,
                entry_ids=found.entry_ids,
            )
        )
    return grounded


# at most two sentences, each under 25 words
def fits_card(observation: str) -> bool:
    sentences = SENTENCE_END.split(observation)
    return len(sentences) <= MAX_SENTENCES and all(
        len(s.split()) <= MAX_SENTENCE_WORDS for s in sentences
    )


# a count in the text could disagree with the figures shown beside it
def has_number(text: str) -> bool:
    return any(c.isdigit() for c in text) or NUMBER_WORDS.search(text) is not None
