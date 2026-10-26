import json
import secrets
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from hindsight.budget import BudgetLedger, budget_day, utcnow
from hindsight.citations import Citation, verified
from hindsight.enrichment import Claim, EntryInput
from hindsight.llm import UserContext
from hindsight.pricing import price_of
from hindsight.reflection_aids import ReflectionAid
from hindsight.spend import REQUEST_OVERHEAD_TOKENS, BudgetMiddleware
from hindsight.tracing import trace_callbacks
from hindsight.wire import WireModel

PROMPT_VERSION = "reflection-aid-v1"
PROMPT = (Path(__file__).parent / "prompts" / "reflection_aid.md").read_text()
MAX_OUTPUT_TOKENS = 512
# newest later entries the model sees; older ones are left out
MAX_LATER_ENTRIES = 10


# vestry's numbers for what happened after the entry
class Trade(WireModel):
    day: date
    type: Literal["BUY", "SELL"]
    shares: float = Field(gt=0)
    price: float = Field(gt=0)


class Since(WireModel):
    price_now: float | None = Field(default=None, gt=0)
    shares_now: float | None = Field(default=None, ge=0)
    trades: list[Trade] = Field(default=[], max_length=20)


# openai strict mode requires additionalProperties: false on every object
class DraftPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    entry_id: int
    quotes: list[str] = Field(max_length=2)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recap: list[DraftPoint] = Field(max_length=3)
    questions: list[str] = Field(max_length=3)


RESPONSE_FORMAT = ProviderStrategy(Draft, strict=True)
SCHEMA = json.dumps(RESPONSE_FORMAT.schema_spec.json_schema)


class ReflectionAidGenerator:
    def __init__(
        self,
        ledger: BudgetLedger,
        model: BaseChatModel,
        model_name: str,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._ledger = ledger
        self._model_name = model_name
        self._price = price_of(model_name)
        self._clock = clock
        self._agent = create_agent(
            model,
            system_prompt=PROMPT,
            response_format=RESPONSE_FORMAT,
            context_schema=UserContext,
            middleware=[BudgetMiddleware(ledger, model_name, "reflect", MAX_OUTPUT_TOKENS)],
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self, entry: EntryInput, later: Iterable[EntryInput], since: Since, user_id: int
    ) -> ReflectionAid:
        if not entry.body.strip():
            raise ValueError("entry has no text to read")
        later = newest_later(entry, later)
        message = render(entry, later, since, budget_day(self._clock()))
        # a fresh key per attempt: the ledger never replays a key, and a retry must stay possible
        job = await self._ledger.start(secrets.token_hex(32), self.allowance_for(message))
        success = False
        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(message)]},
                context=UserContext(user_id=user_id, generation_id=job.id),
                config={"callbacks": trace_callbacks(), "run_name": "reflection-aid"},
            )
            draft = result["structured_response"]
            success = True
        finally:
            await self._ledger.finish(job.id, success)
        assert isinstance(draft, Draft)
        return ground(draft, later)

    # the middleware re-encodes the request as json, which at most doubles each byte
    def allowance_for(self, message: str) -> int:
        request_bytes = len((PROMPT + message + SCHEMA).encode())
        return self._price.cost(2 * request_bytes + REQUEST_OVERHEAD_TOKENS, MAX_OUTPUT_TOKENS)


def newest_later(entry: EntryInput, later: Iterable[EntryInput]) -> list[EntryInput]:
    others = sorted((e for e in later if e.id != entry.id), key=lambda e: e.timestamp)
    return others[-MAX_LATER_ENTRIES:]


def render(entry: EntryInput, later: list[EntryInput], since: Since, today: date) -> str:
    # json keeps each body a quoted string, so it can't pose as part of the prompt
    fields = {
        "entry": entry_fields(entry),
        "since": since_fields(entry, since, today),
        "later_entries": [{"id": e.id, **entry_fields(e)} for e in later],
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


def entry_fields(entry: EntryInput) -> dict[str, Any]:
    return {
        "type": entry.entry_type,
        "ticker": entry.ticker,
        "date": entry.timestamp.date().isoformat(),
        "tags": entry.tags,
        "body": entry.body,
    }


def since_fields(entry: EntryInput, since: Since, today: date) -> dict[str, Any]:
    # vestry stores 0 when it had no price for the ticker
    then = entry.price_snapshot if entry.price_snapshot and entry.price_snapshot > 0 else None
    now = since.price_now if entry.ticker else None
    change = round((now - then) / then * 100, 1) if then and now else None
    return {
        "days_passed": (today - entry.timestamp.date()).days,
        "price_then": then,
        "price_now": now,
        "price_change_pct": change,
        "shares_now": since.shares_now if entry.ticker else None,
        "trades": [
            {"date": t.day.isoformat(), "type": t.type, "shares": t.shares, "price": t.price}
            for t in since.trades
        ],
    }


def ground(draft: Draft, later: list[EntryInput]) -> ReflectionAid:
    # recap points may only cite the later entries the model was shown
    bodies = {e.id: e.body for e in later}
    recap: list[Claim] = []
    for point in draft.recap:
        quotes = [Citation(entry_id=point.entry_id, quote=q) for q in point.quotes]
        citations = verified(quotes, bodies)
        if point.summary.strip() and citations:
            recap.append(Claim(summary=point.summary.strip(), citations=citations))
    questions = list(dict.fromkeys(q.strip() for q in draft.questions if q.strip()))
    return ReflectionAid(recap=recap, questions=questions)
