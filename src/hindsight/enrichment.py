import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hindsight.budget import BudgetLedger
from hindsight.citations import Citation, verified
from hindsight.llm import UserContext
from hindsight.spend import BudgetMiddleware
from hindsight.tracing import trace_callbacks
from hindsight.vocab import Emotion, EntryType, Horizon, IdeaSource

PROMPT_VERSION = "entry-insight-v1"
PROMPT = (Path(__file__).parent / "prompts" / "entry_insight.md").read_text()
MAX_OUTPUT_TOKENS = 512
ALLOWANCE_MICROS = 10_000

Ticker = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$")]


class EntryInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    entry_type: EntryType
    body: str = Field(min_length=1, max_length=2000)
    ticker: str | None = None
    timestamp: datetime
    price_snapshot: float | None = None
    tags: list[str] = []


class ExtractedClaim(BaseModel):
    summary: str
    quotes: list[str] = Field(max_length=3)


class Extraction(BaseModel):
    thesis: ExtractedClaim | None
    exit_condition: ExtractedClaim | None
    horizon: Horizon | None
    conviction: Literal[1, 2, 3, 4, 5] | None
    emotions: list[Emotion] = Field(max_length=3)
    idea_source: IdeaSource | None
    tickers: list[Ticker]
    themes: list[str] = Field(max_length=5)


class Claim(BaseModel):
    summary: str
    citations: list[Citation] = Field(min_length=1)


class EntryInsight(BaseModel):
    thesis: Claim | None
    exit_condition: Claim | None
    horizon: Horizon | None
    conviction: Literal[1, 2, 3, 4, 5] | None
    emotions: list[Emotion]
    idea_source: IdeaSource | None
    tickers: list[Ticker]
    themes: list[str]


@dataclass(frozen=True)
class Enrichment:
    entry_id: int
    insight: EntryInsight
    input_hash: str
    model: str
    prompt_version: str


class EntryEnricher:
    def __init__(self, ledger: BudgetLedger, model: BaseChatModel, model_name: str) -> None:
        self._ledger = ledger
        self._model_name = model_name
        self._agent = create_agent(
            model,
            system_prompt=PROMPT,
            response_format=ProviderStrategy(Extraction, strict=True),
            context_schema=UserContext,
            middleware=[BudgetMiddleware(ledger, model_name, "extract", MAX_OUTPUT_TOKENS)],
        )

    async def enrich(self, entry: EntryInput, user_id: int) -> Enrichment:
        message = render(entry)
        # a fresh key per attempt: the ledger never replays a key, and a retry must stay possible
        job = await self._ledger.start(secrets.token_hex(32), ALLOWANCE_MICROS)
        success = False
        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(message)]},
                context=UserContext(user_id=user_id, generation_id=job.id),
                config={"callbacks": trace_callbacks(), "run_name": "entry-insight"},
            )
            extraction = result["structured_response"]
            success = True
        finally:
            await self._ledger.finish(job.id, success)
        assert isinstance(extraction, Extraction)
        return Enrichment(
            entry_id=entry.id,
            insight=ground(extraction, entry),
            input_hash=input_hash(message, self._model_name),
            model=self._model_name,
            prompt_version=PROMPT_VERSION,
        )


def render(entry: EntryInput) -> str:
    # json keeps the body a quoted string, so it can't pose as part of the prompt
    fields = {
        "type": entry.entry_type,
        "ticker": entry.ticker,
        "date": entry.timestamp.date().isoformat(),
        "tags": entry.tags,
        "body": entry.body,
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


def input_hash(message: str, model_name: str) -> str:
    return hashlib.sha256("\0".join([PROMPT, model_name, message]).encode()).hexdigest()


def ground(extraction: Extraction, entry: EntryInput) -> EntryInsight:
    def claim(extracted: ExtractedClaim | None) -> Claim | None:
        if extracted is None or not extracted.summary.strip():
            return None
        quotes = [Citation(entry_id=entry.id, quote=q) for q in extracted.quotes]
        citations = verified(quotes, {entry.id: entry.body})
        return Claim(summary=extracted.summary.strip(), citations=citations) if citations else None

    return EntryInsight(
        thesis=claim(extraction.thesis),
        exit_condition=claim(extraction.exit_condition),
        horizon=extraction.horizon,
        conviction=extraction.conviction,
        emotions=list(dict.fromkeys(extraction.emotions)),
        idea_source=extraction.idea_source,
        tickers=list(dict.fromkeys(extraction.tickers)),
        themes=list(dict.fromkeys(t.strip() for t in extraction.themes if t.strip())),
    )
