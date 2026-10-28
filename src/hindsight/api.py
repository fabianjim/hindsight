import logging
from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from hindsight.budget import BudgetError, BudgetLedger, Limits, budget_day, utcnow
from hindsight.config import Settings, get_settings
from hindsight.db import get_sessionmaker
from hindsight.enrichment import EntryInsight
from hindsight.insights import InsightStore
from hindsight.llm import get_chat_model
from hindsight.pattern_report import PROMPT_VERSION as PATTERN_PROMPT_VERSION
from hindsight.pattern_report import PatternNarrator
from hindsight.pattern_reports import PatternReport, PatternReportStore
from hindsight.patterns import MAX_ENTRIES, PatternReportRequest, find_patterns
from hindsight.pipeline import TRANSIENT_ERRORS, EntrySource
from hindsight.reflection import PROMPT_VERSION, ReflectionAidGenerator, Since
from hindsight.reflection_aids import ReflectionAid, ReflectionAidStore
from hindsight.wire import WireModel

MAX_ENTRY_IDS = 200
MAX_LATER_ENTRY_IDS = 50

log = logging.getLogger("hindsight.api")

router = APIRouter()


class EntryInsightItem(WireModel):
    entry_id: int
    insight: EntryInsight


class EntryInsightList(WireModel):
    insights: list[EntryInsightItem]


class ReflectionAidRequest(WireModel):
    # vestry picks them; only the user's entries written after this one are used
    later_entry_ids: list[int] = Field(default=[], max_length=MAX_LATER_ENTRY_IDS)
    since: Since = Since()


class ReflectionAidResult(WireModel):
    aid: ReflectionAid
    generated_on: date
    remaining_today: int


class SavedReflectionAid(WireModel):
    entry_id: int
    aid: ReflectionAid
    generated_on: date


class SavedReflectionAidList(WireModel):
    aids: list[SavedReflectionAid]


class SavedPatternReport(WireModel):
    report: PatternReport | None


class Problem(WireModel):
    detail: str


@lru_cache
def get_store() -> InsightStore:
    return InsightStore(get_sessionmaker())


@lru_cache
def get_aid_store() -> ReflectionAidStore:
    return ReflectionAidStore(get_sessionmaker())


@lru_cache
def get_pattern_store() -> PatternReportStore:
    return PatternReportStore(get_sessionmaker())


@lru_cache
def get_source() -> EntrySource:
    return EntrySource(get_sessionmaker())


def ai_enabled(settings: Settings) -> bool:
    return settings.openai_api_key is not None and bool(settings.openai_api_key.get_secret_value())


# none when no api key is set, so saved aids are still served with the ai off
@lru_cache
def get_generator() -> ReflectionAidGenerator | None:
    settings = get_settings()
    if not ai_enabled(settings):
        return None
    ledger = BudgetLedger(get_sessionmaker(), Limits.from_settings(settings))
    return ReflectionAidGenerator(ledger, get_chat_model(), settings.chat_model)


# none when no api key is set, so saved reports are still served with the ai off
@lru_cache
def get_narrator() -> PatternNarrator | None:
    settings = get_settings()
    if not ai_enabled(settings):
        return None
    ledger = BudgetLedger(get_sessionmaker(), Limits.from_settings(settings))
    return PatternNarrator(ledger, get_chat_model(), settings.chat_model)


def get_today() -> date:
    return budget_day(utcnow())


@router.get("/users/{userId}/insights")
async def list_insights(
    user_id: Annotated[int, Path(alias="userId")],
    entry_ids: Annotated[
        list[int], Query(alias="entryIds", min_length=1, max_length=MAX_ENTRY_IDS)
    ],
    store: Annotated[InsightStore, Depends(get_store)],
) -> EntryInsightList:
    found = await store.for_user(user_id, set(entry_ids))
    return EntryInsightList(
        insights=[EntryInsightItem(entry_id=i, insight=found[i]) for i in sorted(found)]
    )


# read-only, so demo visitors can be shown saved aids without ever spending
@router.get("/users/{userId}/reflection-aids")
async def list_reflection_aids(
    user_id: Annotated[int, Path(alias="userId")],
    entry_ids: Annotated[
        list[int], Query(alias="entryIds", min_length=1, max_length=MAX_ENTRY_IDS)
    ],
    store: Annotated[ReflectionAidStore, Depends(get_aid_store)],
) -> SavedReflectionAidList:
    found = await store.latest(user_id, set(entry_ids))
    return SavedReflectionAidList(
        aids=[
            SavedReflectionAid(entry_id=i, aid=found[i].aid, generated_on=found[i].day)
            for i in sorted(found)
        ]
    )


@router.post(
    "/users/{userId}/entries/{entryId}/reflection-aid",
    responses={
        404: {"model": Problem, "description": "No entry with text for this user"},
        429: {"model": Problem, "description": "The user's aids for today are used up"},
        503: {"model": Problem, "description": "AI is off, out of budget or unavailable"},
    },
)
async def create_reflection_aid(
    user_id: Annotated[int, Path(alias="userId")],
    entry_id: Annotated[int, Path(alias="entryId")],
    request: ReflectionAidRequest,
    store: Annotated[ReflectionAidStore, Depends(get_aid_store)],
    source: Annotated[EntrySource, Depends(get_source)],
    generator: Annotated[ReflectionAidGenerator | None, Depends(get_generator)],
    today: Annotated[date, Depends(get_today)],
) -> ReflectionAidResult:
    if cached := await store.for_day(user_id, entry_id, today):
        return ReflectionAidResult(
            aid=cached, generated_on=today, remaining_today=await store.remaining(user_id, today)
        )
    if await store.remaining(user_id, today) == 0:
        raise HTTPException(429, "today's reflection aids are used up")
    if generator is None:
        raise HTTPException(503, "AI is off")

    found = await source.for_user(user_id, [entry_id, *request.later_entry_ids])
    entry = found.get(entry_id)
    if entry is None or not entry.body.strip():
        raise HTTPException(404, "entry not found")
    later = [e for e in found.values() if e.id != entry_id and e.timestamp > entry.timestamp]

    try:
        aid = await generator.generate(entry, later, request.since, user_id)
    except (BudgetError, *TRANSIENT_ERRORS) as error:
        log.warning("reflection aid for entry %d failed (%s)", entry_id, type(error).__name__)
        raise HTTPException(503, "AI is unavailable") from None
    stored = await store.save(user_id, entry_id, today, aid, generator.model_name, PROMPT_VERSION)
    return ReflectionAidResult(
        aid=stored, generated_on=today, remaining_today=await store.remaining(user_id, today)
    )


# read-only, so demo visitors can be shown the template's report without ever spending
@router.get("/users/{userId}/pattern-report")
async def get_pattern_report(
    user_id: Annotated[int, Path(alias="userId")],
    store: Annotated[PatternReportStore, Depends(get_pattern_store)],
) -> SavedPatternReport:
    return SavedPatternReport(report=await store.latest(user_id))


@router.post(
    "/users/{userId}/pattern-report",
    responses={
        503: {"model": Problem, "description": "AI is off, out of budget or unavailable"},
    },
)
async def create_pattern_report(
    user_id: Annotated[int, Path(alias="userId")],
    request: PatternReportRequest,
    store: Annotated[PatternReportStore, Depends(get_pattern_store)],
    source: Annotated[EntrySource, Depends(get_source)],
    insights: Annotated[InsightStore, Depends(get_store)],
    narrator: Annotated[PatternNarrator | None, Depends(get_narrator)],
    today: Annotated[date, Depends(get_today)],
) -> PatternReport:
    if cached := await store.for_day(user_id, today):
        return cached

    entries = await source.newest_for_user(user_id, MAX_ENTRIES)
    found = await insights.for_user(user_id, [e.id for e in entries])
    # older entries weren't read when the cap is reached
    entries_from = entries[-1].timestamp if len(entries) == MAX_ENTRIES else None
    # plain python over up to a thousand entries and trades; keep it off the event loop
    candidates = await run_in_threadpool(find_patterns, request, entries, found, entries_from)
    # empty reports aren't saved, so they never use up the day
    empty = PatternReport(
        generated_on=today,
        as_of=request.as_of,
        entries_up_to=candidates.entries_up_to,
        saved=False,
        patterns=[],
        needs=candidates.needs,
    )
    if not candidates.patterns:
        return empty
    if narrator is None:
        raise HTTPException(503, "AI is off")

    try:
        patterns = await narrator.narrate(candidates.patterns, {e.id: e for e in entries}, user_id)
    except (BudgetError, *TRANSIENT_ERRORS) as error:
        log.warning("pattern report for user %d failed (%s)", user_id, type(error).__name__)
        raise HTTPException(503, "AI is unavailable") from None
    if not patterns:
        # saving it would leave the user nothing to refresh away from until tomorrow
        log.warning("pattern report for user %d had no narration that checked out", user_id)
        return empty
    report = empty.model_copy(update={"saved": True, "patterns": patterns})
    return await store.save(user_id, report, narrator.model_name, PATTERN_PROMPT_VERSION)
