import logging
from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from hindsight.behavior_report import PROMPT_VERSION as BEHAVIOR_PROMPT_VERSION
from hindsight.behavior_report import BehaviorNarrator
from hindsight.behavior_reports import (
    MAX_LISTED,
    BehaviorReport,
    BehaviorReportList,
    BehaviorReportStore,
    SavedBehaviorReport,
)
from hindsight.behaviors import MAX_ENTRIES, BehaviorReportRequest, find_behaviors
from hindsight.budget import BudgetError, BudgetLedger, Limits, budget_day, utcnow
from hindsight.config import Settings, get_settings
from hindsight.db import get_sessionmaker
from hindsight.enrichment import EntryInsight
from hindsight.insights import InsightStore
from hindsight.llm import get_chat_model
from hindsight.pipeline import TRANSIENT_ERRORS, EntrySource
from hindsight.reflection import PROMPT_VERSION, ReflectionAidGenerator, Since
from hindsight.reflection_aids import ReflectionAid, ReflectionAidStore
from hindsight.vocab import AttemptStatus
from hindsight.wire import WireModel

MAX_ENTRY_IDS = 200
MAX_LATER_ENTRY_IDS = 50
ATTEMPTED_TODAY = "ATTEMPTED_TODAY"

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


class Problem(WireModel):
    detail: str
    # set when vestry acts on the reason, e.g. ATTEMPTED_TODAY
    code: str | None = None


@lru_cache
def get_store() -> InsightStore:
    return InsightStore(get_sessionmaker())


@lru_cache
def get_aid_store() -> ReflectionAidStore:
    return ReflectionAidStore(get_sessionmaker())


@lru_cache
def get_report_store() -> BehaviorReportStore:
    return BehaviorReportStore(get_sessionmaker())


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
def get_narrator() -> BehaviorNarrator | None:
    settings = get_settings()
    if not ai_enabled(settings):
        return None
    ledger = BudgetLedger(get_sessionmaker(), Limits.from_settings(settings))
    return BehaviorNarrator(ledger, get_chat_model(), settings.chat_model)


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


# vestry acts on the code; HTTPException would nest it under "detail"
class ProblemError(Exception):
    def __init__(self, status: int, detail: str, code: str | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.problem = Problem(detail=detail, code=code)


def attempted_today() -> ProblemError:
    return ProblemError(409, "today's behavior report attempt failed", ATTEMPTED_TODAY)


# another request started today's attempt first
class AttemptTaken(Exception):
    pass


@router.post(
    "/users/{userId}/behavior-reports",
    responses={
        409: {"model": Problem, "description": "Today's attempt failed (ATTEMPTED_TODAY)"},
        503: {"model": Problem, "description": "AI is off, out of budget or unavailable"},
    },
)
async def create_behavior_report(
    user_id: Annotated[int, Path(alias="userId")],
    request: BehaviorReportRequest,
    store: Annotated[BehaviorReportStore, Depends(get_report_store)],
    source: Annotated[EntrySource, Depends(get_source)],
    insights: Annotated[InsightStore, Depends(get_store)],
    narrator: Annotated[BehaviorNarrator | None, Depends(get_narrator)],
    today: Annotated[date, Depends(get_today)],
) -> BehaviorReport:
    if saved := await store.for_day(user_id, today):
        return saved
    if await store.attempt(user_id, today):
        raise attempted_today()

    entries = await source.newest_for_user(user_id, MAX_ENTRIES)
    found = await insights.for_user(user_id, [e.id for e in entries])
    doesnt_fit = [f.report_day for f in request.feedback if f.reaction == "DOESNT_FIT"]
    history = await store.history(user_id, today, doesnt_fit)
    # older entries weren't read when the cap is reached
    entries_from = entries[-1].timestamp if len(entries) == MAX_ENTRIES else None
    # plain python over up to a thousand entries and trades; keep it off the event loop
    findings = await run_in_threadpool(
        find_behaviors, request, entries, found, history, entries_from
    )
    # empty reports aren't saved, so they never use up the day
    empty = BehaviorReport(
        day=today,
        generated_at=utcnow(),
        as_of=request.as_of,
        entries_up_to=findings.entries_up_to,
        saved=False,
        behaviors=[],
        needs=findings.needs,
    )
    if not findings.behaviors:
        return empty
    if narrator is None:
        raise ProblemError(503, "AI is off")

    # the notes come from core_read, since they may be older than the entries read
    by_id = {e.id: e for e in entries}
    wanted = {i for b in findings.behaviors for i in (*b.shown_entry_ids, *b.note_entry_ids)}
    by_id |= await source.for_user(user_id, sorted(wanted - by_id.keys()))

    started = False

    async def start() -> None:
        nonlocal started
        if not await store.start_attempt(user_id, today):
            raise AttemptTaken
        started = True

    try:
        behaviors = await narrator.narrate(findings.behaviors, by_id, user_id, on_start=start)
    except AttemptTaken:
        if saved := await store.for_day(user_id, today):
            return saved
        raise attempted_today() from None
    except Exception as error:
        log.warning("behavior report for user %d failed (%s)", user_id, type(error).__name__)
        # a refusal never reached the model, so it doesn't use up the day
        if isinstance(error, BudgetError):
            if started:
                await store.discard_attempt(user_id, today)
            raise ProblemError(503, "AI is unavailable") from None
        if not started:
            if isinstance(error, TRANSIENT_ERRORS):
                raise ProblemError(503, "AI is unavailable") from None
            raise
        await store.finish_attempt(user_id, today, AttemptStatus.FAILED)
        if isinstance(error, TRANSIENT_ERRORS):
            raise attempted_today() from None
        raise
    if not behaviors:
        await store.finish_attempt(user_id, today, AttemptStatus.FAILED)
        log.warning("behavior report for user %d had no narration that checked out", user_id)
        raise attempted_today()
    report = empty.model_copy(update={"saved": True, "behaviors": behaviors})
    return await store.save(user_id, report, narrator.model_name, BEHAVIOR_PROMPT_VERSION)


# read-only, so demo visitors can be shown the template's reports without ever spending
@router.get("/users/{userId}/behavior-reports")
async def list_behavior_reports(
    user_id: Annotated[int, Path(alias="userId")],
    store: Annotated[BehaviorReportStore, Depends(get_report_store)],
    before: Annotated[date | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LISTED)] = 10,
) -> BehaviorReportList:
    return await store.page(user_id, before, limit)


@router.get("/users/{userId}/behavior-reports/{day}")
async def get_behavior_report(
    user_id: Annotated[int, Path(alias="userId")],
    day: Annotated[date, Path()],
    store: Annotated[BehaviorReportStore, Depends(get_report_store)],
) -> SavedBehaviorReport:
    return SavedBehaviorReport(report=await store.for_day(user_id, day))
