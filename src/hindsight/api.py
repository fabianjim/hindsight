from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from hindsight.db import get_sessionmaker
from hindsight.enrichment import EntryInsight
from hindsight.insights import InsightStore
from hindsight.wire import WireModel

MAX_ENTRY_IDS = 200

router = APIRouter()


class EntryInsightItem(WireModel):
    entry_id: int
    insight: EntryInsight


class EntryInsightList(WireModel):
    insights: list[EntryInsightItem]


@lru_cache
def get_store() -> InsightStore:
    return InsightStore(get_sessionmaker())


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
