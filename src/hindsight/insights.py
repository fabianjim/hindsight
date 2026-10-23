from collections.abc import Iterable

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hindsight.enrichment import Claim, Enrichment, EntryInsight
from hindsight.models import EntryInsightRecord


class InsightStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def current_hash(self, entry_id: int) -> str | None:
        async with self._sessions() as session:
            return await session.scalar(
                select(EntryInsightRecord.input_hash).where(EntryInsightRecord.entry_id == entry_id)
            )

    # false when the stored insight already came from the same input
    async def save(self, enrichment: Enrichment, user_id: int) -> bool:
        insight = enrichment.insight
        values = {
            "user_id": user_id,
            "input_hash": enrichment.input_hash,
            "model": enrichment.model,
            "prompt_version": enrichment.prompt_version,
            "thesis": dump(insight.thesis),
            "exit_condition": dump(insight.exit_condition),
            "horizon": insight.horizon,
            "conviction": insight.conviction,
            "idea_source": insight.idea_source,
            "emotions": [emotion.value for emotion in insight.emotions],
            "tickers": insight.tickers,
            "themes": insight.themes,
        }
        statement = insert(EntryInsightRecord).values(entry_id=enrichment.entry_id, **values)
        statement = statement.on_conflict_do_update(
            index_elements=[EntryInsightRecord.entry_id],
            set_={**values, "updated_at": func.now()},
            where=EntryInsightRecord.input_hash != statement.excluded.input_hash,
        ).returning(EntryInsightRecord.entry_id)
        async with self._sessions.begin() as session:
            return await session.scalar(statement) is not None

    async def delete(self, entry_id: int) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                delete(EntryInsightRecord).where(EntryInsightRecord.entry_id == entry_id)
            )

    async def for_user(self, user_id: int, entry_ids: Iterable[int]) -> dict[int, EntryInsight]:
        async with self._sessions() as session:
            records = await session.scalars(
                select(EntryInsightRecord).where(
                    EntryInsightRecord.user_id == user_id,
                    EntryInsightRecord.entry_id.in_(list(entry_ids)),
                )
            )
            return {record.entry_id: load(record) for record in records}


def dump(claim: Claim | None) -> dict[str, object] | None:
    return claim.model_dump(mode="json") if claim else None


def load(record: EntryInsightRecord) -> EntryInsight:
    return EntryInsight.model_validate(
        {
            "thesis": record.thesis,
            "exit_condition": record.exit_condition,
            "horizon": record.horizon,
            "conviction": record.conviction,
            "idea_source": record.idea_source,
            "emotions": record.emotions,
            "tickers": record.tickers,
            "themes": record.themes,
        }
    )
