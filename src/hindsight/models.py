import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    MetaData,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.schema import UniqueConstraint

from hindsight.vocab import Horizon, IdeaSource

SCHEMA = "ai"


class Base(DeclarativeBase):
    metadata = MetaData(
        schema=SCHEMA,
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        },
    )


# locked by every spend; a missing or blocked row stops all spending
class Budget(Base):
    __tablename__ = "budget"
    __table_args__ = (CheckConstraint("id = 1", name="single_row"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    blocked: Mapped[bool] = mapped_column(default=False)


class Status(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


def enum_values(members: type[StrEnum]) -> list[str]:
    return [m.value for m in members]


def text_enum(members: type[StrEnum], name: str) -> Enum:
    # varchar plus a check constraint, so adding a value is a plain migration
    return Enum(
        members,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=16,
        values_callable=enum_values,
    )


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_key: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[Status] = mapped_column(text_enum(Status, "status"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    budget_day: Mapped[date] = mapped_column(index=True)
    reserved_micros: Mapped[int] = mapped_column(BigInteger)
    charged_micros: Mapped[int] = mapped_column(BigInteger)


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (UniqueConstraint("generation_id", "step"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    generation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("generations.id"))
    step: Mapped[str] = mapped_column(String(80))
    reserved_micros: Mapped[int] = mapped_column(BigInteger)
    charged_micros: Mapped[int] = mapped_column(BigInteger)
    settled: Mapped[bool] = mapped_column(default=False)


class EntryInsightRecord(Base):
    __tablename__ = "entry_insights"
    __table_args__ = (CheckConstraint("conviction between 1 and 5", name="conviction"),)

    # vestry's entry id; no foreign key since vestry owns that table
    entry_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    input_hash: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    thesis: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    exit_condition: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    horizon: Mapped[Horizon | None] = mapped_column(text_enum(Horizon, "horizon"))
    conviction: Mapped[int | None] = mapped_column(SmallInteger)
    idea_source: Mapped[IdeaSource | None] = mapped_column(text_enum(IdeaSource, "idea_source"))
    emotions: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    tickers: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    themes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
