import uuid
from datetime import date, datetime
from enum import StrEnum

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Enum, ForeignKey, MetaData, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.schema import UniqueConstraint

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


def status_values(statuses: type[Status]) -> list[str]:
    return [s.value for s in statuses]


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_key: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[Status] = mapped_column(
        Enum(
            Status,
            name="status",
            native_enum=False,
            create_constraint=True,
            length=16,
            values_callable=status_values,
        )
    )
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
