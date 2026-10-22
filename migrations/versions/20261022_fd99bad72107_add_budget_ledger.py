"""add budget ledger

Revision ID: fd99bad72107
Revises:
Create Date: 2026-10-22 18:09:50.575386
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "fd99bad72107"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "budget",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("blocked", sa.Boolean(), nullable=False),
        sa.CheckConstraint("id = 1", name=op.f("ck_budget_single_row")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_budget")),
        schema="ai",
    )
    op.create_table(
        "generations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_key", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "succeeded",
                "failed",
                "unknown",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=16,
                schema="ai",
            ),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("budget_day", sa.Date(), nullable=False),
        sa.Column("reserved_micros", sa.BigInteger(), nullable=False),
        sa.Column("charged_micros", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generations")),
        sa.UniqueConstraint("request_key", name=op.f("uq_generations_request_key")),
        schema="ai",
    )
    op.create_index(
        op.f("ix_generations_budget_day"), "generations", ["budget_day"], unique=False, schema="ai"
    )
    op.create_table(
        "calls",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("step", sa.String(length=80), nullable=False),
        sa.Column("reserved_micros", sa.BigInteger(), nullable=False),
        sa.Column("charged_micros", sa.BigInteger(), nullable=False),
        sa.Column("settled", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["generation_id"],
            ["ai.generations.id"],
            name=op.f("fk_calls_generation_id_generations"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_calls")),
        sa.UniqueConstraint("generation_id", "step", name=op.f("uq_calls_generation_id_step")),
        schema="ai",
    )
    op.execute("insert into ai.budget (id, blocked) values (1, false)")


def downgrade() -> None:
    op.drop_table("calls", schema="ai")
    op.drop_index(op.f("ix_generations_budget_day"), table_name="generations", schema="ai")
    op.drop_table("generations", schema="ai")
    op.drop_table("budget", schema="ai")
