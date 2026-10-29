"""add behavior reports

Revision ID: 5f4414bab119
Revises: 2538685d3aee
Create Date: 2026-10-29 10:55:02.323343
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "5f4414bab119"
down_revision: str | Sequence[str] | None = "2538685d3aee"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "behavior_report_attempts",
        sa.Column("user_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "SAVED",
                "FAILED",
                name="attempt_status",
                native_enum=False,
                create_constraint=True,
                length=16,
                schema="ai",
            ),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("user_id", "day", name=op.f("pk_behavior_report_attempts")),
        schema="ai",
    )
    op.create_table(
        "behavior_reports",
        sa.Column("user_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("entries_up_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("report", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "day", name=op.f("pk_behavior_reports")),
        schema="ai",
    )


def downgrade() -> None:
    op.drop_table("behavior_reports", schema="ai")
    op.drop_table("behavior_report_attempts", schema="ai")
