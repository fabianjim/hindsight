"""add pattern reports

Revision ID: 1b0265e7b76e
Revises: b4c3b9d8e5ae
Create Date: 2026-10-27 20:08:58.392093
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "1b0265e7b76e"
down_revision: str | Sequence[str] | None = "b4c3b9d8e5ae"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pattern_reports",
        sa.Column("user_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("report", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("entries_up_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id", "day", name=op.f("pk_pattern_reports")),
        schema="ai",
    )


def downgrade() -> None:
    op.drop_table("pattern_reports", schema="ai")
