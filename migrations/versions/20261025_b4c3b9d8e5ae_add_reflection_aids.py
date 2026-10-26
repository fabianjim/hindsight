"""add reflection aids

Revision ID: b4c3b9d8e5ae
Revises: 065b2cc8b299
Create Date: 2026-10-25 21:21:52.722213
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b4c3b9d8e5ae"
down_revision: str | Sequence[str] | None = "065b2cc8b299"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reflection_aids",
        sa.Column("user_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("entry_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("recap", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("questions", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id", "day", "entry_id", name=op.f("pk_reflection_aids")),
        schema="ai",
    )


def downgrade() -> None:
    op.drop_table("reflection_aids", schema="ai")
