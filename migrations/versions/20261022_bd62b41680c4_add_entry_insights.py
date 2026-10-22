"""add entry insights

Revision ID: bd62b41680c4
Revises: fd99bad72107
Create Date: 2026-10-22 19:05:01.377303
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "bd62b41680c4"
down_revision: str | Sequence[str] | None = "fd99bad72107"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entry_insights",
        sa.Column("entry_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("thesis", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("exit_condition", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "horizon",
            sa.Enum(
                "days",
                "weeks",
                "months",
                "years",
                name="horizon",
                native_enum=False,
                create_constraint=True,
                length=16,
                schema="ai",
            ),
            nullable=True,
        ),
        sa.Column("conviction", sa.SmallInteger(), nullable=True),
        sa.Column(
            "idea_source",
            sa.Enum(
                "own_research",
                "news",
                "social_media",
                "friend_or_family",
                "professional",
                name="idea_source",
                native_enum=False,
                create_constraint=True,
                schema="ai",
            ),
            nullable=True,
        ),
        sa.Column("emotions", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("tickers", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("themes", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("conviction between 1 and 5", name=op.f("ck_entry_insights_conviction")),
        sa.PrimaryKeyConstraint("entry_id", name=op.f("pk_entry_insights")),
        schema="ai",
    )
    op.create_index(
        op.f("ix_entry_insights_user_id"), "entry_insights", ["user_id"], unique=False, schema="ai"
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_entry_insights_user_id"), table_name="entry_insights", schema="ai")
    op.drop_table("entry_insights", schema="ai")
