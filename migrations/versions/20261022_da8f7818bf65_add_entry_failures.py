"""add entry failures

Revision ID: da8f7818bf65
Revises: f4da2ebd8aaa
Create Date: 2026-10-22 20:46:22.730086
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "da8f7818bf65"
down_revision: str | Sequence[str] | None = "f4da2ebd8aaa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entry_failures",
        sa.Column("entry_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("failures", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("entry_id", name=op.f("pk_entry_failures")),
        schema="ai",
    )


def downgrade() -> None:
    op.drop_table("entry_failures", schema="ai")
