"""add entry events queue

Revision ID: f4da2ebd8aaa
Revises: bd62b41680c4
Create Date: 2026-10-22 20:11:05.498884
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f4da2ebd8aaa"
down_revision: str | Sequence[str] | None = "bd62b41680c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("create extension if not exists pgmq")
    op.execute("select pgmq.create('entry_events')")


def downgrade() -> None:
    # the extension stays; other queues may use it
    op.execute("select pgmq.drop_queue('entry_events')")
