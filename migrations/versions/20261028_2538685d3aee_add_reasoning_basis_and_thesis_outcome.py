"""add reasoning basis and thesis outcome

Revision ID: 2538685d3aee
Revises: 1b0265e7b76e
Create Date: 2026-10-28 22:54:28.855201
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2538685d3aee"
down_revision: str | Sequence[str] | None = "1b0265e7b76e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "entry_insights",
        sa.Column(
            "reasoning_basis",
            sa.Enum(
                "fundamentals",
                "price_action",
                "commentary",
                "mixed",
                name="reasoning_basis",
                native_enum=False,
                create_constraint=True,
                length=16,
                schema="ai",
            ),
            nullable=True,
        ),
        schema="ai",
    )
    op.add_column(
        "entry_insights",
        sa.Column(
            "thesis_outcome",
            sa.Enum(
                "played_out",
                "did_not",
                "mixed",
                name="thesis_outcome",
                native_enum=False,
                create_constraint=True,
                length=16,
                schema="ai",
            ),
            nullable=True,
        ),
        schema="ai",
    )


def downgrade() -> None:
    op.drop_column("entry_insights", "thesis_outcome", schema="ai")
    op.drop_column("entry_insights", "reasoning_basis", schema="ai")
