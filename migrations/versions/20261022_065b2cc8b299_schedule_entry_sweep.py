"""schedule entry sweep

Revision ID: 065b2cc8b299
Revises: da8f7818bf65
Create Date: 2026-10-22 20:58:53.139655
"""

from collections.abc import Sequence

from alembic import op

revision: str = "065b2cc8b299"
down_revision: str | Sequence[str] | None = "da8f7818bf65"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# pg_cron only installs in the database named by cron.database_name, so tests skip it
UPGRADE = """
do $$
begin
    if current_database() = current_setting('cron.database_name', true) then
        create extension if not exists pg_cron;
        perform cron.schedule(
            'hindsight-entry-sweep',
            '17 * * * *',
            $job$select pgmq.send('entry_events', '{"sweep": "entries"}')$job$
        );
    else
        raise notice 'pg_cron runs in %, not scheduling the sweep',
            coalesce(current_setting('cron.database_name', true), 'no database');
    end if;
end $$
"""

DOWNGRADE = """
do $$
begin
    if to_regnamespace('cron') is not null
        and exists (select 1 from cron.job where jobname = 'hindsight-entry-sweep') then
        perform cron.unschedule('hindsight-entry-sweep');
    end if;
end $$
"""


def upgrade() -> None:
    op.execute(UPGRADE)


def downgrade() -> None:
    # the extension stays; other jobs may use it
    op.execute(DOWNGRADE)
