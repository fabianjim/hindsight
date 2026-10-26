-- privileges for hindsight's runtime login. run as postgres after `alembic upgrade head`;
-- the role itself is created separately with its password, so no secret lives here.
-- safe to rerun.

-- hindsight's own tables. migrations run as postgres, so tables they add later get the same grants
grant usage on schema ai to hindsight;
grant select, insert, update, delete on all tables in schema ai to hindsight;
revoke insert, update, delete on ai.alembic_version from hindsight;
alter default privileges for role postgres in schema ai
    grant select, insert, update, delete on tables to hindsight;

-- vestry's read contract. the view's owner reads the underlying tables, so hindsight never touches them
grant usage on schema core_read to hindsight;
grant select on core_read.journal_entries to hindsight;

-- the entry_events queue and its archive, nothing else in pgmq
grant usage on schema pgmq to hindsight;
grant select, insert, update, delete on pgmq.q_entry_events, pgmq.a_entry_events to hindsight;

-- a stuck query can't hold a pooled connection forever
alter role hindsight set statement_timeout = '30s';
