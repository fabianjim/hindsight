from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from hindsight.config import get_settings
from hindsight.db import get_engine

SERVICE_TOKEN = "test-token-" + "x" * 32
SERVER_URL = "postgresql+psycopg://postgres:postgres@localhost:54322"
TEST_DATABASE = "hindsight_test"
TRUNCATED = [
    "ai.calls",
    "ai.generations",
    "ai.budget",
    "ai.entry_insights",
    "ai.entry_failures",
    "core_read.journal_entries",
    "pgmq.q_entry_events",
    "pgmq.a_entry_events",
]
CORE_READ_DDL = """
create schema core_read;
create table core_read.journal_entries (
    id integer primary key,
    user_id integer not null,
    entry_type text not null,
    body text not null,
    ticker text,
    timestamp timestamptz not null,
    price_snapshot double precision,
    tags text[] not null default '{}'
)
"""


@pytest.fixture(autouse=True)
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HINDSIGHT_SERVICE_TOKEN", SERVICE_TOKEN)
    monkeypatch.setenv("HINDSIGHT_DATABASE_URL", f"{SERVER_URL}/postgres")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def database_url() -> str:
    admin = create_engine(f"{SERVER_URL}/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f"drop database if exists {TEST_DATABASE} with (force)"))
        connection.execute(text(f"create database {TEST_DATABASE}"))
    admin.dispose()

    url = f"{SERVER_URL}/{TEST_DATABASE}"
    with pytest.MonkeyPatch.context() as env:
        env.setenv("HINDSIGHT_SERVICE_TOKEN", SERVICE_TOKEN)
        env.setenv("HINDSIGHT_DATABASE_URL", url)
        get_settings.cache_clear()
        get_engine.cache_clear()
        command.upgrade(Config(toml_file=Path(__file__).parents[1] / "pyproject.toml"), "head")
        get_settings.cache_clear()
        get_engine.cache_clear()

    # stands in for vestry's view, with the same columns
    test_db = create_engine(url)
    with test_db.begin() as connection:
        connection.execute(text(CORE_READ_DDL))
    test_db.dispose()
    return url


@pytest.fixture
async def sessions(database_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(database_url, poolclass=NullPool)
    async with engine.begin() as connection:
        await connection.execute(text(f"truncate {', '.join(TRUNCATED)}"))
        await connection.execute(text("insert into ai.budget (id, blocked) values (1, false)"))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()
