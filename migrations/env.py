import asyncio

from alembic import context
from sqlalchemy import text
from sqlalchemy.engine import Connection

from hindsight.db import get_engine
from hindsight.models import SCHEMA, Base


def include_name(name: str | None, type_: str, _: object) -> bool:
    # vestry owns every other schema
    return name == SCHEMA if type_ == "schema" else True


def run_migrations(connection: Connection) -> None:
    connection.execute(text(f"create schema if not exists {SCHEMA}"))
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


async def main() -> None:
    engine = get_engine()
    async with engine.connect() as connection:
        await connection.run_sync(run_migrations)
        await connection.commit()
    await engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline migrations aren't supported")

asyncio.run(main())
