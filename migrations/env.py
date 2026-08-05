"""Alembic environment.

Two things here differ from the generated default, both on purpose.

**The DSN comes from `ct.config.Settings`, not from `alembic.ini`.** One source
of truth for the database URL, and a trading database password never sits in a
committed file. `QD_DB_DSN` — or `QD_DB_DSN_FILE`, the Docker-secret form —
is what compose already supplies to every service.

**There is no `target_metadata`, so `--autogenerate` is unavailable.** That is
the correct trade for this project rather than a gap. The schema is hand-written
SQL with TimescaleDB hypertables, continuous aggregates, compression policies
and partial indexes; SQLAlchemy has no model layer here to reflect, and
autogenerate against a hypertable emits confident, wrong DDL — it sees the
chunk tables Timescale manages and proposes dropping them. Migrations here are
written by hand, which is also what "schema changes on a live trading DB need to
be reversible" actually requires: a human deciding what the downgrade is.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Importable as `ct.*` when alembic is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False is load-bearing, not a style preference.
    # fileConfig() defaults to True, which silently switches OFF every logger
    # created before this point — and `qd migrate` runs inside the same process
    # as the rest of the CLI, so applying a migration would leave the engine's
    # and the risk module's loggers dead for the remainder of the run.
    #
    # Found by a test: after test_migrations.py loaded this file, an unrelated
    # test asserting that a malformed password hash gets logged saw no records
    # at all. `ct.web.security` had been disabled by a migration config.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = None


def _dsn() -> str:
    """The database URL, translated for SQLAlchemy.

    The translation itself lives in `ct.config.sqlalchemy_dsn` because three
    callers need it and none of them can import this module — Alembic loads
    env.py with a live `context` established, so a plain import raises
    `AttributeError: module 'alembic.context' has no attribute 'config'`.
    """
    from ct.config import Settings, sqlalchemy_dsn

    try:
        return sqlalchemy_dsn(Settings().db_dsn)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def run_migrations_offline() -> None:
    """`--sql` mode: emit DDL to stdout without connecting.

    Worth keeping for exactly one situation — reviewing what a migration will do
    to a live trading database *before* it does it.
    """
    # db/schema.sql draws its section banners with box characters, and a cp1252
    # Windows console raises UnicodeEncodeError *inside the write* — so the
    # review command fails having produced a truncated file, which is the worst
    # possible outcome for a command whose entire job is showing you the truth
    # before you touch a live database. `qd migrate --sql` is already protected
    # by _survive_a_narrow_console(); this covers a bare `alembic ... --sql`.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        import contextlib as _ctx
        with _ctx.suppress(ValueError, OSError):
            reconfigure(errors="replace")

    context.configure(
        url=_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _dsn()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # DDL in a transaction. Postgres supports transactional DDL, so a
            # migration that fails halfway rolls back rather than leaving the
            # schema in a state that is neither the old one nor the new one.
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
