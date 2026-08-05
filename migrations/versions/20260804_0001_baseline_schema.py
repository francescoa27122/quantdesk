"""baseline — adopt the existing schema

Revision ID: 0001_baseline
Revises:
Created: 2026-08-04

Why this change:
    Alembic is arriving after the database already exists. There is a live
    deployment whose schema was created by Postgres running db/schema.sql from
    /docker-entrypoint-initdb.d, and there are fresh installs that will do the
    same. Neither has ever had an alembic_version table.

    So this revision must be a no-op against a database that already has the
    schema, and must create it against one that does not. It detects which case
    it is in rather than assuming, because assuming wrong in the first case
    means `CREATE TABLE` errors on a live trading database, and assuming wrong
    in the second leaves an empty database stamped as up to date.

    That makes `qd migrate` safe to run on the existing NAS deployment with no
    special ceremony, which is the entire point — a migration tool nobody dares
    run on the real database is not a migration tool.

Downgrade safety:
    Refuses. Downgrading the baseline means dropping every table in the system,
    including `fills`, from which `positions` is reproducible and without which
    it is not. There is no situation where that is the right automated action;
    if you genuinely want an empty database, drop it by hand and say so out
    loud.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from alembic import context, op

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None

# Any one of these existing means the schema is already present. `accounts` is
# the probe because it is small, central, and created early in schema.sql — if
# it exists, schema.sql ran.
_PROBE_TABLE = "accounts"

_SCHEMA_SQL = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


def _schema_present(conn: sa.Connection) -> bool:
    return bool(conn.execute(sa.text(
        "SELECT to_regclass('public.' || :t) IS NOT NULL"),
        {"t": _PROBE_TABLE}).scalar())


def _read_schema_sql() -> str:
    if not _SCHEMA_SQL.is_file():
        raise RuntimeError(
            f"{_SCHEMA_SQL} is missing and the schema is not present. "
            "The baseline cannot build a database it cannot read.")
    # Executed as one script. schema.sql contains CREATE EXTENSION, hypertable
    # calls and policies that must run in file order; splitting on semicolons
    # would break the ones containing them inside function bodies and strings.
    return _SCHEMA_SQL.read_text(encoding="utf-8")


def upgrade() -> None:
    # Offline (`--sql`) mode has no connection, so the probe below is
    # impossible: op.get_bind() returns None and .execute() raises. That is not
    # a hypothetical — `qd migrate --sql` is the "review it before running it
    # against a live trading database" path, and it crashed here on the first
    # attempt. Render the fresh-install branch instead, and say so in the output
    # rather than letting somebody assume the DDL applies to their database.
    if context.is_offline_mode():
        op.execute(sa.text(
            "-- BASELINE, FRESH-INSTALL BRANCH.\n"
            "-- This migration is conditional and cannot be rendered exactly\n"
            "-- offline: against a database that already has the schema (one\n"
            "-- built by db/schema.sql via docker-entrypoint-initdb.d) it is a\n"
            "-- NO-OP that only stamps alembic_version. What follows is what\n"
            "-- runs against an EMPTY database. Do not apply it to a populated\n"
            "-- one.\nSELECT 1"))
        op.execute(sa.text(_read_schema_sql()))
        return

    conn = op.get_bind()
    if _schema_present(conn):
        # Already built by initdb. Adopting it is the whole job.
        op.execute(sa.text("SELECT 1"))
        return

    op.execute(sa.text(_read_schema_sql()))


def downgrade() -> None:
    raise RuntimeError(
        "refusing to downgrade the baseline: this would drop every table, "
        "including `fills`, which is the only record `positions` can be "
        "rebuilt from. Drop the database by hand if that is really what you "
        "want.")
