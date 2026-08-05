"""scheduler jobstore table

Revision ID: 0003_apscheduler
Revises: 0002_config_hash
Created: 2026-08-04

Why this change:
    ARCHITECTURE.md Merge 5 specifies "in-process AsyncIOScheduler with a
    Postgres jobstore". APScheduler's SQLAlchemyJobStore creates its own table
    on first use with `create_all(checkfirst=True)`.

    In a database that is otherwise migration-managed, a library quietly doing
    DDL at process start is the thing Alembic exists to prevent: the schema
    would then depend on which version of a dependency happened to boot first,
    and `qd migrate --sql` would not show it. Declaring it here makes the table
    reviewable and versioned like everything else.

    The definition is APScheduler 3.11's, reproduced exactly — id VARCHAR(191),
    next_run_time double precision, job_state bytea, plus the index the store
    uses to find due jobs. 191 rather than 255 is APScheduler's own choice (a
    MySQL utf8mb4 index-length limit); matching it matters because the store
    compares against its own schema, not because Postgres cares.

    `IF NOT EXISTS` throughout, so a database where APScheduler already
    auto-created the table adopts it rather than failing.

Downgrade safety:
    Drops the table, losing any persisted job schedule. Harmless in practice:
    the engine re-adds both jobs at every boot with replace_existing=True. What
    is lost is the daily promotion job's next-run time, so a downgrade near
    midnight could cause that day's evaluation to be skipped or repeated — and
    it is idempotent, so repeated is a no-op.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003_apscheduler"
down_revision: str | None = "0002_config_hash"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS apscheduler_jobs (
            id            VARCHAR(191) NOT NULL,
            next_run_time DOUBLE PRECISION,
            job_state     BYTEA NOT NULL,
            CONSTRAINT apscheduler_jobs_pkey PRIMARY KEY (id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_apscheduler_jobs_next_run_time
            ON apscheduler_jobs (next_run_time)
    """))
    op.execute(sa.text(
        "COMMENT ON TABLE apscheduler_jobs IS "
        "'APScheduler job store (ct/engine/scheduler.py). Managed here rather "
        "than by the library''s create_all so the schema stays reviewable.'"))


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_apscheduler_jobs_next_run_time"))
    op.execute(sa.text("DROP TABLE IF EXISTS apscheduler_jobs"))
