"""record which config revision produced a backtest

Revision ID: 0002_config_hash
Revises: 0001_baseline
Created: 2026-08-04

Why this change:
    ARCHITECTURE.md §2: "strategy params from versioned YAML that is hashed into
    every backtest record for reproducibility."

    `params_hash` already covers the parameters themselves, and that is the half
    that matters for the reproducibility invariant the UNIQUE constraint rests
    on. What it does not answer is "which revision of config/strategies.yml was
    this?" — and that is the question you ask when a result looks wrong six
    weeks later. Two different config files can produce the same params for one
    strategy while disagreeing about every other strategy, the weights, and the
    symbol lists.

    Both columns are nullable, and deliberately NOT part of the UNIQUE
    constraint. Adding them to it would change the identity of a backtest run:
    the same code, data and params would stop deduplicating across a config
    reformat, and the divergence check in BacktestRepository.record — the one
    that shouts when identical inputs produce different numbers — would go
    quiet. Provenance is metadata about a run, not part of what makes it that
    run.

Downgrade safety:
    Drops both columns, losing the provenance of historical rows. No metric,
    equity curve or reproducibility hash is touched, so nothing that feeds the
    promotion gate is affected.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002_config_hash"
down_revision: str | None = "0001_baseline"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # IF NOT EXISTS because the baseline may have just built the schema from a
    # future schema.sql on a fresh install. Idempotent beats ordering-dependent
    # on a database somebody is trading against.
    op.execute(sa.text(
        "ALTER TABLE backtests ADD COLUMN IF NOT EXISTS config_hash text"))
    op.execute(sa.text(
        "ALTER TABLE backtests ADD COLUMN IF NOT EXISTS config_version integer"))
    op.execute(sa.text(
        "COMMENT ON COLUMN backtests.config_hash IS "
        "'sha256 of config/strategies.yml that produced this run; NULL for "
        "runs recorded before the config layer existed, or with no config file'"))


def downgrade() -> None:
    op.execute(sa.text("ALTER TABLE backtests DROP COLUMN IF EXISTS config_version"))
    op.execute(sa.text("ALTER TABLE backtests DROP COLUMN IF EXISTS config_hash"))
