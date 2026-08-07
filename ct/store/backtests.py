"""Backtest records — the `backtests` table's first writer.

The schema promised this from day one ("Reproducibility is enforced, not
hoped for: same three hashes => same result, or there is a bug") and then no
code ever wrote a row: every run evaporated, two runs could not be compared,
and three of the promotion gate's six criteria were permanently unmeasurable.

The reproducibility contract is enforced here, not merely stored: recording a
run whose (strategy, params_hash, data_hash, code_git_sha, kind) tuple
already exists compares the key metrics —
* identical  → "reproduced", row untouched (idempotent re-runs);
* different  → "diverged", logged as an ERROR and the row updated, because
  same code + same data + same params producing different numbers means
  nondeterminism somewhere, and that invalidates every backtest number until
  it is explained.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from ct.store.db import Database

log = logging.getLogger(__name__)

#: Metrics compared for the reproducibility check, with absolute tolerance.
_REPRO_KEYS = (("total_return", 1e-9), ("sharpe", 1e-6), ("n_trades", 0),
               ("max_drawdown", 1e-9))


def data_hash(bars_by_symbol: dict[str, list[Any]]) -> str:
    """Stable digest of the exact candle window a run consumed.

    Hashes every bar's timestamp and close per symbol, so the same window
    re-loaded from Postgres or from a CSV produces the same digest — and a
    single revised candle produces a different one.
    """
    h = hashlib.sha256()
    for symbol in sorted(bars_by_symbol):
        for bar in bars_by_symbol[symbol]:
            h.update(f"{symbol}|{bar.timeframe}|{bar.ts.isoformat()}|"
                     f"{bar.close!r}\n".encode())
    return h.hexdigest()


def downsample(curve: list[float], limit: int = 400) -> list[float]:
    """Keep the stored equity curve bounded; first and last points always kept."""
    if len(curve) <= limit:
        return list(curve)
    step = (len(curve) - 1) / (limit - 1)
    return [curve[round(i * step)] for i in range(limit)]


class BacktestRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(
        self, *, strategy: str, symbols: list[str], timeframe: str,
        start_ts: datetime, end_ts: datetime,
        params: dict[str, Any], params_hash: str,
        data_hash: str, code_git_sha: str, cost_model: dict[str, Any],
        metrics: dict[str, Any], equity_curve: list[float] | None = None,
        n_trials: int = 1, kind: str = "single",
        config_hash: str | None = None, config_version: int | None = None,
    ) -> str:
        """Insert or verify a run. Returns 'inserted' | 'reproduced' | 'diverged'."""
        existing = await self.db.fetchrow(
            """SELECT id, metrics FROM backtests
               WHERE strategy = $1 AND params_hash = $2 AND data_hash = $3
                 AND code_git_sha = $4 AND kind = $5""",
            strategy, params_hash, data_hash, code_git_sha, kind)
        if existing is not None:
            old = existing["metrics"]
            old = json.loads(old) if isinstance(old, str) else (old or {})
            diverged = []
            for key, tol in _REPRO_KEYS:
                a, b = old.get(key), metrics.get(key)
                if a is None or b is None:
                    continue
                if abs(float(a) - float(b)) > tol:
                    diverged.append(f"{key}: {a} -> {b}")
            if not diverged:
                return "reproduced"
            # Same code, same data, same params, different numbers. Every
            # backtest figure is suspect until this is explained.
            log.error("BACKTEST DIVERGENCE for %s (%s): same three hashes, "
                      "different results — %s. Nondeterminism somewhere; "
                      "stored row updated, previous values shown here only.",
                      strategy, kind, "; ".join(diverged))
            await self.db.execute(
                """UPDATE backtests SET metrics = $2::jsonb,
                       equity_curve = $3::jsonb, created_at = now()
                   WHERE id = $1""",
                existing["id"], json.dumps(metrics, default=str),
                json.dumps(equity_curve) if equity_curve is not None else None)
            return "diverged"

        # config_hash/config_version answer "which revision of
        # config/strategies.yml was this?" — deliberately NOT part of the UNIQUE
        # constraint above. Adding them would change a run's identity, so the
        # same code+data+params would stop deduplicating across a config
        # reformat and the divergence check would go quiet. Provenance is
        # metadata about a run, not part of what makes it that run.
        await self.db.execute(
            """INSERT INTO backtests
                   (strategy, symbols, timeframe, start_ts, end_ts, params,
                    params_hash, data_hash, code_git_sha, cost_model, metrics,
                    equity_curve, n_trials, kind, config_hash, config_version)
               VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10::jsonb,$11::jsonb,
                       $12::jsonb,$13,$14,$15,$16)""",
            strategy, symbols, timeframe, start_ts, end_ts,
            json.dumps(params, default=str), params_hash, data_hash,
            code_git_sha, json.dumps(cost_model, default=str),
            json.dumps(metrics, default=str),
            json.dumps(equity_curve) if equity_curve is not None else None,
            n_trials, kind, config_hash, config_version)
        return "inserted"

    async def latest_mc_p95_dd(self, strategy: str) -> float | None:
        """Newest Monte-Carlo p95 drawdown for a strategy, or None.

        This is what makes promotion criterion 4 (paper max drawdown vs
        backtest MC p95) measurable at all."""
        val = await self.db.fetchval(
            """SELECT (metrics->'montecarlo'->>'p95_max_drawdown')::float8
               FROM backtests
               WHERE strategy = $1 AND metrics ? 'montecarlo'
               ORDER BY created_at DESC LIMIT 1""", strategy)
        return float(val) if val is not None else None


__all__ = ["BacktestRepository", "data_hash", "downsample"]
