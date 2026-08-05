"""The promotion evaluator — `promotion_status`'s first writer.

Until 2026-07-30 that table had **no writer anywhere in the codebase** while
four readers (the `qd arm-live` gate, `qd promotion-status`, the dashboard,
the Home page's framing) treated it as the record of whether a strategy has
earned live capital. The same no-writer disease that produced `risk_events`
(no writer), `resolve()` (no caller) and `strategies` (no writer) — found by
a systematic sweep this time, not by an incident.

ARCHITECTURE §8.3 defines six criteria, "computed by qd-worker nightly".
Three of them need backtest records that do not exist yet (`backtests` has no
writer — Monte-Carlo p95 drawdown, backtest/paper correlation, benchmark
Calmar). This evaluator computes what IS honestly computable today and NAMES
the rest as failures instead of faking them:

* computable now — paper days (from the audit trail of the human enable),
  closed-trade count, and once ≥30 trades exist: per-trade Sharpe annualised
  by the *observed* trade frequency, and max drawdown of the per-trade
  equity path;
* not yet measurable — the three backtest-dependent criteria, each listed in
  `failures[]` with the reason, so `passed` can never drift to True on
  criteria nobody measured.

Runs inside the engine's housekeeping once per UTC day: it is a handful of
SQL reads and one upsert, which does not need `qd-worker`'s ML image. The
Monte-Carlo and correlation parts move to the worker when backtest records
exist (Phase 5), as §8.3 intended.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

MIN_PAPER_DAYS = 90
MIN_TRADES = 30
MIN_SHARPE = 1.0

#: How much worse than the backtest's Monte-Carlo p95 drawdown the paper
#: record may be before criterion 4 fails (ARCHITECTURE §8.3).
MAX_DD_VS_MC = 1.5

#: Criteria that still cannot be measured (they need stored backtest return
#: series aligned with the paper window — Phase 5). Named in full so
#: `qd promotion-status` shows *why* they fail, not just that.
_UNMEASURED = (
    "backtest/paper correlation: not measurable — needs aligned return "
    "series (Phase 5)",
    "beats-benchmark Calmar: not measurable — needs aligned benchmark "
    "window (Phase 5)",
)


def trade_sharpe(returns: list[float], *, first: datetime, last: datetime,
                 ) -> float | None:
    """Sharpe over per-trade returns, annualised by OBSERVED trade frequency.

    There is no per-strategy equity curve (strategies share one cash
    balance), so the honest sample is the closed-trade return series. The
    annualisation factor is how often this strategy actually trades — not a
    bar count it does not have — measured from its own first and last exits.
    """
    if len(returns) < 2:
        return None
    span_days = max((last - first).total_seconds() / 86_400.0, 1.0)
    trades_per_year = len(returns) * 365.0 / span_days
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var) * math.sqrt(trades_per_year)


def max_drawdown_of(returns: list[float]) -> float:
    """Max drawdown of the compounded per-trade equity path, as a fraction."""
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, 1.0 - equity / peak)
    return worst


async def evaluate(db, *, now: datetime) -> list[dict[str, Any]]:
    """Evaluate every strategy currently in paper or live; upsert one row each.

    Returns the rows written (for logging/tests)."""
    strategies = await db.fetch(
        """SELECT name, status, timeframe FROM strategies
           WHERE status IN ('paper', 'live') ORDER BY name""")
    out: list[dict[str, Any]] = []
    for srow in strategies:
        name = srow["name"]
        # The enable is audited; its FIRST occurrence is the paper clock's
        # start and survives later lifecycle churn (strategies.updated_at
        # does not — it moves on every save).
        started = await db.fetchval(
            """SELECT min(ts) FROM audit_log
               WHERE action = 'strategy:paper' AND target = $1""", name)
        paper_days = 0
        if started is not None:
            paper_days = max(0, (now - started).days)

        closed = await db.fetch(
            """SELECT ts, (payload->>'return_pct')::float8 AS ret
               FROM events
               WHERE type = 'portfolio.position_closed'
                 AND payload->>'strategy' = $1
               ORDER BY ts""", name)
        returns = [r["ret"] for r in closed if r["ret"] is not None]
        n_trades = len(closed)

        failures: list[str] = []
        if paper_days < MIN_PAPER_DAYS:
            failures.append(f"paper_days<{MIN_PAPER_DAYS} ({paper_days})")
        if n_trades < MIN_TRADES:
            failures.append(f"paper_trades<{MIN_TRADES} ({n_trades}; below "
                            f"this no metric is meaningful)")

        sharpe = None
        max_dd = None
        if n_trades >= MIN_TRADES and returns:
            sharpe = trade_sharpe(returns, first=closed[0]["ts"],
                                  last=closed[-1]["ts"])
            max_dd = max_drawdown_of(returns)
            if sharpe is None:
                failures.append("paper_sharpe: not computable from this "
                                "return series (zero variance or too short)")
            elif sharpe < MIN_SHARPE:
                failures.append(f"paper_sharpe<{MIN_SHARPE} ({sharpe:.2f})")

        # Criterion 4 — measurable since 2026-07-30, when `backtests` gained
        # a writer: the newest walkforward Monte-Carlo p95 drawdown for this
        # strategy ON ITS OWN TIMEFRAME. Without the timeframe pin, a 1d
        # walkforward record would be compared against 4h paper trading —
        # a mismatched envelope smuggled into the gate.
        tf = srow.get("timeframe") if hasattr(srow, "get") else None
        mc_p95 = await db.fetchval(
            """SELECT (metrics->'montecarlo'->>'p95_max_drawdown')::float8
               FROM backtests
               WHERE strategy = $1 AND metrics ? 'montecarlo'
                 AND ($2::text IS NULL OR timeframe = $2)
               ORDER BY created_at DESC LIMIT 1""", name, tf)
        if mc_p95 is None:
            failures.append(
                "paper_max_dd vs backtest MC p95: not measurable — run "
                "`qd walkforward --strategy "
                f"{name} --montecarlo` to record one")
        elif max_dd is not None and max_dd > MAX_DD_VS_MC * float(mc_p95):
            failures.append(
                f"paper_max_dd>{MAX_DD_VS_MC}x MC p95 "
                f"({max_dd:.1%} vs {float(mc_p95):.1%})")

        failures.extend(_UNMEASURED)
        row = {
            "strategy": name, "passed": not failures,
            "paper_days": paper_days, "paper_trades": n_trades,
            "paper_sharpe": sharpe, "paper_max_dd": max_dd,
            "mc_p95_dd": float(mc_p95) if mc_p95 is not None else None,
            "failures": failures,
        }
        await db.execute(
            """INSERT INTO promotion_status
                   (strategy, evaluated_at, passed, paper_days, paper_trades,
                    paper_sharpe, paper_max_dd, mc_p95_dd, failures)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
               ON CONFLICT (strategy) DO UPDATE SET
                   evaluated_at = EXCLUDED.evaluated_at,
                   passed       = EXCLUDED.passed,
                   paper_days   = EXCLUDED.paper_days,
                   paper_trades = EXCLUDED.paper_trades,
                   paper_sharpe = EXCLUDED.paper_sharpe,
                   paper_max_dd = EXCLUDED.paper_max_dd,
                   mc_p95_dd    = EXCLUDED.mc_p95_dd,
                   failures     = EXCLUDED.failures""",
            name, now, row["passed"], paper_days, n_trades, sharpe, max_dd,
            row["mc_p95_dd"], failures)
        out.append(row)
        log.info("promotion: %s day %d/%d, %d trades — %s", name, paper_days,
                 MIN_PAPER_DAYS, n_trades,
                 "PASSED" if row["passed"] else f"{len(failures)} criteria open")
    return out


__all__ = ["MIN_PAPER_DAYS", "MIN_SHARPE", "MIN_TRADES", "evaluate",
           "max_drawdown_of", "trade_sharpe"]
