"""Promotion evaluator tests.

The load-bearing property: `passed` can NEVER be True while any criterion is
unmeasured. Three of §8.3's six criteria need backtest records that do not
exist yet, so today the honest ceiling is failed-with-reasons — a gate that
passes on criteria nobody measured is a green light that cannot turn red.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from ct.engine import promotion

NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)


class _Db:
    def __init__(self, *, strategies=None, started=None, closed=None,
                 mc_p95=None):
        self.strategies = strategies or []
        self.started = started
        self.closed = closed or []
        self.mc_p95 = mc_p95
        self.written: list[tuple] = []

    async def fetch(self, sql, *args):
        if "FROM strategies" in sql:
            return self.strategies
        if "position_closed" in sql:
            return self.closed
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        if "audit_log" in sql:
            return self.started
        if "montecarlo" in sql:
            return self.mc_p95
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        self.written.append(args)


def _trades(n, ret=0.02, span_days=60):
    step = timedelta(days=span_days / max(n - 1, 1))
    # Vary around `ret`: identical returns have zero variance and therefore,
    # correctly, no Sharpe at all.
    return [{"ts": NOW - timedelta(days=span_days) + i * step,
             "ret": ret + (0.005 if i % 2 else -0.004)}
            for i in range(n)]


def test_day_one_writes_an_honest_failing_row():
    db = _Db(strategies=[{"name": "donchian_breakout", "status": "paper"}],
             started=NOW - timedelta(hours=10))
    rows = asyncio.run(promotion.evaluate(db, now=NOW))
    assert len(rows) == 1 and len(db.written) == 1
    row = rows[0]
    assert row["passed"] is False
    assert row["paper_days"] == 0 and row["paper_trades"] == 0
    assert row["paper_sharpe"] is None, "no metric below 30 trades"
    assert any("paper_days<90" in f for f in row["failures"])
    assert any("paper_trades<30" in f for f in row["failures"])
    assert sum("not measurable" in f for f in row["failures"]) == 3


def test_never_evaluated_strategies_are_not_rows():
    db = _Db(strategies=[])
    assert asyncio.run(promotion.evaluate(db, now=NOW)) == []
    assert db.written == []


def test_enough_trades_computes_sharpe_and_drawdown():
    db = _Db(strategies=[{"name": "s", "status": "paper"}],
             started=NOW - timedelta(days=95),
             closed=_trades(40, ret=0.02, span_days=80))
    row = asyncio.run(promotion.evaluate(db, now=NOW))[0]
    assert row["paper_trades"] == 40
    assert row["paper_sharpe"] is not None and row["paper_sharpe"] != 0
    assert row["paper_max_dd"] == 0.0, "monotonic gains have no drawdown"
    assert not any("paper_days" in f for f in row["failures"])
    assert not any("paper_trades" in f for f in row["failures"])


def test_passed_is_never_true_while_criteria_are_unmeasured():
    """Even a perfect paper record cannot pass until the backtest-dependent
    criteria are measurable. The gate fails closed, with reasons."""
    db = _Db(strategies=[{"name": "s", "status": "paper"}],
             started=NOW - timedelta(days=120),
             closed=_trades(60, ret=0.03, span_days=100))
    row = asyncio.run(promotion.evaluate(db, now=NOW))[0]
    assert row["passed"] is False
    assert all("not measurable" in f for f in row["failures"]), row["failures"]


def test_trade_sharpe_annualises_by_observed_frequency():
    first, last = NOW - timedelta(days=36.5), NOW
    rets = [0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.02, 0.01, 0.02, 0.01]
    s = promotion.trade_sharpe(rets, first=first, last=last)
    # 10 trades in 36.5 days = 100/yr; per-trade mean/std ~ 0.62 -> ~6.2
    assert s is not None and 4.0 < s < 9.0
    flat = promotion.trade_sharpe([0.01, 0.01], first=first, last=last)
    assert flat is None, "zero variance has no Sharpe"


def test_max_drawdown_of_a_known_path():
    # 1.0 -> 1.10 -> 0.55 -> 0.66: worst peak-to-trough = 50%
    dd = promotion.max_drawdown_of([0.10, -0.50, 0.20])
    assert abs(dd - 0.5) < 1e-9


def test_mc_criterion_measures_once_a_walkforward_record_exists():
    """Criterion 4 goes from 'not measurable' to a real comparison the moment
    a Monte-Carlo-bearing walkforward is recorded."""
    base = dict(strategies=[{"name": "s", "status": "paper"}],
                started=NOW - timedelta(days=100),
                closed=_trades(40, ret=0.02, span_days=90))
    without = asyncio.run(promotion.evaluate(_Db(**base), now=NOW))[0]
    assert any("qd walkforward" in f for f in without["failures"])
    assert without["mc_p95_dd"] is None

    ok = asyncio.run(promotion.evaluate(_Db(**base, mc_p95=0.30), now=NOW))[0]
    assert ok["mc_p95_dd"] == 0.30
    assert not any("MC p95" in f for f in ok["failures"]), ok["failures"]

    # A paper drawdown worse than 1.5x the MC p95 fails with the numbers.
    deep = dict(base, closed=[
        *_trades(35, ret=0.02, span_days=80),
        {"ts": NOW - timedelta(days=2), "ret": -0.60},
    ])
    bad = asyncio.run(promotion.evaluate(_Db(**deep, mc_p95=0.10), now=NOW))[0]
    assert any("paper_max_dd>1.5x MC p95" in f for f in bad["failures"])
