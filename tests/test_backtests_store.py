"""BacktestRepository — the reproducibility record and its enforcement.

The property that matters: same (strategy, params, data, code, kind) must
produce the same numbers, and when it does not the repository SHOUTS
(diverged) instead of silently overwriting — nondeterminism invalidates
every backtest figure until explained.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from ct.models import Bar
from ct.store.backtests import BacktestRepository, data_hash, downsample

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Db:
    def __init__(self, existing=None):
        self.existing = existing
        self.executed: list[tuple[str, tuple]] = []

    async def fetchrow(self, sql, *args):
        return self.existing

    async def fetchval(self, sql, *args):
        return None

    async def execute(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))


def _bars(n=5, close=100.0):
    return [Bar(ts=TS + i * timedelta(hours=4), open=close, high=close,
                low=close, close=close + i, volume=1.0, symbol="BTC/USD",
                timeframe="4h") for i in range(n)]


def _record(db, metrics=None):
    return asyncio.run(BacktestRepository(db).record(
        strategy="s", symbols=["BTC/USD"], timeframe="4h",
        start_ts=TS, end_ts=TS + timedelta(days=1), params={"fast": 5},
        params_hash="ph", data_hash="dh", code_git_sha="sha",
        cost_model={"taker_fee_bps": 40.0},
        metrics=metrics or {"sharpe": 1.2, "total_return": 0.1,
                            "n_trades": 12, "max_drawdown": 0.05},
        equity_curve=[100.0, 101.0], kind="single"))


def test_first_record_inserts():
    db = _Db()
    assert _record(db) == "inserted"
    assert any("INSERT INTO backtests" in sql for sql, _ in db.executed)


def test_identical_rerun_is_reproduced_and_touches_nothing():
    db = _Db(existing={"id": 1, "metrics": {"sharpe": 1.2, "total_return": 0.1,
                                            "n_trades": 12,
                                            "max_drawdown": 0.05}})
    assert _record(db) == "reproduced"
    assert db.executed == [], "a reproduced run must not write"


def test_divergence_is_loud_and_updates(caplog):
    db = _Db(existing={"id": 1, "metrics": {"sharpe": 2.0, "total_return": 0.1,
                                            "n_trades": 12,
                                            "max_drawdown": 0.05}})
    assert _record(db) == "diverged"
    assert any("UPDATE backtests" in sql for sql, _ in db.executed)
    assert any("DIVERGENCE" in r.message for r in caplog.records)


def test_existing_metrics_survive_json_string_form():
    db = _Db(existing={"id": 1, "metrics": json.dumps(
        {"sharpe": 1.2, "total_return": 0.1, "n_trades": 12,
         "max_drawdown": 0.05})})
    assert _record(db) == "reproduced"


def test_data_hash_is_stable_and_sensitive():
    a = data_hash({"BTC/USD": _bars()})
    assert a == data_hash({"BTC/USD": _bars()}), "same window, same digest"
    changed = _bars()
    changed[2] = Bar(ts=changed[2].ts, open=100, high=100, low=100,
                     close=999.0, volume=1.0, symbol="BTC/USD", timeframe="4h")
    assert a != data_hash({"BTC/USD": changed}), "one revised candle changes it"


def test_downsample_bounds_and_keeps_endpoints():
    curve = [float(i) for i in range(10_000)]
    out = downsample(curve, limit=400)
    assert len(out) == 400
    assert out[0] == 0.0 and out[-1] == 9999.0
    assert downsample([1.0, 2.0]) == [1.0, 2.0]
