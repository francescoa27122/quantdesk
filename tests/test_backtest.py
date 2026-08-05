"""Backtest-engine and indicator tests.

`test_no_lookahead` and `test_indicators_are_causal` are the two tests that
make every backtest number in this project believable. If either ever fails,
stop trusting all reported performance until it passes again.
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ct.backtest.engine import BacktestEngine, CostModel
from ct.backtest.metrics import (
    build_report,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from ct.backtest.walkforward import (
    monte_carlo,
    param_grid,
    purged_walk_forward_splits,
    walk_forward,
)
from ct.models import Bar
from ct.risk.engine import RiskLimits
from ct.strategies import REGISTRY
from ct.strategies.base import Context, Strategy
from ct.strategies.indicators import (
    adx,
    atr,
    bollinger,
    donchian,
    ema,
    macd,
    rsi,
    sma,
    supertrend,
    zscore,
)
from ct.strategies.trend import EmaCrossStrategy

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def make_bars(n: int = 1200, seed: int = 42, trend: float = 0.0002,
              vol: float = 0.02, symbol: str = "BTC/USD") -> list[Bar]:
    """Geometric random walk with a drift. Deterministic seed on purpose."""
    rng = np.random.default_rng(seed)
    price = 10_000.0
    bars: list[Bar] = []
    for i in range(n):
        ret = trend + vol * rng.standard_normal()
        new = max(1.0, price * math.exp(ret))
        hi = max(price, new) * (1 + abs(rng.standard_normal()) * vol * 0.3)
        lo = min(price, new) * (1 - abs(rng.standard_normal()) * vol * 0.3)
        bars.append(Bar(START + timedelta(hours=4 * i), price, hi, lo, new,
                        float(rng.uniform(50, 500)), symbol, "4h"))
        price = new
    return bars


# ── the two tests that matter most ───────────────────────────────────────

def test_indicators_are_causal():
    """Appending future data must not change any historical value.

    This is the strongest available statement that an indicator cannot see
    forward, and it catches the whole class of `.shift()` and centred-window
    bugs that produce fantastic backtests.
    """
    bars = make_bars(400)
    h = np.array([b.high for b in bars])
    l = np.array([b.low for b in bars])
    c = np.array([b.close for b in bars])

    cut = 300
    funcs = {
        "sma": lambda H, L, C: sma(C, 20),
        "ema": lambda H, L, C: ema(C, 20),
        "rsi": lambda H, L, C: rsi(C, 14),
        "atr": lambda H, L, C: atr(H, L, C, 14),
        "adx": lambda H, L, C: adx(H, L, C, 14),
        "zscore": lambda H, L, C: zscore(C, 20),
        "macd_hist": lambda H, L, C: macd(C)[2],
        "bb_lower": lambda H, L, C: bollinger(C, 20)[2],
        "donchian_up": lambda H, L, C: donchian(H, L, 20)[0],
        "supertrend_dir": lambda H, L, C: supertrend(H, L, C)[1],
    }
    for name, fn in funcs.items():
        short = fn(h[:cut], l[:cut], c[:cut])
        full = fn(h, l, c)[:cut]
        assert np.allclose(np.nan_to_num(short, nan=-999),
                           np.nan_to_num(full, nan=-999), atol=1e-9), \
            f"{name} is NOT causal: future bars changed its history"


def test_no_lookahead_in_the_engine():
    """A strategy that peeks at ctx beyond ctx.index cannot: the arrays end
    there. Assert the Context slice length equals index+1 on every call."""
    seen: list[tuple[int, int]] = []

    class Peeker(Strategy):
        name = "peeker"
        family = "trend"
        DEFAULTS: dict = {}

        @property
        def warmup(self) -> int:
            return 5

        def on_bar(self, ctx: Context):
            seen.append((ctx.index, len(ctx.close)))
            assert len(ctx.close) == ctx.index + 1
            assert len(ctx.high) == ctx.index + 1
            assert ctx.close[-1] == ctx.price
            return []

    bars = make_bars(200)
    BacktestEngine(Peeker(), timeframe="4h").run(bars)
    assert seen, "strategy was never called"
    assert all(n == i + 1 for i, n in seen)


def test_orders_fill_on_the_next_bar_never_the_current_one():
    """A fill at the signal bar's close is a 100%-accurate crystal ball."""
    class BuyOnce(Strategy):
        name = "buy_once"
        family = "trend"
        DEFAULTS: dict = {}
        fired = False

        @property
        def warmup(self) -> int:
            return 10

        def on_bar(self, ctx: Context):
            if BuyOnce.fired or not ctx.position.is_flat:
                return []
            BuyOnce.fired = True
            return [self._signal(ctx, 1, atr=ctx.price * 0.02,
                                 stop=ctx.price * 0.95,
                                 target=ctx.price * 1.20)]

    BuyOnce.fired = False
    bars = make_bars(300, seed=7)
    limits = RiskLimits(prefer_maker=False, min_edge_bps=0.0)
    res = BacktestEngine(BuyOnce(), limits, timeframe="4h").run(bars)
    assert res.trades or res.signals
    for sig, dec in res.signals:
        if dec.approved:
            # The signal bar and any resulting fill must not share a timestamp.
            fill_ts = [t.entry_ts for t in res.trades]
            assert sig.ts not in fill_ts


# ── engine behaviour ─────────────────────────────────────────────────────

def test_all_registered_strategies_run_without_crashing():
    bars = make_bars(1500, seed=11)
    for name, cls in REGISTRY.items():
        strat = cls()
        if len(bars) < strat.warmup + 2:
            pytest.skip(f"{name} needs more bars than the fixture provides")
        res = BacktestEngine(strat, timeframe="4h").run(bars)
        # S2-3: metrics start after warm-up; the full curve stays for plotting.
        assert res.report.n_periods == len(bars) - strat.warmup
        assert len(res.equity_curve) == len(bars)
        assert res.equity_curve[0] > 0


def test_fees_reduce_returns_monotonically():
    """Sanity on the cost model — and a reminder of §0.1: at these fee levels
    the cost model is not a detail, it is the result."""
    bars = make_bars(1500, seed=3, trend=0.0008)
    limits = RiskLimits(prefer_maker=False, min_edge_bps=0.0)
    cheap = BacktestEngine(EmaCrossStrategy(), limits,
                           CostModel(maker_fee_bps=0, taker_fee_bps=0,
                                     half_spread_bps=0, impact_k=0.0),
                           timeframe="4h").run(bars)
    dear = BacktestEngine(EmaCrossStrategy(), limits,
                          CostModel(maker_fee_bps=25, taker_fee_bps=40,
                                    half_spread_bps=2, impact_k=0.1),
                          timeframe="4h").run(bars)
    assert dear.report.total_return <= cheap.report.total_return
    assert dear.report.total_fees > 0
    assert cheap.report.total_fees == 0


def test_engine_refuses_a_run_shorter_than_warmup():
    with pytest.raises(ValueError, match="warmup"):
        BacktestEngine(EmaCrossStrategy(), timeframe="4h").run(make_bars(30))


def test_rejections_are_counted_and_named():
    bars = make_bars(1500, seed=5)
    limits = RiskLimits(max_gross_exposure=0.0001)   # reject nearly everything
    res = BacktestEngine(EmaCrossStrategy(), limits, timeframe="4h").run(bars)
    if res.rejections:
        assert all(k.isupper() for k in res.rejections)
        assert "no signals" not in res.rejection_summary()


def test_benchmark_is_net_of_fees():
    bars = make_bars(500, seed=9, trend=0.001)
    res = BacktestEngine(EmaCrossStrategy(), timeframe="4h").run(bars)
    raw = bars[-1].close / bars[0].close - 1.0
    assert res.report.benchmark_return < raw   # holding is not free either


# ── metrics ──────────────────────────────────────────────────────────────

def test_sharpe_annualizes_with_365():
    rets = np.full(365, 0.001)
    s = sharpe_ratio(rets, "1d")
    assert s > 0
    # Constant returns have zero stdev; use a noisy series for the real check.
    rng = np.random.default_rng(1)
    noisy = 0.001 + 0.01 * rng.standard_normal(3650)
    expected = noisy.mean() / noisy.std(ddof=1) * math.sqrt(365)
    assert sharpe_ratio(noisy, "1d") == pytest.approx(expected, rel=1e-9)


def test_sortino_uses_the_full_sample_in_the_denominator():
    rng = np.random.default_rng(2)
    r = 0.001 + 0.01 * rng.standard_normal(2000)
    downside = np.minimum(r, 0.0)
    dd = math.sqrt(float((downside ** 2).sum()) / len(r))
    expected = r.mean() / dd * math.sqrt(365)
    assert sortino_ratio(r, "1d") == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_and_duration():
    eq = [100, 120, 90, 95, 130, 110]
    mdd, dur = max_drawdown(eq)
    assert mdd == pytest.approx((120 - 90) / 120)
    # peaks are [100,120,120,120,130,130]; below-peak runs are (2,3) and (5),
    # so the longest UNBROKEN underwater stretch is 2, not 3. Index 4 makes a
    # new high and resets the counter.
    assert dur == 2


def test_deflated_sharpe_penalises_a_large_search():
    """The whole point: the same returns, judged against more trials, must
    score lower."""
    rng = np.random.default_rng(4)
    r = 0.0008 + 0.01 * rng.standard_normal(1500)
    one = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=500)
    assert many < one
    assert 0.0 <= many <= 1.0


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01)
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_probabilistic_sharpe_is_a_probability():
    rng = np.random.default_rng(6)
    r = 0.001 + 0.01 * rng.standard_normal(800)
    p = probabilistic_sharpe_ratio(r)
    assert 0.0 <= p <= 1.0


def test_profit_factor_of_zero_when_there_are_no_losses():
    """Infinite profit factor is a sample-size signal, not a skill signal; it
    must not be able to win a parameter sweep."""
    from decimal import Decimal

    from ct.models import ClosedTrade
    t = ClosedTrade("s", "BTC/USD", 1, Decimal(1), START, START,
                    Decimal(100), Decimal(110), Decimal(10), Decimal(0))
    rep = build_report([100.0, 110.0], [t], "1d", initial_equity=100.0)
    assert rep.profit_factor == 0.0


def test_verdict_flags_insufficient_data():
    rep = build_report([100.0, 101.0, 102.0], [], "1d", initial_equity=100.0)
    assert "INSUFFICIENT DATA" in rep.verdict()


# ── walk-forward & Monte Carlo ───────────────────────────────────────────

def test_purged_splits_never_overlap_and_respect_the_embargo():
    splits = purged_walk_forward_splits(1000, n_splits=5, embargo_pct=0.02,
                                        label_horizon=10)
    embargo = int(1000 * 0.02)
    for s in splits:
        assert s.train[1] <= s.test[0] - 10 - embargo
        assert s.train[0] < s.train[1] < s.test[0] < s.test[1]
    # Test folds tile the tail without gaps or overlap.
    for a, b in itertools.pairwise(splits):
        assert a.test[1] == b.test[0]


def test_param_grid_is_the_cartesian_product():
    g = param_grid({"fast": [5, 10], "slow": [20, 30, 40]})
    assert len(g) == 6
    assert {"fast": 10, "slow": 40} in g


def test_walk_forward_reports_the_honest_trial_count():
    bars = make_bars(1400, seed=13, trend=0.0005)
    res = walk_forward(EmaCrossStrategy, bars,
                       {"fast": [10, 20], "slow": [50, 100]},
                       n_splits=3, timeframe="4h")
    assert res.n_trials >= 1
    assert len(res.chosen_params) == len(res.splits)
    # DSR must be computed against every variant tried, not just the winner.
    assert res.oos_report.deflated_sharpe <= res.oos_report.probabilistic_sharpe + 1e-9
    assert "VERDICT" in res.summary()


def test_monte_carlo_is_deterministic_and_block_based():
    rets = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02, 0.01, -0.01] * 5
    a = monte_carlo(rets, n_runs=300, seed=1)
    b = monte_carlo(rets, n_runs=300, seed=1)
    assert a == b, "a risk number that changes when you look at it is not one"
    assert a.p95_max_drawdown >= a.p50_max_drawdown
    assert 0.0 <= a.prob_of_loss <= 1.0
    assert "size against this" in a.summary()


def test_monte_carlo_refuses_a_tiny_sample():
    with pytest.raises(ValueError, match=">=5"):
        monte_carlo([0.01, -0.01], n_runs=10)


# ── strategy hygiene ─────────────────────────────────────────────────────

def test_unknown_parameters_are_rejected():
    """Silently accepting an unknown param lets a walk-forward 'optimise' a
    value the strategy never reads."""
    with pytest.raises(ValueError, match="unknown parameters"):
        EmaCrossStrategy(fastt=10)


def test_every_strategy_declares_a_family_the_risk_engine_understands():
    known = {"trend", "momentum", "breakout", "reversion", "mean_reversion"}
    for name, cls in REGISTRY.items():
        assert cls.family in known, f"{name} has family {cls.family!r}"
        assert cls.name == name


def test_strategies_never_emit_a_signal_before_warmup():
    bars = make_bars(1500, seed=17)
    for name, cls in REGISTRY.items():
        strat = cls()
        emitted: list[int] = []

        original = strat.on_bar

        def wrapped(ctx, _o=original, _e=emitted):
            out = _o(ctx)
            if out:
                _e.append(ctx.index)
            return out

        strat.on_bar = wrapped  # type: ignore[method-assign]
        BacktestEngine(strat, timeframe="4h").run(bars)
        assert all(i >= cls().warmup for i in emitted), \
            f"{name} emitted a signal before warmup"


def test_consecutive_loss_breaker_expires_and_does_not_ban_for_life():
    """Regression: the backtest must model the SAME cooldown production has.

    Without the cooldown the breaker is a lifetime ban inside a backtest while
    `strategies.paused_until` expires in production — two different systems
    wearing the same name, which is the failure this architecture exists to
    prevent.
    """

    limits = RiskLimits(max_consecutive_losses=2, loss_pause_hours=24.0)
    bars = make_bars(1800, seed=23, trend=-0.0004)   # downtrend: force losses
    res = BacktestEngine(REGISTRY["macd_momentum"](), limits,
                         timeframe="4h").run(bars)

    blocked = res.rejections.get("CONSECUTIVE_LOSSES", 0)
    approved_after = [i for i, (_, d) in enumerate(res.signals) if d.approved]
    first_block = next((i for i, (_, d) in enumerate(res.signals)
                        if d.code and d.code.value == "CONSECUTIVE_LOSSES"), None)
    if first_block is not None and blocked:
        # Something must get approved AFTER the first block, or the pause never
        # expired.
        assert any(i > first_block for i in approved_after), \
            "consecutive-loss breaker never released — it is a lifetime ban"


def test_loss_pause_bars_scale_with_the_timeframe():
    """24h is 6 bars on 4h and 1 bar on 1d. A hard-coded bar count would make
    the breaker 24× harsher on one timeframe than another."""
    from ct.risk.sizing import periods_per_year
    for tf, expected in (("4h", 6), ("1d", 1), ("1h", 24)):
        bars_per_hour = periods_per_year(tf) / (365.0 * 24.0)
        assert max(1, round(24.0 * bars_per_hour)) == expected


# ══════════════════════════════════════════════════════════════════════════
# S2-3 (2026-07-29): warm-up bars are excluded from every performance metric.
# The ~200-bar flat prefix depressed return mean and stdev, inflated
# n_periods (and years, under CAGR), and diluted exposure's denominator —
# a systematic bias in the exact numbers the promotion gate reads.
# ══════════════════════════════════════════════════════════════════════════


class _NeverTrades(Strategy):
    name = "never_trades"
    family = "test"
    DEFAULTS: dict = {}

    @property
    def warmup(self) -> int:
        return 200

    def on_bar(self, ctx: Context):
        return []


def test_warmup_bars_are_excluded_from_metrics():
    bars = make_bars(500)
    res = BacktestEngine(_NeverTrades(), timeframe="4h").run(bars)
    assert len(res.equity_curve) == 500, "the full curve stays, for plotting"
    assert res.report.n_periods == 300, (
        "metrics must start where trading can: 500 bars - 200 warmup")


def test_warmup_exclusion_changes_the_annualised_numbers():
    """The same trades over a shorter honest window: CAGR and exposure must
    both move once the dead prefix stops being counted."""
    bars = make_bars(900, trend=0.001)
    limits = RiskLimits(prefer_maker=False, min_edge_bps=0.0,
                        min_order_notional=0.0)
    res = BacktestEngine(EmaCrossStrategy(), limits, timeframe="4h").run(bars)
    warmup = EmaCrossStrategy().warmup
    assert res.report.n_periods == len(bars) - warmup
    if res.report.n_trades:
        # Exposure's denominator is the tradeable window, not the whole file.
        full_window_exposure = (res.report.exposure
                                * res.report.n_periods / len(bars))
        assert res.report.exposure > full_window_exposure


def test_benchmark_starts_at_the_same_bar_as_the_strategy():
    """A benchmark bought at bar 0 gets the warm-up drift the strategy is
    forbidden from trading. Entry must be at the metrics window's start."""
    up = make_bars(400, trend=0.002, vol=0.001, seed=7)
    res = BacktestEngine(_NeverTrades(), timeframe="4h").run(up)
    # 200 warm-up bars of strong drift: bought at bar 0 the benchmark would
    # show roughly double this return. Bought at bar 200 it shows only the
    # post-warmup half. Loose bound, tight enough to catch the off-by-window.
    post_warmup = up[-1].close / up[200].close - 1.0
    full_series = up[-1].close / up[0].close - 1.0
    assert res.report.benchmark_return < (post_warmup + full_series) / 2, (
        "benchmark seems to include the pre-warmup drift")


# ══════════════════════════════════════════════════════════════════════════
# 2026-07-30: `_simulate_fill` dropped `strategy` from its fills — the exact
# PaperBroker defect of 2026-07-28, alive in the backtester. Fills booked
# under "|SYMBOL", so every strategy saw itself flat forever: no exit
# signals, no protective stops, zero closed trades, phantom equity. Found
# when a cadence investigation noticed all 7 strategies closing 0 trades
# on real data while paying fees.
# ══════════════════════════════════════════════════════════════════════════


class _RoundTrip(Strategy):
    """Enters once, exits via its own dir-0 signal three bars later — but
    ONLY if it can see its own open position, which is the point."""

    name = "round_trip"
    family = "test"
    DEFAULTS: dict = {}

    @property
    def warmup(self) -> int:
        return 5

    def on_bar(self, ctx: Context):
        if not ctx.position.is_flat:
            return [self._signal(ctx, 0, atr=ctx.price * 0.02)]
        if ctx.index == self.warmup:
            return [self._signal(ctx, 1, atr=ctx.price * 0.02,
                                 stop=ctx.price * 0.5)]
        return []


def test_single_engine_fills_carry_the_strategy_and_trades_close():
    from ct.risk.engine import RiskLimits

    bars = make_bars(60, trend=0.0, vol=0.005, seed=3)
    limits = RiskLimits(prefer_maker=False, min_edge_bps=0.0,
                        min_order_notional=0.0)
    res = BacktestEngine(_RoundTrip(), limits, timeframe="4h").run(bars)
    assert len(res.trades) == 1, (
        "an entered-and-exited position must record exactly one closed trade; "
        "zero means the strategy never saw its own position")
    assert res.trades[0].strategy == "round_trip"
    assert res.report.exposure > 0, (
        "bars_in_market reads the strategy-keyed book; 0 means the fill "
        "landed under the anonymous key")
