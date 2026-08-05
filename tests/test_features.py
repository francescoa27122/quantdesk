"""The indicator cache — Phase 2 item 4, the fix for B2.

This file exists to justify one optimisation. `FeatureSet` computes each
indicator **once over the whole series** and hands each bar a view truncated at
that bar, instead of recomputing over a growing prefix. That is only correct if

    f(full)[i] == f(full[:i + 1])[i]      for every i

— i.e. if every indicator is causal. So the load-bearing tests here are not
about speed at all:

1. **Causality, for every indicator in the module**, not a chosen sample. The
   pre-existing `test_indicators_are_causal` covered 10 of 17; this covers all
   of them, because the optimisation now depends on the property universally.
2. **Signal equivalence** — each of the seven strategies emits byte-identical
   signals with the cache and without it, over the same bars. If this ever
   fails, the cache is changing what the system trades, which is the only
   outcome that would make the speedup not worth having.
3. **A view cannot see past its bar** — the structural look-ahead guarantee
   must survive the refactor.

The speed test is last and is the least important thing here.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from ct.models import Position, Regime
from ct.strategies import REGISTRY
from ct.strategies import indicators as ind
from ct.strategies.base import Context
from ct.strategies.features import FeatureSet

N = 400


def _series(n: int = N, seed: int = 7):
    """Deterministic OHLCV without depending on the stdlib's RNG internals."""
    x = seed
    o = np.empty(n)
    h = np.empty(n)
    lo = np.empty(n)
    c = np.empty(n)
    v = np.empty(n)
    px = 30_000.0
    for i in range(n):
        x = (1103515245 * x + 12345) % (2**31)
        u = x / 2**31
        px *= math.exp(0.0008 + 0.02 * (u - 0.5))
        o[i] = px * (1 + 0.002 * (u - 0.5))
        h[i] = px * (1 + 0.012 * u)
        lo[i] = px * (1 - 0.012 * (1 - u))
        c[i] = px
        v[i] = 100 + 400 * u
    h = np.maximum.reduce([h, o, c])
    lo = np.minimum.reduce([lo, o, c])
    return o, h, lo, c, v


O, H, L, C, V = _series()


# ── 1. causality, for every indicator ────────────────────────────────────


ALL_INDICATORS = {
    "sma":          lambda h, l, c, v: ind.sma(c, 20),
    "ema":          lambda h, l, c, v: ind.ema(c, 20),
    "rma":          lambda h, l, c, v: ind.rma(c, 14),
    "rsi":          lambda h, l, c, v: ind.rsi(c, 14),
    "roc":          lambda h, l, c, v: ind.roc(c, 12),
    "rolling_std":  lambda h, l, c, v: ind.rolling_std(c, 20),
    "zscore":       lambda h, l, c, v: ind.zscore(c, 20),
    "realized_vol": lambda h, l, c, v: ind.realized_vol(c, 30),
    "true_range":   lambda h, l, c, v: ind.true_range(h, l, c),
    "atr":          lambda h, l, c, v: ind.atr(h, l, c, 14),
    "adx":          lambda h, l, c, v: ind.adx(h, l, c, 14),
    "macd_line":    lambda h, l, c, v: ind.macd(c)[0],
    "macd_signal":  lambda h, l, c, v: ind.macd(c)[1],
    "macd_hist":    lambda h, l, c, v: ind.macd(c)[2],
    "bb_upper":     lambda h, l, c, v: ind.bollinger(c, 20)[0],
    "bb_mid":       lambda h, l, c, v: ind.bollinger(c, 20)[1],
    "bb_lower":     lambda h, l, c, v: ind.bollinger(c, 20)[2],
    "kc_upper":     lambda h, l, c, v: ind.keltner(h, l, c)[0],
    "kc_lower":     lambda h, l, c, v: ind.keltner(h, l, c)[2],
    "donchian_up":  lambda h, l, c, v: ind.donchian(h, l, 20)[0],
    "donchian_mid": lambda h, l, c, v: ind.donchian(h, l, 20)[1],
    "donchian_dn":  lambda h, l, c, v: ind.donchian(h, l, 20)[2],
    "supertrend":   lambda h, l, c, v: ind.supertrend(h, l, c)[0],
    "supertrend_d": lambda h, l, c, v: ind.supertrend(h, l, c)[1],
    "squeeze_on":   lambda h, l, c, v: ind.squeeze_on(h, l, c).astype(float),
    "sma_volume":   lambda h, l, c, v: ind.sma(v, 20),
}


def test_every_public_indicator_is_covered_here():
    """A causality test that silently stops covering a new indicator is how
    this optimisation would quietly become invalid."""
    public = {n for n in ind.__all__ if not n.startswith("_")}
    covered = {k.split("_")[0] for k in ALL_INDICATORS} | {
        "macd", "bollinger", "keltner", "donchian", "supertrend",
        "rolling_std", "realized_vol", "true_range", "squeeze_on", "sma_volume"}
    missing = {n for n in public if n not in covered and n not in {"Arr"}}
    assert not missing, f"indicators with no causality test: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(ALL_INDICATORS))
def test_indicator_is_causal(name):
    """Appending future bars must not change any historical value.

    This is the property the whole cache rests on: if it holds, computing over
    the full series and slicing is identical to recomputing per prefix.
    """
    fn = ALL_INDICATORS[name]
    cut = 300
    short = fn(H[:cut], L[:cut], C[:cut], V[:cut])
    full = fn(H, L, C, V)[:cut]
    assert np.allclose(np.nan_to_num(short, nan=-999.0),
                       np.nan_to_num(full, nan=-999.0), atol=1e-9), \
        f"{name} changed a historical value when future bars were appended"


@pytest.mark.parametrize("name", sorted(ALL_INDICATORS))
def test_causality_holds_at_every_prefix_not_just_one(name):
    """One cut point can pass by luck. This walks a range of them."""
    fn = ALL_INDICATORS[name]
    full = fn(H, L, C, V)
    for cut in (60, 97, 150, 233, 399):
        short = fn(H[:cut], L[:cut], C[:cut], V[:cut])
        assert np.allclose(np.nan_to_num(short, nan=-999.0),
                           np.nan_to_num(full[:cut], nan=-999.0), atol=1e-9), \
            f"{name} is not causal at prefix {cut}"


# ── 2. the cache returns what the direct call returns ────────────────────


def test_feature_set_matches_the_direct_call_at_every_index():
    fs = FeatureSet(O, H, L, C, V)
    pairs = [
        (fs.ema(20), ind.ema(C, 20)),
        (fs.atr(14), ind.atr(H, L, C, 14)),
        (fs.adx(14), ind.adx(H, L, C, 14)),
        (fs.rsi(14), ind.rsi(C, 14)),
        (fs.zscore(20), ind.zscore(C, 20)),
        (fs.sma_volume(20), ind.sma(V, 20)),
        (fs.macd()[2], ind.macd(C)[2]),
        (fs.bollinger(20)[2], ind.bollinger(C, 20)[2]),
        (fs.donchian(20)[0], ind.donchian(H, L, 20)[0]),
        (fs.supertrend()[1], ind.supertrend(H, L, C)[1]),
        (fs.keltner()[0], ind.keltner(H, L, C)[0]),
        (fs.squeeze_on().astype(float), ind.squeeze_on(H, L, C).astype(float)),
    ]
    for got, want in pairs:
        assert np.array_equal(np.nan_to_num(got, nan=-999.0),
                              np.nan_to_num(want, nan=-999.0))


def test_the_cache_actually_caches():
    fs = FeatureSet(O, H, L, C, V)
    for _ in range(50):
        fs.ema(20)
        fs.atr(14)
    assert fs.misses == 2, f"{fs.misses} computations for 2 distinct indicators"
    assert fs.hits == 98


def test_different_parameters_are_different_entries():
    """A cache that collides on parameters returns the wrong indicator, which
    is worse than no cache."""
    fs = FeatureSet(O, H, L, C, V)
    assert not np.array_equal(np.nan_to_num(fs.ema(10)),
                              np.nan_to_num(fs.ema(50)))
    assert fs.misses == 2


def test_volume_and_price_smas_do_not_share_a_key():
    """`sma` and `sma_volume` run the same function over different series. One
    shared key would silently hand a strategy the volume average as a price."""
    fs = FeatureSet(O, H, L, C, V)
    assert not np.array_equal(np.nan_to_num(fs.sma(20)),
                              np.nan_to_num(fs.sma_volume(20)))


# ── 3. a view cannot see past its bar ────────────────────────────────────


@pytest.mark.parametrize("index", [0, 1, 50, 199, N - 1])
def test_a_feature_view_cannot_see_past_the_current_bar(index):
    """The structural look-ahead guarantee, preserved through the refactor."""
    view = FeatureSet(O, H, L, C, V).view(index)
    for got in (view.ema(20), view.atr(14), view.adx(14), view.rsi(14)):
        assert len(got) == index + 1
    for tup in (view.macd(), view.bollinger(20), view.donchian(20),
                view.supertrend(), view.keltner()):
        for arr in tup:
            assert len(arr) == index + 1


def test_a_view_matches_what_the_old_prefix_call_produced():
    """The exact substitution the optimisation makes, checked directly."""
    for index in (80, 150, 399):
        view = FeatureSet(O, H, L, C, V).view(index)
        old = ind.ema(C[:index + 1], 20)
        assert np.allclose(np.nan_to_num(view.ema(20), nan=-999.0),
                           np.nan_to_num(old, nan=-999.0), atol=1e-9)


# ── 4. the strategies emit the same signals ──────────────────────────────


def _signals_for(strategy, use_cache: bool):
    """Run a strategy bar by bar, with and without the cache, and collect what
    it emitted. Position is held flat so the comparison isolates the
    indicators rather than the trade state machine."""
    fs = FeatureSet(O, H, L, C, V) if use_cache else None
    out = []
    flat = Position(strategy=strategy.name, symbol="BTC/USD")
    for i in range(len(C)):
        ctx = Context(
            symbol="BTC/USD", timeframe="4h", index=i, ts=None,
            open=O[:i + 1], high=H[:i + 1], low=L[:i + 1],
            close=C[:i + 1], volume=V[:i + 1],
            position=flat, regime=Regime.UNKNOWN, params=strategy.params,
            f=fs.view(i) if fs is not None else None)
        for sig in strategy.on_bar(ctx):
            out.append((i, sig.direction, round(sig.strength, 12),
                        round(sig.ref_price or 0.0, 10),
                        round(sig.stop_price or 0.0, 10),
                        round(sig.target_price or 0.0, 10)))
    return out


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_the_cache_does_not_change_what_a_strategy_trades(name):
    """The test that decides whether this optimisation ships.

    A faster backtest that produces different signals is not an optimisation,
    it is a new strategy with the old one's name.
    """
    cached = _signals_for(REGISTRY[name](), use_cache=True)
    direct = _signals_for(REGISTRY[name](), use_cache=False)
    assert cached == direct, (
        f"{name}: {len(cached)} cached vs {len(direct)} direct signals differ")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_strategy_actually_uses_the_cache(name):
    """Guards against a strategy quietly reverting to a module-level call and
    losing the speedup without failing anything."""
    fs = FeatureSet(O, H, L, C, V)
    strategy = REGISTRY[name]()
    flat = Position(strategy=name, symbol="BTC/USD")
    for i in range(strategy.warmup + 3):
        ctx = Context(symbol="BTC/USD", timeframe="4h", index=i, ts=None,
                      open=O[:i + 1], high=H[:i + 1], low=L[:i + 1],
                      close=C[:i + 1], volume=V[:i + 1], position=flat,
                      regime=Regime.UNKNOWN, params=strategy.params,
                      f=fs.view(i))
        strategy.on_bar(ctx)
    assert fs.misses > 0, f"{name} never asked the cache for anything"


# ── 5. and only then, speed ──────────────────────────────────────────────


def test_the_cache_is_actually_faster():
    """The point of the exercise, measured rather than assumed.

    Deliberately a loose bound: this asserts an order of magnitude, not a
    number, because a tight threshold on a shared machine is a flaky test.
    """
    o, h, low_, c, v = _series(1500, seed=11)
    strategy = REGISTRY["ema_cross"]()
    flat = Position(strategy="ema_cross", symbol="BTC/USD")

    def run(use_cache: bool) -> float:
        fs = FeatureSet(o, h, low_, c, v) if use_cache else None
        started = time.perf_counter()
        for i in range(strategy.warmup, len(c)):
            ctx = Context(symbol="BTC/USD", timeframe="4h", index=i, ts=None,
                          open=o[:i + 1], high=h[:i + 1], low=low_[:i + 1],
                          close=c[:i + 1], volume=v[:i + 1], position=flat,
                          regime=Regime.UNKNOWN, params=strategy.params,
                          f=fs.view(i) if fs is not None else None)
            strategy.on_bar(ctx)
        return time.perf_counter() - started

    slow = run(False)
    fast = run(True)
    speedup = slow / max(fast, 1e-9)
    print(f"\n    1,500 bars, ema_cross: {slow * 1000:.0f} ms uncached vs "
          f"{fast * 1000:.0f} ms cached — {speedup:.1f}x")
    assert speedup > 3.0, f"only {speedup:.1f}x — the cache is not being used"
