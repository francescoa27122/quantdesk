"""Per-run indicator cache — the fix for B2 (O(n²) recomputation).

## The problem

`Strategy.on_bar` is handed a prefix of the series and calls
`ema(ctx.close, 20)`, which walks that prefix. Over `n` bars that is O(n²), and
every indicator in the strategy pays it again. Measured before this existed: a
6,000-bar × 18-parameter × 5-fold walk-forward **did not finish in ten minutes**
on the N100 and was killed. Champion/challenger, parameter search and nightly
re-fits all multiply that number.

## Why this is allowed to be fast

**Every indicator here is causal**, which means

    f(full)[i] == f(full[:i + 1])[i]      for every i

so an indicator computed once over the whole series is *identical* to the same
indicator recomputed from scratch on each prefix. That is not an assumption
this module makes quietly — `tests/test_features.py` asserts it for every
indicator in `ct.strategies.indicators`, and the strategy-level equivalence
test asserts that all seven strategies emit byte-identical signals with the
cache and without it. The backtest engine already relied on this property for
its ATR hoist; this generalises it.

Cost goes from O(n²) per indicator to **O(n) once**, plus an O(1) slice per bar.

## Why a strategy still cannot see the future

`FeatureView` hands out `array[:index + 1]` — a numpy view truncated at the
current bar. The strategy receives exactly what it received before: an array
whose last element is the current bar. Reaching forward is not merely
discouraged, there is nothing there to reach. `test_a_feature_view_cannot_see
_past_the_current_bar` pins it.

## Fallback

`Context.f` is `None` for a hand-built `Context` (most unit tests). Strategies
reach features through `ctx.features`, which builds a throwaway `FeatureSet`
over the arrays it was given. That is the old O(n²) behaviour and the old
results — correct, just not fast. Nothing breaks if an engine forgets to
supply a cache; it only gets slower, which is the right direction for a
mistake to fail in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ct.strategies import indicators as ind

Arr = NDArray[np.float64]


class FeatureSet:
    """Indicators over one symbol's full series, computed once and memoised.

    One instance per (strategy-run, symbol). Keyed by the indicator name and
    its parameters, so two strategies asking for `ema(20)` on the same series
    share one computation — which is the other half of the saving in a
    portfolio run where several strategies use the same ADX or ATR period.
    """

    __slots__ = (
        "_cache",
        "_close",
        "_high",
        "_low",
        "_open",
        "_volume",
        "hits",
        "misses",
    )

    def __init__(self, open: Arr, high: Arr, low: Arr, close: Arr,
                 volume: Arr | None = None) -> None:
        self._open = np.asarray(open, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)
        self._low = np.asarray(low, dtype=np.float64)
        self._close = np.asarray(close, dtype=np.float64)
        self._volume = (np.asarray(volume, dtype=np.float64)
                        if volume is not None else np.zeros_like(self._close))
        self._cache: dict[tuple[Any, ...], Any] = {}
        # Cheap evidence that the cache is doing its job. A run whose misses
        # equal its hits is a cache that is being rebuilt per bar.
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._close)

    # ── memoisation ──────────────────────────────────────────────────────

    def _memo[T](self, key: tuple[Any, ...], compute: Callable[[], T]) -> T:
        """Generic in the computed type, which is what makes every accessor
        below type-check. While this returned a bare `Any` the cache silently
        erased the type of all nineteen of them — one signature accounted for
        half this module's type findings.

        The cast is the real assertion: `_cache` is heterogeneous by design
        (arrays, tuples of arrays, boolean masks), so the key is what ties a
        slot to its type. Never reuse a key for a different computation.
        """
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cast("T", cached)
        self.misses += 1
        value = compute()
        self._cache[key] = value
        return value

    # ── indicators, full-series ──────────────────────────────────────────

    def sma(self, period: int) -> Arr:
        return self._memo(("sma", period), lambda: ind.sma(self._close, period))

    def ema(self, period: int) -> Arr:
        return self._memo(("ema", period), lambda: ind.ema(self._close, period))

    def rma(self, period: int) -> Arr:
        return self._memo(("rma", period), lambda: ind.rma(self._close, period))

    def sma_volume(self, period: int) -> Arr:
        """Volume, not price. Separate accessor rather than a `source`
        argument: two series behind one cache key is how a cache starts
        returning the wrong array."""
        return self._memo(("sma_volume", period),
                          lambda: ind.sma(self._volume, period))

    def rsi(self, period: int = 14) -> Arr:
        return self._memo(("rsi", period), lambda: ind.rsi(self._close, period))

    def roc(self, period: int = 12) -> Arr:
        return self._memo(("roc", period), lambda: ind.roc(self._close, period))

    def rolling_std(self, period: int) -> Arr:
        return self._memo(("rolling_std", period),
                          lambda: ind.rolling_std(self._close, period))

    def zscore(self, period: int = 20) -> Arr:
        return self._memo(("zscore", period),
                          lambda: ind.zscore(self._close, period))

    def realized_vol(self, period: int = 30) -> Arr:
        return self._memo(("realized_vol", period),
                          lambda: ind.realized_vol(self._close, period))

    def true_range(self) -> Arr:
        return self._memo(("true_range",),
                          lambda: ind.true_range(self._high, self._low,
                                                 self._close))

    def atr(self, period: int = 14) -> Arr:
        return self._memo(("atr", period),
                          lambda: ind.atr(self._high, self._low, self._close,
                                          period))

    def adx(self, period: int = 14) -> Arr:
        return self._memo(("adx", period),
                          lambda: ind.adx(self._high, self._low, self._close,
                                          period))

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9
             ) -> tuple[Arr, Arr, Arr]:
        return self._memo(("macd", fast, slow, signal),
                          lambda: ind.macd(self._close, fast, slow, signal))

    def bollinger(self, period: int = 20, num_std: float = 2.0
                  ) -> tuple[Arr, Arr, Arr]:
        return self._memo(("bollinger", period, num_std),
                          lambda: ind.bollinger(self._close, period, num_std))

    def keltner(self, period: int = 20, atr_period: int = 10,
                mult: float = 1.5) -> tuple[Arr, Arr, Arr]:
        return self._memo(("keltner", period, atr_period, mult),
                          lambda: ind.keltner(self._high, self._low,
                                              self._close, period, atr_period,
                                              mult))

    def donchian(self, period: int = 20) -> tuple[Arr, Arr, Arr]:
        return self._memo(("donchian", period),
                          lambda: ind.donchian(self._high, self._low, period))

    def supertrend(self, period: int = 10, multiplier: float = 3.0
                   ) -> tuple[Arr, Arr]:
        return self._memo(("supertrend", period, multiplier),
                          lambda: ind.supertrend(self._high, self._low,
                                                 self._close, period,
                                                 multiplier))

    def squeeze_on(self, bb_period: int = 20, bb_std: float = 2.0,
                   kc_period: int = 20, kc_mult: float = 1.5,
                   ) -> NDArray[np.bool_]:
        """No `atr_period`: the real function derives it as `kc_period // 2`.
        Adding one here would have been a parameter the cache key tracked and
        the computation ignored."""
        return self._memo(
            ("squeeze_on", bb_period, bb_std, kc_period, kc_mult),
            lambda: ind.squeeze_on(self._high, self._low, self._close,
                                   bb_period, bb_std, kc_period, kc_mult))

    # ── per-bar view ─────────────────────────────────────────────────────

    def view(self, index: int) -> FeatureView:
        return FeatureView(self, index)


def _cut[T](value: T, index: int) -> T:
    """Truncate an array, or each array of a tuple, at `index` inclusive.

    A view, not a copy: O(1) and no allocation, which is the whole point.

    Generic because truncation preserves the shape it was given: an array in,
    an array out; a 3-tuple of arrays in, a 3-tuple out. Returning `Any`
    erased that for every `FeatureView` accessor.
    """
    stop = index + 1
    sliceable: Any = value
    if isinstance(value, tuple):
        return cast("T", tuple(v[:stop] for v in value))
    return cast("T", sliceable[:stop])


class FeatureView:
    """One bar's window onto a `FeatureSet`.

    Every accessor returns the indicator truncated at the current bar, so a
    strategy holding a view has exactly the reach it had when it computed the
    indicator itself — and no more.
    """

    __slots__ = ("_fs", "_i")

    def __init__(self, features: FeatureSet, index: int) -> None:
        self._fs = features
        self._i = index

    @property
    def index(self) -> int:
        return self._i

    def sma(self, period: int) -> Arr:
        return _cut(self._fs.sma(period), self._i)

    def ema(self, period: int) -> Arr:
        return _cut(self._fs.ema(period), self._i)

    def rma(self, period: int) -> Arr:
        return _cut(self._fs.rma(period), self._i)

    def sma_volume(self, period: int) -> Arr:
        return _cut(self._fs.sma_volume(period), self._i)

    def rsi(self, period: int = 14) -> Arr:
        return _cut(self._fs.rsi(period), self._i)

    def roc(self, period: int = 12) -> Arr:
        return _cut(self._fs.roc(period), self._i)

    def rolling_std(self, period: int) -> Arr:
        return _cut(self._fs.rolling_std(period), self._i)

    def zscore(self, period: int = 20) -> Arr:
        return _cut(self._fs.zscore(period), self._i)

    def realized_vol(self, period: int = 30) -> Arr:
        return _cut(self._fs.realized_vol(period), self._i)

    def true_range(self) -> Arr:
        return _cut(self._fs.true_range(), self._i)

    def atr(self, period: int = 14) -> Arr:
        return _cut(self._fs.atr(period), self._i)

    def adx(self, period: int = 14) -> Arr:
        return _cut(self._fs.adx(period), self._i)

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9
             ) -> tuple[Arr, Arr, Arr]:
        return _cut(self._fs.macd(fast, slow, signal), self._i)

    def bollinger(self, period: int = 20, num_std: float = 2.0
                  ) -> tuple[Arr, Arr, Arr]:
        return _cut(self._fs.bollinger(period, num_std), self._i)

    def keltner(self, period: int = 20, atr_period: int = 10,
                mult: float = 1.5) -> tuple[Arr, Arr, Arr]:
        return _cut(self._fs.keltner(period, atr_period, mult), self._i)

    def donchian(self, period: int = 20) -> tuple[Arr, Arr, Arr]:
        return _cut(self._fs.donchian(period), self._i)

    def supertrend(self, period: int = 10, multiplier: float = 3.0
                   ) -> tuple[Arr, Arr]:
        return _cut(self._fs.supertrend(period, multiplier), self._i)

    def squeeze_on(self, bb_period: int = 20, bb_std: float = 2.0,
                   kc_period: int = 20, kc_mult: float = 1.5,
                   ) -> NDArray[np.bool_]:
        return _cut(self._fs.squeeze_on(bb_period, bb_std, kc_period, kc_mult),
                    self._i)


__all__ = ["FeatureSet", "FeatureView"]
