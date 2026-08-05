"""Causal technical indicators.

Every function here obeys one rule: **the value at index i depends only on
inputs at indices <= i**. That is enforced by a property test
(`tests/test_backtest.py::test_indicators_are_causal`) which appends arbitrary
future data and asserts the historical prefix is byte-identical.

Warm-up periods are returned as NaN rather than silently backfilled. A
backfilled warm-up is a look-ahead bug wearing a disguise: it makes the first
N bars of every backtest look like the strategy already knew the mean.

Pure numpy — no TA-Lib. TA-Lib needs a C build on the N100 and pins the whole
image to a compiler toolchain for a dozen functions of ten lines each.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def _as_array(x) -> Arr:
    return np.asarray(x, dtype=np.float64)


# ── Moving averages ──────────────────────────────────────────────────────

def sma(values, period: int) -> Arr:
    v = _as_array(values)
    out = np.full(v.shape, np.nan)
    if period <= 0 or len(v) < period:
        return out
    csum = np.cumsum(np.insert(v, 0, 0.0))
    out[period - 1:] = (csum[period:] - csum[:-period]) / period
    return out


def ema(values, period: int) -> Arr:
    """Wilder-free EMA, alpha = 2/(n+1), seeded with the first SMA.

    Seeding with the SMA rather than the first observation matters: seeding
    with v[0] makes the series depend heavily on where your data happens to
    start, which shows up as an unreproducible backtest.
    """
    v = _as_array(values)
    out = np.full(v.shape, np.nan)
    if period <= 0 or len(v) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = v[:period].mean()
    for i in range(period, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def rma(values, period: int) -> Arr:
    """Wilder's smoothing (alpha = 1/n). Used by RSI, ATR, ADX."""
    v = _as_array(values)
    out = np.full(v.shape, np.nan)
    if period <= 0 or len(v) < period:
        return out
    out[period - 1] = v[:period].mean()
    for i in range(period, len(v)):
        out[i] = (out[i - 1] * (period - 1) + v[i]) / period
    return out


# ── Volatility ───────────────────────────────────────────────────────────

def true_range(high, low, close) -> Arr:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    prev = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - prev), np.abs(l[1:] - prev)])
    return tr


def atr(high, low, close, period: int = 14) -> Arr:
    return rma(true_range(high, low, close), period)


def rolling_std(values, period: int) -> Arr:
    v = _as_array(values)
    out = np.full(v.shape, np.nan)
    if len(v) < period or period < 2:
        return out
    for i in range(period - 1, len(v)):
        out[i] = v[i - period + 1:i + 1].std(ddof=1)
    return out


# ── Trend / momentum ─────────────────────────────────────────────────────

def macd(close, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[Arr, Arr, Arr]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(np.nan_to_num(line, nan=0.0), signal)
    sig[:slow + signal - 2] = np.nan
    return line, sig, line - sig


def rsi(close, period: int = 14) -> Arr:
    c = _as_array(close)
    out = np.full(c.shape, np.nan)
    if len(c) <= period:
        return out
    delta = np.diff(c, prepend=c[0])
    gain = rma(np.where(delta > 0, delta, 0.0), period)
    loss = rma(np.where(delta < 0, -delta, 0.0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(loss == 0, np.inf, gain / loss)
        out = 100.0 - 100.0 / (1.0 + rs)
    out[np.isnan(gain)] = np.nan
    return out


def adx(high, low, close, period: int = 14) -> Arr:
    """Average Directional Index — trend *strength*, direction-agnostic.

    This is the regime filter that keeps the reversion book out of trends and
    vice versa. Convention: >25 trending, <20 ranging.
    """
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    out = np.full(n, np.nan)
    if n < 2 * period + 1:
        return out
    up, dn = np.zeros(n), np.zeros(n)
    um, dm = h[1:] - h[:-1], l[:-1] - l[1:]
    up[1:] = np.where((um > dm) & (um > 0), um, 0.0)
    dn[1:] = np.where((dm > um) & (dm > 0), dm, 0.0)
    tr = rma(true_range(h, l, c), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * rma(up, period) / tr
        ndi = 100.0 * rma(dn, period) / tr
        dx = 100.0 * np.abs(pdi - ndi) / (pdi + ndi)
    return rma(np.nan_to_num(dx, nan=0.0), period)


def roc(close, period: int = 12) -> Arr:
    c = _as_array(close)
    out = np.full(c.shape, np.nan)
    if len(c) <= period:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period:] = (c[period:] / c[:-period] - 1.0) * 100.0
    return out


def supertrend(high, low, close, period: int = 10, multiplier: float = 3.0
               ) -> tuple[Arr, Arr]:
    """Return (supertrend_line, direction) where direction is +1 / -1.

    The band-locking logic (bands may only tighten while the trend holds) is
    the whole point of Supertrend; a naive re-computation each bar produces a
    whipsaw machine.
    """
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    upper, lower = hl2 + multiplier * a, hl2 - multiplier * a
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    direction = np.full(n, np.nan)
    st = np.full(n, np.nan)

    start = period
    if n <= start:
        return st, direction
    fu[start], fl[start], direction[start] = upper[start], lower[start], 1.0
    st[start] = fl[start]
    for i in range(start + 1, n):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            direction[i] = 1.0
        elif c[i] < fl[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
        st[i] = fl[i] if direction[i] > 0 else fu[i]
    return st, direction


# ── Bands / channels ─────────────────────────────────────────────────────

def bollinger(close, period: int = 20, num_std: float = 2.0
              ) -> tuple[Arr, Arr, Arr]:
    mid = sma(close, period)
    sd = rolling_std(close, period)
    return mid + num_std * sd, mid, mid - num_std * sd


def keltner(high, low, close, period: int = 20, atr_period: int = 10,
            multiplier: float = 1.5) -> tuple[Arr, Arr, Arr]:
    mid = ema(close, period)
    a = atr(high, low, close, atr_period)
    return mid + multiplier * a, mid, mid - multiplier * a


def donchian(high, low, period: int = 20) -> tuple[Arr, Arr, Arr]:
    """Upper/mid/lower channel EXCLUDING the current bar.

    Excluding the current bar is not a stylistic choice: including it means
    "price broke above the highest high including its own high", which is true
    on every new high and generates a signal you cannot actually trade.
    """
    h, l = _as_array(high), _as_array(low)
    n = len(h)
    up, lo = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(period, n):
        up[i] = h[i - period:i].max()
        lo[i] = l[i - period:i].min()
    return up, (up + lo) / 2.0, lo


def zscore(values, period: int = 20) -> Arr:
    v = _as_array(values)
    mean, sd = sma(v, period), rolling_std(v, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (v - mean) / sd, np.nan)


def squeeze_on(high, low, close, bb_period: int = 20, bb_std: float = 2.0,
               kc_period: int = 20, kc_mult: float = 1.5) -> NDArray[np.bool_]:
    """True where the Bollinger bands sit INSIDE the Keltner channel."""
    bu, _, bl = bollinger(close, bb_period, bb_std)
    ku, _, kl = keltner(high, low, close, kc_period, kc_period // 2 or 1, kc_mult)
    return (bu < ku) & (bl > kl)


def realized_vol(close, period: int = 30) -> Arr:
    """Per-bar stdev of log returns. Annualise with sizing.annualize_vol."""
    c = _as_array(close)
    out = np.full(c.shape, np.nan)
    if len(c) < period + 1:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(c), prepend=np.log(c[0]))
    lr[0] = np.nan
    for i in range(period, len(c)):
        out[i] = np.nanstd(lr[i - period + 1:i + 1], ddof=1)
    return out


__all__ = [
    "adx",
    "atr",
    "bollinger",
    "donchian",
    "ema",
    "keltner",
    "macd",
    "realized_vol",
    "rma",
    "roc",
    "rolling_std",
    "rsi",
    "sma",
    "squeeze_on",
    "supertrend",
    "true_range",
    "zscore",
]
