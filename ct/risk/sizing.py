"""Position sizing. Pure functions — no I/O, no state, no exchange.

The rule (ARCHITECTURE.md §6.1): compute three independent candidate sizes and
take the MINIMUM. Each is a different question:

    risk-budget  — "how much can I lose if the stop hits?"
    vol-target   — "how much risk does this position add to the portfolio?"
    notional cap — "how concentrated am I willing to be?"

Taking the min means adding a sizer can only ever make positions smaller. That
asymmetry is the whole safety argument: a bug in a new sizer cannot inflate a
trade, only shrink it.
"""

from __future__ import annotations

import math
from decimal import Decimal

# Crypto trades 24/7/365. Using 252 here (the equity-market convention)
# understates annualised volatility by ~19% and therefore OVERSIZES every
# position by the same factor. This constant is load-bearing.
ANNUALIZATION_DAYS = 365.0

_TF_PER_YEAR = {
    "1m": ANNUALIZATION_DAYS * 24 * 60,
    "5m": ANNUALIZATION_DAYS * 24 * 12,
    "15m": ANNUALIZATION_DAYS * 24 * 4,
    "1h": ANNUALIZATION_DAYS * 24,
    "4h": ANNUALIZATION_DAYS * 6,
    "1d": ANNUALIZATION_DAYS,
    "1w": ANNUALIZATION_DAYS / 7,
}


def periods_per_year(timeframe: str) -> float:
    try:
        return _TF_PER_YEAR[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; add it to _TF_PER_YEAR rather "
            f"than guessing — a wrong value silently mis-sizes every position"
        ) from exc


def annualize_vol(period_vol: float, timeframe: str) -> float:
    """Scale a per-period stdev of log returns to an annual figure."""
    return period_vol * math.sqrt(periods_per_year(timeframe))


def ewma_volatility(returns: list[float] | tuple[float, ...], halflife: int = 30) -> float:
    """EWMA stdev of returns, per period.

    EWMA rather than a rolling window: crypto vol regime-shifts fast, and a
    simple 30-bar window gives every observation equal weight right up to the
    moment it falls off a cliff, producing a step change in position size.
    """
    if len(returns) < 2:
        return 0.0
    lam = 0.5 ** (1.0 / max(1, halflife))
    weights, var, wsum = [], 0.0, 0.0
    n = len(returns)
    for i in range(n):
        w = lam ** (n - 1 - i)
        weights.append(w)
        wsum += w
    mean = sum(w * r for w, r in zip(weights, returns)) / wsum
    for w, r in zip(weights, returns):
        var += w * (r - mean) ** 2
    return math.sqrt(var / wsum)


def risk_budget_size(
    equity: Decimal,
    risk_per_trade: float,
    stop_distance: float,
) -> Decimal:
    """Units such that a stop-out loses exactly `risk_per_trade` of equity.

    `stop_distance` is in price units (typically `atr_mult * ATR`). This is the
    primary sizer: it automatically shrinks positions when stops must be wide.
    """
    if stop_distance <= 0 or equity <= 0 or risk_per_trade <= 0:
        return Decimal(0)
    return (equity * Decimal(str(risk_per_trade))) / Decimal(str(stop_distance))


def vol_target_size(
    equity: Decimal,
    price: float,
    realized_ann_vol: float,
    target_ann_vol: float,
    weight: float = 1.0,
    max_leverage_multiple: float = 3.0,
) -> Decimal:
    """Units that make this position contribute ~`target_ann_vol` of risk.

    The `max_leverage_multiple` clamp matters: as realized vol → 0 the ratio
    explodes, and a quiet market would otherwise produce an enormous position
    immediately before the volatility that ended the quiet period.
    """
    if price <= 0 or equity <= 0 or realized_ann_vol <= 0:
        return Decimal(0)
    scalar = min(target_ann_vol / realized_ann_vol, max_leverage_multiple)
    notional = equity * Decimal(str(scalar * max(0.0, weight)))
    return notional / Decimal(str(price))


def notional_cap_size(equity: Decimal, price: float, max_position_pct: float) -> Decimal:
    if price <= 0 or equity <= 0:
        return Decimal(0)
    return (equity * Decimal(str(max_position_pct))) / Decimal(str(price))


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Full-Kelly optimal fraction: f* = W - (1-W)/R.

    Returned for reporting; NEVER applied at full size. See `fractional_kelly`.
    """
    if payoff_ratio <= 0:
        return 0.0
    return win_rate - (1.0 - win_rate) / payoff_ratio


def fractional_kelly(
    win_rate: float,
    payoff_ratio: float,
    fraction: float = 0.0,
    hard_cap: float = 0.25,
) -> float:
    """A multiplier in [0, 1] applied on top of the min-of-three size.

    Defaults to 0.0 → returns 1.0 (Kelly disabled). This is deliberate.

    Full Kelly is growth-optimal only if you know the true edge distribution.
    You have a small, non-stationary sample, and Kelly is extremely sensitive
    to over-estimated edge: a 2× error in `payoff_ratio` routinely turns
    half-Kelly into >60% drawdowns. So:

      * `fraction` is hard-capped at quarter-Kelly,
      * the result is clamped to [0, 1], so Kelly can only ever REDUCE size,
        never increase it,
      * a negative Kelly (no edge) sizes to zero — which is the correct answer.
    """
    if fraction <= 0.0:
        return 1.0
    f = kelly_fraction(win_rate, payoff_ratio)
    if f <= 0:
        return 0.0
    return max(0.0, min(1.0, f * min(fraction, hard_cap)))


def round_to_increment(qty: Decimal, increment: Decimal) -> Decimal:
    """Round DOWN to the exchange's lot size.

    Always down: rounding up can breach a cap that was just checked, and
    "slightly too small" is never the expensive error.
    """
    if increment <= 0:
        return qty
    return (qty // increment) * increment


__all__ = [
    "ANNUALIZATION_DAYS",
    "annualize_vol",
    "ewma_volatility",
    "fractional_kelly",
    "kelly_fraction",
    "notional_cap_size",
    "periods_per_year",
    "risk_budget_size",
    "round_to_increment",
    "vol_target_size",
]
