"""Mean-reversion strategies. 🟡 IMPLEMENTED BUT UNPROVEN.

Read this before touching anything here:

Mean reversion is the family most likely to look brilliant in a backtest and
lose money live, for two reasons that are both addressed structurally below.

1. **Fees.** Reversion trades are short-horizon and frequent. At 0.5–0.8%
   round-trip (ARCHITECTURE.md §0.1) a reversion strategy that targets a 1%
   move is barely break-even before it is wrong even once. Both strategies
   here therefore require a MINIMUM target distance measured in ATR, and pass
   `expected_edge_bps` to the risk engine's fee gate, which will refuse the
   trade outright if the target does not clear costs.

2. **Regime.** Buying dips works until the dip is the beginning of a trend, at
   which point the strategy buys every rung on the way down. Both strategies
   require ADX below a threshold (ranging market) AND the risk engine's regime
   gate independently blocks the whole family when the HMM says `trending`.
   Two independent filters, because this failure mode is what kills accounts.

There is deliberately no VWAP reversion and no Stochastic-RSI strategy here.
Both are intraday by construction and arithmetically dead at this fee tier.
"""

from __future__ import annotations

import numpy as np

from ct.models import Signal
from ct.strategies.base import Context, Strategy
from ct.strategies.indicators import sma, zscore


class BollingerReversionStrategy(Strategy):
    """Buy the lower band in a ranging market; exit at the mid-band.

    Exit at the MID band, not the upper: the upper band is where the trade
    would have been most profitable in hindsight, and waiting for it converts a
    high-win-rate strategy into a low-win-rate one with the same expectancy but
    a much worse drawdown profile.
    """

    name = "bollinger_reversion"
    family = "reversion"
    DEFAULTS = {
        "period": 20,
        "num_std": 2.0,
        "adx_period": 14,
        "adx_max": 20.0,          # only trade when NOT trending
        "rsi_period": 14,
        "rsi_max": 35.0,          # confirmation: actually oversold
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "min_target_atr": 1.5,    # refuse trades whose target is too close
        "regime_ema": 200,
        "require_above_regime": True,   # only buy dips in an uptrend
        "long_only": True,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        # int(): see breakout.warmup — `params` is dict[str, Any] and a warmup
        # is a bar count, so coerce at the boundary rather than cast.
        return int(max(p["regime_ema"] if p["require_above_regime"] else 0,
                       p["period"], 2 * p["adx_period"] + 1,
                       p["atr_period"])) + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        upper, mid, lower = f.bollinger(p["period"], p["num_std"])
        strength = f.adx(p["adx_period"])
        r = f.rsi(p["rsi_period"])
        a = f.atr(p["atr_period"])

        px = ctx.price
        cur_atr, mid_now, low_now, up_now = float(a[i]), mid[i], lower[i], upper[i]
        adx_now, rsi_now = strength[i], r[i]
        if not all(np.isfinite(x) for x in (cur_atr, mid_now, low_now, up_now,
                                            adx_now, rsi_now)) or cur_atr <= 0:
            return []

        pos = ctx.position

        # Exit: price has reverted to the mean.
        if not pos.is_flat:
            if pos.qty > 0 and px >= mid_now:
                return [self._signal(ctx, 0, atr=cur_atr, reason="reverted_to_mid")]
            if pos.qty < 0 and px <= mid_now:
                return [self._signal(ctx, 0, atr=cur_atr, reason="reverted_to_mid")]
            return []

        if adx_now > p["adx_max"]:
            return []

        if p["require_above_regime"]:
            reg = f.ema(p["regime_ema"])
            if not np.isfinite(reg[i]) or px < float(reg[i]):
                return []

        # Long: closed below the lower band and RSI confirms oversold.
        if px < low_now and rsi_now < p["rsi_max"]:
            target = float(mid_now)
            # The fee gate in ct/risk/engine.py will reject this anyway if the
            # edge is too thin, but rejecting here keeps the `signals` table
            # honest about WHY (strategy declined vs risk vetoed).
            if (target - px) < p["min_target_atr"] * cur_atr:
                return []
            # Conviction from depth below the band, saturating at 1 extra ATR.
            depth = (float(low_now) - px) / cur_atr
            return [self._signal(
                ctx, 1, strength=min(1.0, 0.5 + depth), atr=cur_atr,
                stop=px - p["stop_atr_mult"] * cur_atr,
                target=target, rsi=float(rsi_now), adx=float(adx_now),
            )]

        if not p["long_only"] and px > up_now and rsi_now > (100 - p["rsi_max"]):
            target = float(mid_now)
            if (px - target) < p["min_target_atr"] * cur_atr:
                return []
            depth = (px - float(up_now)) / cur_atr
            return [self._signal(
                ctx, -1, strength=min(1.0, 0.5 + depth), atr=cur_atr,
                stop=px + p["stop_atr_mult"] * cur_atr,
                target=target, rsi=float(rsi_now), adx=float(adx_now),
            )]
        return []


class ZScoreReversionStrategy(Strategy):
    """Z-score of log price vs. its own rolling mean.

    Uses LOG price, not price. Z-scoring raw price makes the same percentage
    deviation register as a larger z at higher price levels, so the strategy
    silently becomes more aggressive as an asset appreciates — which is the
    opposite of what you want.
    """

    name = "zscore_reversion"
    family = "reversion"
    DEFAULTS = {
        "period": 30,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_z": 4.0,             # beyond this it is not noise, it is news
        "adx_period": 14,
        "adx_max": 20.0,
        "atr_period": 14,
        "stop_atr_mult": 2.5,
        "min_target_atr": 1.5,
        "long_only": True,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return int(max(p["period"], 2 * p["adx_period"] + 1,
                       p["atr_period"])) + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        with np.errstate(divide="ignore", invalid="ignore"):
            log_close = np.log(np.maximum(ctx.close, 1e-12))
        z = zscore(log_close, p["period"])
        mean_log = sma(log_close, p["period"])
        strength = f.adx(p["adx_period"])
        a = f.atr(p["atr_period"])

        z_now, cur_atr, adx_now = z[i], float(a[i]), strength[i]
        if not all(np.isfinite(x) for x in (z_now, cur_atr, adx_now, mean_log[i])):
            return []
        if cur_atr <= 0:
            return []

        px, pos = ctx.price, ctx.position
        mean_price = float(np.exp(mean_log[i]))

        if not pos.is_flat:
            if abs(float(z_now)) <= p["exit_z"]:
                return [self._signal(ctx, 0, atr=cur_atr, reason="z_reverted",
                                     z=float(z_now))]
            return []

        if adx_now > p["adx_max"]:
            return []
        # A z beyond max_z is a regime break, not an overreaction. Standing in
        # front of it is the classic way this strategy family blows up.
        if abs(float(z_now)) > p["max_z"]:
            return []

        if z_now <= -p["entry_z"]:
            if (mean_price - px) < p["min_target_atr"] * cur_atr:
                return []
            conviction = min(1.0, (abs(float(z_now)) - p["entry_z"]) /
                             max(0.1, p["max_z"] - p["entry_z"]))
            return [self._signal(
                ctx, 1, strength=0.5 + 0.5 * conviction, atr=cur_atr,
                stop=px - p["stop_atr_mult"] * cur_atr,
                target=mean_price, z=float(z_now),
            )]

        if not p["long_only"] and z_now >= p["entry_z"]:
            if (px - mean_price) < p["min_target_atr"] * cur_atr:
                return []
            conviction = min(1.0, (abs(float(z_now)) - p["entry_z"]) /
                             max(0.1, p["max_z"] - p["entry_z"]))
            return [self._signal(
                ctx, -1, strength=0.5 + 0.5 * conviction, atr=cur_atr,
                stop=px + p["stop_atr_mult"] * cur_atr,
                target=mean_price, z=float(z_now),
            )]
        return []


__all__ = ["BollingerReversionStrategy", "ZScoreReversionStrategy"]
