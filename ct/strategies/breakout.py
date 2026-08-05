"""Breakout strategies. 🟡 IMPLEMENTED BUT UNPROVEN.

Breakouts are the family most vulnerable to *backtest* fantasy rather than
live fantasy: a naive implementation compares the current bar's high to a
channel that includes the current bar's high, which is true on every new high.
`indicators.donchian` excludes the current bar for exactly this reason, and the
no-lookahead property test covers it.

The second trap is the false break. Both strategies below require a volume
confirmation, because a break on no volume is overwhelmingly noise and each
one costs a full round-trip in fees.
"""

from __future__ import annotations

import numpy as np

from ct.models import Signal
from ct.strategies.base import Context, Strategy


class DonchianTrendStrategy(Strategy):
    """Turtle-style N-bar channel break with an ATR trailing stop.

    The exit channel is deliberately shorter than the entry channel. A
    symmetric exit gives back most of the trend; the asymmetry is what makes
    the classic system work.
    """

    name = "donchian_breakout"
    family = "breakout"
    DEFAULTS = {
        "entry_period": 20,
        "exit_period": 10,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 3.0,
        "target_atr_mult": 8.0,
        "vol_period": 20,
        "vol_mult": 1.2,          # break must come with above-average volume
        "regime_ema": 200,
        "long_only": True,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["entry_period"], p["exit_period"], p["atr_period"],
                   p["vol_period"], p["regime_ema"] if p["long_only"] else 0) + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        up_in, _, lo_in = f.donchian(p["entry_period"])
        up_out, _, lo_out = f.donchian(p["exit_period"])
        a = f.atr(p["atr_period"])
        vol_ma = f.sma_volume(p["vol_period"])

        px, cur_atr = ctx.price, float(a[i])
        if not all(np.isfinite(x) for x in (up_in[i], lo_in[i], up_out[i],
                                            lo_out[i], cur_atr, vol_ma[i])):
            return []
        if cur_atr <= 0:
            return []

        pos = ctx.position
        if not pos.is_flat:
            # Exit on the shorter channel breaking the other way.
            if pos.qty > 0 and px < float(lo_out[i]):
                return [self._signal(ctx, 0, atr=cur_atr, reason="exit_channel")]
            if pos.qty < 0 and px > float(up_out[i]):
                return [self._signal(ctx, 0, atr=cur_atr, reason="exit_channel")]
            return []

        vol_ok = float(ctx.volume[i]) > p["vol_mult"] * float(vol_ma[i])
        if not vol_ok:
            return []

        if p["long_only"]:
            reg = f.ema(p["regime_ema"])
            if not np.isfinite(reg[i]) or px < float(reg[i]):
                return []

        if px > float(up_in[i]):
            return [self._signal(
                ctx, 1, atr=cur_atr,
                stop=px - p["stop_atr_mult"] * cur_atr,
                target=px + p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"],
                channel=float(up_in[i]),
            )]
        if px < float(lo_in[i]) and not p["long_only"]:
            return [self._signal(
                ctx, -1, atr=cur_atr,
                stop=px + p["stop_atr_mult"] * cur_atr,
                target=px - p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"],
                channel=float(lo_in[i]),
            )]
        return []


class AtrSqueezeBreakoutStrategy(Strategy):
    """Volatility-squeeze release (Bollinger inside Keltner), direction by close.

    The squeeze identifies compression; it says nothing about direction. The
    strategy therefore waits for the squeeze to RELEASE and takes the direction
    of the release bar. Entering during the squeeze — which is what most
    implementations do — is a coin flip that pays fees.
    """

    name = "atr_squeeze"
    family = "breakout"
    DEFAULTS = {
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_mult": 1.5,
        "min_squeeze_bars": 6,    # compression must be sustained
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 2.5,
        "target_atr_mult": 5.0,
        "vol_period": 20,
        "vol_mult": 1.2,
        "long_only": True,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["bb_period"], p["kc_period"], p["atr_period"],
                   p["vol_period"]) + p["min_squeeze_bars"] + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        sq = f.squeeze_on(p["bb_period"], p["bb_std"],
                        p["kc_period"], p["kc_mult"])
        a = f.atr(p["atr_period"])
        vol_ma = f.sma_volume(p["vol_period"])
        mid = f.sma(p["bb_period"])

        cur_atr = float(a[i])
        if not all(np.isfinite(x) for x in (cur_atr, vol_ma[i], mid[i])) or cur_atr <= 0:
            return []

        px, pos = ctx.price, ctx.position

        # Exit is the trailing stop (enforced engine-side) plus a re-entry into
        # a fresh squeeze, which means the move is over.
        if not pos.is_flat:
            if bool(sq[i]) and not bool(sq[i - 1]):
                return [self._signal(ctx, 0, atr=cur_atr, reason="squeeze_reformed")]
            return []

        released = bool(sq[i - 1]) and not bool(sq[i])
        if not released:
            return []
        window = sq[max(0, i - p["min_squeeze_bars"]):i]
        if not (len(window) and bool(np.all(window))):
            return []
        if float(ctx.volume[i]) <= p["vol_mult"] * float(vol_ma[i]):
            return []

        direction = 1 if px > float(mid[i]) else -1
        if direction < 0 and p["long_only"]:
            return []

        return [self._signal(
            ctx, direction, atr=cur_atr,
            stop=px - direction * p["stop_atr_mult"] * cur_atr,
            target=px + direction * p["target_atr_mult"] * cur_atr,
            trail_atr_mult=p["trail_atr_mult"],
            squeeze_bars=int(p["min_squeeze_bars"]),
        )]


__all__ = ["AtrSqueezeBreakoutStrategy", "DonchianTrendStrategy"]
