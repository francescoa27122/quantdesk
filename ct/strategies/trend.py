"""Trend and momentum strategies.

All three are 🟡 IMPLEMENTED BUT UNPROVEN. None has passed the promotion gate
(ARCHITECTURE.md §8.3). Do not allocate capital on the strength of a backtest
in this file.

Design note common to all: entries are filtered by ADX (trend strength) and
exits lean on an ATR trailing stop rather than a symmetric opposite signal.
Trend systems make their money on a small number of large moves; an exit that
fires on the first counter-signal cuts exactly those. The trailing stop is the
strategy.
"""

from __future__ import annotations

import numpy as np

from ct.models import Signal
from ct.strategies.base import Context, Strategy


class EmaCrossStrategy(Strategy):
    """Fast/slow EMA cross, gated by ADX, trailed by ATR.

    The ADX gate is what separates this from the textbook version that loses
    money: an EMA cross in a range fires constantly and every one of those
    trades pays 0.5–0.8% in fees (§0.1).
    """

    name = "ema_cross"
    family = "trend"
    DEFAULTS = {
        "fast": 20,
        "slow": 50,
        "adx_period": 14,
        "adx_min": 25.0,        # below this, treat the market as ranging
        "atr_period": 14,
        "stop_atr_mult": 2.5,
        "trail_atr_mult": 3.0,
        "target_atr_mult": 6.0,  # 1:2.4 reward:risk; also feeds the fee gate
        "long_only": True,       # spot default
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["slow"], 2 * p["adx_period"] + 1, p["atr_period"]) + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        fast = f.ema(p["fast"])
        slow = f.ema(p["slow"])
        a = f.atr(p["atr_period"])
        strength = f.adx(p["adx_period"])

        cur_atr = float(a[i])
        if not np.isfinite(cur_atr) or cur_atr <= 0:
            return []
        adx_now = float(strength[i])
        if not np.isfinite(adx_now):
            return []

        px = ctx.price
        pos = ctx.position

        # Exit first: an open position's exit must never be blocked by the
        # same filter that would block a new entry.
        if not pos.is_flat:
            if pos.qty > 0 and self._crossed_down(fast, slow, i):
                return [self._signal(ctx, 0, atr=cur_atr, reason="ema_cross_down")]
            if pos.qty < 0 and self._crossed_up(fast, slow, i):
                return [self._signal(ctx, 0, atr=cur_atr, reason="ema_cross_up")]
            return []

        if adx_now < p["adx_min"]:
            return []

        # Conviction scales with how far ADX is above the threshold, saturating
        # at 50 (the point beyond which ADX stops discriminating).
        conviction = min(1.0, (adx_now - p["adx_min"]) / max(1.0, 50.0 - p["adx_min"]))
        conviction = 0.5 + 0.5 * conviction

        if self._crossed_up(fast, slow, i):
            return [self._signal(
                ctx, 1, strength=conviction, atr=cur_atr,
                stop=px - p["stop_atr_mult"] * cur_atr,
                target=px + p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"], adx=adx_now,
            )]
        if self._crossed_down(fast, slow, i) and not p["long_only"]:
            return [self._signal(
                ctx, -1, strength=conviction, atr=cur_atr,
                stop=px + p["stop_atr_mult"] * cur_atr,
                target=px - p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"], adx=adx_now,
            )]
        return []


class SupertrendStrategy(Strategy):
    """Supertrend flip, long-only above a long-period EMA.

    The regime filter (price above the 200-EMA) is doing most of the work:
    Supertrend alone flips constantly in a downtrend and bleeds fees.
    """

    name = "supertrend"
    family = "trend"
    DEFAULTS = {
        "st_period": 10,
        "st_mult": 3.0,
        "regime_ema": 200,
        "atr_period": 14,
        "stop_atr_mult": 2.5,
        "trail_atr_mult": 3.0,
        "target_atr_mult": 6.0,
        "long_only": True,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["regime_ema"], p["st_period"], p["atr_period"]) + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        _, direction = f.supertrend(p["st_period"], p["st_mult"])
        regime = f.ema(p["regime_ema"])
        a = f.atr(p["atr_period"])

        d_now, d_prev = direction[i], direction[i - 1]
        cur_atr = float(a[i])
        if not all(np.isfinite(x) for x in (d_now, d_prev, cur_atr, regime[i])):
            return []
        if cur_atr <= 0:
            return []

        px, pos = ctx.price, ctx.position
        flipped_up = d_prev < 0 and d_now > 0
        flipped_down = d_prev > 0 and d_now < 0

        if not pos.is_flat:
            if (pos.qty > 0 and flipped_down) or (pos.qty < 0 and flipped_up):
                return [self._signal(ctx, 0, atr=cur_atr, reason="supertrend_flip")]
            return []

        above = px > float(regime[i])
        if flipped_up and above:
            return [self._signal(
                ctx, 1, atr=cur_atr,
                stop=px - p["stop_atr_mult"] * cur_atr,
                target=px + p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"],
            )]
        if flipped_down and not above and not p["long_only"]:
            return [self._signal(
                ctx, -1, atr=cur_atr,
                stop=px + p["stop_atr_mult"] * cur_atr,
                target=px - p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"],
            )]
        return []


class MacdMomentumStrategy(Strategy):
    """MACD histogram sign change, confirmed by positive rate-of-change.

    The ROC confirmation exists because MACD histogram flips are frequent and
    most are noise. Requiring price to actually have moved in the signal's
    direction over `roc_period` bars cuts trade count roughly in half, which at
    0.5–0.8% round-trip is worth more than the signals it discards.
    """

    name = "macd_momentum"
    family = "momentum"
    DEFAULTS = {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "roc_period": 12,
        "roc_min": 0.0,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 2.5,
        "target_atr_mult": 4.0,
        "long_only": True,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["slow"] + p["signal"], p["roc_period"], p["atr_period"]) + 5

    def on_bar(self, ctx: Context) -> list[Signal]:
        p = self.params
        i = ctx.index
        f = ctx.features
        if i < self.warmup:
            return []

        _, _, hist = f.macd(p["fast"], p["slow"], p["signal"])
        momentum = f.roc(p["roc_period"])
        a = f.atr(p["atr_period"])

        h_now, h_prev, r_now, cur_atr = hist[i], hist[i - 1], momentum[i], float(a[i])
        if not all(np.isfinite(x) for x in (h_now, h_prev, r_now, cur_atr)):
            return []
        if cur_atr <= 0:
            return []

        px, pos = ctx.price, ctx.position
        turned_up = h_prev <= 0 < h_now
        turned_down = h_prev >= 0 > h_now

        if not pos.is_flat:
            if (pos.qty > 0 and turned_down) or (pos.qty < 0 and turned_up):
                return [self._signal(ctx, 0, atr=cur_atr, reason="macd_flip")]
            return []

        if turned_up and r_now > p["roc_min"]:
            return [self._signal(
                ctx, 1, atr=cur_atr,
                stop=px - p["stop_atr_mult"] * cur_atr,
                target=px + p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"], roc=float(r_now),
            )]
        if turned_down and r_now < -p["roc_min"] and not p["long_only"]:
            return [self._signal(
                ctx, -1, atr=cur_atr,
                stop=px + p["stop_atr_mult"] * cur_atr,
                target=px - p["target_atr_mult"] * cur_atr,
                trail_atr_mult=p["trail_atr_mult"], roc=float(r_now),
            )]
        return []


__all__ = ["EmaCrossStrategy", "MacdMomentumStrategy", "SupertrendStrategy"]
