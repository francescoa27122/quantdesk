"""Strategy interface and the read-only Context handed to it.

The contract (ARCHITECTURE.md §4):

    Strategy.on_bar(ctx) -> list[Signal]

A strategy receives ONLY `ctx`, and `ctx` physically cannot contain the future:
the arrays are slices ending at the current closed bar. This is structural
look-ahead prevention — not a convention a future edit can quietly violate.

A strategy emits Signals, never Orders. It cannot size, cannot submit, cannot
see cash. If a strategy wants a bigger position it says `strength=1.0` and the
risk engine decides what that means in dollars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from ct.models import Bar, Position, Regime, Signal


@dataclass(frozen=True, slots=True)
class Context:
    """Read-only view of the world at the close of bar `index`."""
    symbol: str
    timeframe: str
    index: int
    ts: datetime
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    position: Position
    regime: Regime = Regime.UNKNOWN
    # S4-3: was `= None  # type: ignore[assignment]` — a non-optional field
    # with a None default, silenced instead of fixed. `default_factory` is
    # the correct spelling; every access site already treats it as a dict.
    params: dict[str, Any] = field(default_factory=dict)
    # Per-run indicator cache, supplied by whichever engine is driving. `None`
    # for a hand-built Context — see `features` below.
    f: Any = None

    @property
    def features(self):
        """Indicators for this bar, cached across bars when an engine supplies
        the cache and recomputed from scratch when it does not.

        The fallback is the old O(n²) behaviour and the old numbers: correct,
        merely slow. An engine that forgets to pass a cache therefore loses
        speed, never accuracy — the right direction for that mistake to fail
        in, and the reason every existing hand-built `Context` in the test
        suite keeps working untouched.
        """
        if self.f is not None:
            return self.f
        from ct.strategies.features import FeatureSet

        return FeatureSet(self.open, self.high, self.low, self.close,
                          self.volume).view(self.index)

    @property
    def bar(self) -> Bar:
        i = self.index
        return Bar(self.ts, float(self.open[i]), float(self.high[i]),
                   float(self.low[i]), float(self.close[i]),
                   float(self.volume[i]), self.symbol, self.timeframe)

    @property
    def price(self) -> float:
        return float(self.close[self.index])

    @property
    def n(self) -> int:
        return self.index + 1

    def __len__(self) -> int:
        return self.n


class Strategy(ABC):
    """Base class. Subclasses set `name`, `family`, `warmup`, and DEFAULTS."""

    name: ClassVar[str] = "unnamed"
    # 'trend' | 'momentum' | 'breakout' | 'reversion' — read by the risk
    # engine's regime gate (ct/risk/engine.py §6).
    family: ClassVar[str] = "trend"
    DEFAULTS: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            # Silently accepting an unknown param is how a walk-forward ends up
            # optimising a value the strategy never reads.
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params: dict[str, Any] = {**self.DEFAULTS, **params}

    # ── required ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars needed before any signal is valid. The engine will not call
        `on_bar` until this many bars exist."""

    @abstractmethod
    def on_bar(self, ctx: Context) -> list[Signal]:
        """Return zero or more Signals for the CLOSED bar at ctx.index."""

    # ── helpers available to subclasses ──────────────────────────────────

    def _signal(
        self,
        ctx: Context,
        direction: int,
        *,
        strength: float = 1.0,
        atr: float | None = None,
        stop: float | None = None,
        target: float | None = None,
        trail_atr_mult: float | None = None,
        expected_edge_bps: float | None = None,
        **meta: Any,
    ) -> Signal:
        """Build a Signal with `family` and edge estimate attached.

        `expected_edge_bps` is what the risk engine's fee gate consumes. A
        strategy that cannot estimate its own edge should pass None — the gate
        then abstains rather than guessing, and the walk-forward report flags
        the strategy as un-gated.
        """
        m: dict[str, Any] = {"family": self.family, **meta}
        if expected_edge_bps is not None:
            m["expected_edge_bps"] = expected_edge_bps
        elif target is not None and direction != 0:
            px = ctx.price
            if px > 0:
                m["expected_edge_bps"] = abs(target - px) / px * 10_000.0
        return Signal(
            ts=ctx.ts,
            strategy=self.name,
            symbol=ctx.symbol,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
            ref_price=ctx.price,
            stop_price=stop,
            target_price=target,
            trail_atr_mult=trail_atr_mult,
            atr=atr,
            regime=ctx.regime,
            meta=m,
        )

    @staticmethod
    def _crossed_up(series: NDArray[np.float64], other: NDArray[np.float64],
                    i: int) -> bool:
        if i < 1:
            return False
        a0, a1, b0, b1 = series[i - 1], series[i], other[i - 1], other[i]
        if any(np.isnan(x) for x in (a0, a1, b0, b1)):
            return False
        return a0 <= b0 and a1 > b1

    @staticmethod
    def _crossed_down(series: NDArray[np.float64], other: NDArray[np.float64],
                      i: int) -> bool:
        if i < 1:
            return False
        a0, a1, b0, b1 = series[i - 1], series[i], other[i - 1], other[i]
        if any(np.isnan(x) for x in (a0, a1, b0, b1)):
            return False
        return a0 >= b0 and a1 < b1

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


__all__ = ["Context", "Strategy"]
