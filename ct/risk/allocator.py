"""Capital allocation and signal arbitration — Phase 2 item 7.

Two jobs that both answer "which strategy gets the scarce thing?".

## 1. Arbitration — the bug this fixes

`PortfolioBacktestEngine` evaluates signals in *allocation order*, and caps bind
on committed exposure, so when several strategies signal in the same pass the
capacity goes to whichever happens to be listed first. Reorder the allocations
and you get a different backtest from the same data — the audit recorded this as
"strategies are still first-come-first-served within a pass rather than
arbitrated".

That is not a tuning question, it is an artefact: **the result depended on
registry order.** `rank()` orders competing signals by expected edge net of
fees, scaled by the strategy's weight, so scarce capacity goes to the *best*
available signal and the outcome no longer depends on how the list was typed.
Ties break on `(strategy, symbol)` so runs stay reproducible.

## 2. Weights — and why this is not HRP yet

ARCHITECTURE §8 specifies Hierarchical Risk Parity over strategy-level return
streams. **HRP needs a correlation matrix between per-strategy equity curves,
and those do not exist**: strategies share one cash balance, and §7A recorded
that a per-strategy equity curve would need an allocation rule that had not been
invented. Building HRP on the data available today would mean inventing the
inputs, which is the failure mode this project spends the most effort avoiding.

So this ships the honest intermediate: **inverse-volatility weights** (risk
parity's diagonal — no correlation term), computed from whatever per-strategy
trade returns exist, with an explicit fall back to equal weight when there is
not enough history to estimate a variance. `Allocation.method` says which was
used, so a report can never imply a risk model that did not run.

HRP lands when per-strategy return streams do. That is a data dependency, not a
scheduling one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

#: Below this many observations, a variance estimate is noise pretending to be
#: information. Sized for trade returns, not bars: 20 round trips is already a
#: month or more of a 4h strategy at this project's deliberate low frequency.
MIN_OBSERVATIONS = 20


@dataclass(frozen=True, slots=True)
class Weights:
    """What each strategy gets, and how it was decided."""

    weights: dict[str, float]
    method: str                      # "inverse_vol" | "equal" | "mixed"
    reasons: dict[str, str] = field(default_factory=dict)

    def of(self, strategy: str) -> float:
        return self.weights.get(strategy, 0.0)

    def summary(self) -> str:
        rows = ", ".join(f"{k} {v:.1%}" for k, v in sorted(self.weights.items()))
        return f"{self.method}: {rows}"


def _cap_and_redistribute(weights: dict[str, float], cap: float
                          ) -> dict[str, float]:
    """Enforce a maximum weight, then hand the excess to everyone else.

    **Capping and then renormalising does not cap anything** — it scales the
    capped weight straight back up. Measured: a 0.5 cap on a two-strategy split
    of 0.996/0.004 produced 0.992/0.008. The excess has to be redistributed to
    the *other* weights and the check repeated, which is why this iterates.
    """
    w = dict(weights)
    for _ in range(len(w) + 1):
        over = [n for n, v in w.items() if v > cap + 1e-12]
        if not over:
            break
        excess = sum(w[n] - cap for n in over)
        for n in over:
            w[n] = cap
        free = [n for n in w if n not in over]
        if not free:
            break                    # everything is at the cap; normalise below
        room = sum(w[n] for n in free)
        for n in free:
            share = (w[n] / room) if room > 0 else (1.0 / len(free))
            w[n] += excess * share
    total = sum(w.values()) or 1.0
    return {n: v / total for n, v in w.items()}


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, var))


class Allocator:
    """Weights across strategies, and ordering within a pass."""

    def __init__(self, *, min_observations: int = MIN_OBSERVATIONS,
                 hard_max: float = 0.6, max_multiple_of_equal: float = 3.0
                 ) -> None:
        self.min_observations = min_observations
        # The cap exists to bound a *reciprocal*, not to impose diversification.
        # An inverse-vol weight is 1/σ, so a strategy that happened to be flat
        # across its sample — the estimator's least reliable case — would
        # otherwise take nearly the whole budget. Measured: 0.9960 of two.
        #
        # It is expressed two ways and the tighter wins, because either alone
        # misbehaves. A flat 0.5 forces *exact equality* on a two-strategy
        # book, erasing the weighting entirely; a pure multiple-of-equal never
        # binds when there are only two. `min(hard_max, multiple × 1/n)` keeps
        # the ordering meaningful at every book size while still bounding the
        # pathological case.
        self.hard_max = hard_max
        self.max_multiple_of_equal = max_multiple_of_equal

    def _cap_for(self, n: int) -> float:
        if n <= 1:
            return 1.0
        return min(self.hard_max, self.max_multiple_of_equal / n)

    # ── weights ──────────────────────────────────────────────────────────

    def weights(self, returns: dict[str, Sequence[float]]) -> Weights:
        """Inverse-volatility weights, or equal weight where unmeasurable.

        Returns are per-trade fractional returns, not bar returns: that is what
        `ClosedTrade.return_pct` gives, and a strategy that trades ten times a
        year has ten observations regardless of how many bars went by.
        """
        if not returns:
            return Weights({}, "equal", {})

        vols: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for name, series in returns.items():
            values = [float(r) for r in series]
            if len(values) < self.min_observations:
                reasons[name] = (f"{len(values)} observations < "
                                 f"{self.min_observations}: equal weight")
                continue
            vol = _stdev(values)
            if vol <= 0:
                reasons[name] = "zero variance in sample: equal weight"
                continue
            vols[name] = vol

        names = sorted(returns)
        if not vols:
            equal = 1.0 / len(names)
            return Weights({n: equal for n in names}, "equal", reasons)

        # Inverse vol over the measurable ones; the unmeasurable ones get the
        # average measurable weight rather than zero — absence of an estimate
        # is not evidence a strategy deserves nothing.
        inv = {n: 1.0 / v for n, v in vols.items()}
        total_inv = sum(inv.values())
        provisional = {n: inv[n] / total_inv * (len(vols) / len(names))
                       for n in vols}
        leftover = 1.0 - sum(provisional.values())
        unmeasured = [n for n in names if n not in vols]
        if unmeasured:
            share = leftover / len(unmeasured)
            for n in unmeasured:
                provisional[n] = share

        final = _cap_and_redistribute(provisional, self._cap_for(len(names)))
        method = ("inverse_vol" if len(vols) == len(names) else "mixed")
        for n in vols:
            reasons.setdefault(n, f"stdev {vols[n]:.4f} over {len(returns[n])}")
        return Weights(final, method, reasons)

    # ── arbitration ──────────────────────────────────────────────────────

    def rank(self, candidates: Sequence[Any], *,
             weights: Weights | None = None,
             key: Callable[[Any], Any] | None = None) -> list[Any]:
        """Order competing signals best-first for one engine pass.

        The score is the signal's own expected edge in bps — the number the fee
        gate already consumes (§0.1) — scaled by the strategy's weight. A
        signal that declines to estimate its edge scores 0 and sorts last
        rather than being dropped: it may still be filled if capacity remains,
        which keeps un-gated strategies visible instead of silently starved.

        Sorting is stable and fully deterministic: equal scores fall back to
        `(strategy, symbol)`, never to input order.
        """
        def score(item: Any) -> tuple[float, str, str]:
            signal = key(item) if key is not None else item
            meta = getattr(signal, "meta", None) or {}
            edge = meta.get("expected_edge_bps")
            try:
                edge_bps = float(edge) if edge is not None else 0.0
            except (TypeError, ValueError):
                edge_bps = 0.0
            # `.get(name, 1.0)`, not `.of(name)`: a strategy absent from the
            # weights map has no opinion attached to it and must rank on its
            # own edge, while an explicit 0.0 correctly sorts last. Conflating
            # the two would silently starve any strategy the weighting run had
            # not yet seen.
            weight = (weights.weights.get(signal.strategy, 1.0)
                      if weights else 1.0)
            strength = float(getattr(signal, "strength", 1.0) or 0.0)
            # Negative so a plain ascending sort puts the best first, and the
            # tie-breakers stay ascending and readable.
            return (-(edge_bps * weight * strength),
                    str(getattr(signal, "strategy", "")),
                    str(getattr(signal, "symbol", "")))

        return sorted(candidates, key=score)

    def rank_allocations(self, allocations: Sequence[Any], *,
                         weights: Weights | None = None) -> list[Any]:
        """Deterministic order for allocations themselves.

        Used where signals are generated rather than collected, so the *bar
        loop* stops depending on how the allocation list was typed.
        """
        def key(alloc: Any) -> tuple[float, str]:
            name = getattr(getattr(alloc, "strategy", None), "name", "")
            weight = weights.of(name) if weights else -getattr(alloc, "weight", 0.0)
            return (-weight, name)

        return sorted(allocations, key=key)


def returns_by_strategy(trades: Sequence[Any]) -> dict[str, list[float]]:
    """Per-strategy trade returns from `ClosedTrade`s.

    Trade returns rather than an equity curve, deliberately: strategies share
    one cash balance, so there is no honest per-strategy equity series to
    compute a variance from (§7A). A trade return is attributable without
    inventing an allocation rule to produce it.
    """
    out: dict[str, list[float]] = {}
    for trade in trades:
        name = getattr(trade, "strategy", None)
        if not name:
            continue
        value = getattr(trade, "return_pct", None)
        if value is None:
            continue
        out.setdefault(name, []).append(float(value))
    return out


def notional_cap(weight: float, equity: Decimal, cap_pct: float) -> Decimal:
    """A strategy's slice of the per-symbol notional cap.

    Kept here rather than in the risk engine so the engine's gates stay a
    function of the *account*, not of who is asking — the allocator scales the
    request before risk ever sees it, and risk still has the last word.
    """
    return Decimal(str(max(0.0, weight * cap_pct))) * equity


__all__ = [
    "MIN_OBSERVATIONS",
    "Allocator",
    "Weights",
    "notional_cap",
    "returns_by_strategy",
]
