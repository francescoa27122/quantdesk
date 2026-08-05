"""Market-data integrity — the component that can stop trading.

Every risk gate downstream is computed from a price. A stale, gapped or
out-of-order feed therefore does not degrade the system's decisions gracefully;
it invalidates all of them at once, silently, while everything still looks
healthy. So the rule from the brief is taken literally:

    If data integrity is uncertain: pause trading, recover, resume safely.

This module is the "uncertain" detector. It is pure — no network, no clock of
its own, no database — because a health monitor that cannot be tested
deterministically is a health monitor you do not believe at 3 a.m.

Four independent faults, each detected separately because each recovers
differently:

  * **Staleness** — the newest bar is too old. Exchange down, websocket dead, or
    a symbol delisted. Recovers by itself when data resumes.
  * **Gaps** — bars missing in the middle. Reconnect dropped messages. Needs a
    REST backfill, which `ct/data/backfill.py` can do because `since` works.
  * **Out-of-order / duplicate** — the same or an earlier bar arriving again.
    Normal on reconnect; harmless if deduplicated, corrupting if not.
  * **Implausible values** — non-positive prices, high < low, a close outside
    [low, high], or a move so large it is more likely bad data than a market
    event. A single bad tick can trip every stop in the book.

A symbol is `DEGRADED` until its fault clears **and** a settling period passes.
The delay is deliberate: resuming the instant one good bar arrives means
flapping in and out of halt during exactly the conditions that caused the fault.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ct.models import Bar

log = logging.getLogger(__name__)

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14_400, "1d": 86_400, "1w": 604_800,
}


def timeframe_seconds(timeframe: str) -> int:
    try:
        return TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unknown timeframe {timeframe!r} — add it to TIMEFRAME_SECONDS "
            f"rather than guessing; a wrong value mis-detects every gap") from exc


class IntegrityState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"     # nothing seen yet — not the same as healthy


class FaultKind(str, Enum):
    STALE = "stale"
    GAP = "gap"
    OUT_OF_ORDER = "out_of_order"
    IMPLAUSIBLE = "implausible"


@dataclass(frozen=True, slots=True)
class Fault:
    kind: FaultKind
    symbol: str
    timeframe: str
    detail: str
    at: datetime

    def __str__(self) -> str:
        return f"{self.symbol} {self.timeframe} {self.kind.value}: {self.detail}"


@dataclass
class SymbolHealth:
    symbol: str
    timeframe: str
    state: IntegrityState = IntegrityState.UNKNOWN
    last_bar_ts: datetime | None = None
    last_update: datetime | None = None
    bars_seen: int = 0
    gaps_detected: int = 0
    duplicates_dropped: int = 0
    out_of_order_dropped: int = 0
    implausible_dropped: int = 0
    missing_bars: int = 0
    fault: Fault | None = None
    healthy_since: datetime | None = None

    @property
    def is_tradeable(self) -> bool:
        """UNKNOWN is not tradeable. 'We have never seen a price' and 'the price
        is fine' must never collapse into the same answer."""
        return self.state is IntegrityState.OK

    def age_bars(self, now: datetime) -> int:
        if self.last_bar_ts is None:
            return 10**6            # effectively infinite; nothing is fresh
        secs = (now - self.last_bar_ts).total_seconds()
        return max(0, int(secs // timeframe_seconds(self.timeframe)) - 1)


@dataclass
class IntegrityConfig:
    # A bar is late once this many intervals have passed without it. 2 allows
    # for one missed publish plus clock skew before crying wolf.
    max_stale_bars: int = 2
    # Consecutive bars that must arrive cleanly before DEGRADED clears. Without
    # a settling period the system flaps in and out of halt during precisely
    # the conditions that caused the fault.
    recovery_bars: int = 3
    # A single-bar move beyond this is more likely bad data than a market event.
    # Crypto does move 20% in a day — but not in one 4h candle without the whole
    # book moving with it, and a bad tick here trips every stop in the book.
    max_bar_move_pct: float = 0.35
    # Gaps up to this many bars are backfilled quietly; beyond it, degrade.
    max_autofill_gap_bars: int = 10
    clock_skew_tolerance: timedelta = timedelta(seconds=5)


@dataclass
class _Recovery:
    clean_bars: int = 0


class DataIntegrityMonitor:
    """Per-(symbol, timeframe) health. Feed it bars; ask it what is tradeable.

    `accept(bar)` returns True if the bar should be stored. Rejecting is not an
    error — duplicates on reconnect are routine — but every rejection is counted
    so "why is this symbol quiet?" has an answer.
    """

    def __init__(self, config: IntegrityConfig | None = None) -> None:
        self.config = config or IntegrityConfig()
        self.health: dict[tuple[str, str], SymbolHealth] = {}
        self._recovery: dict[tuple[str, str], _Recovery] = {}
        self.faults: list[Fault] = []

    # ── registration ─────────────────────────────────────────────────────

    def track(self, symbol: str, timeframe: str) -> SymbolHealth:
        key = (symbol, timeframe)
        h = self.health.get(key)
        if h is None:
            timeframe_seconds(timeframe)         # fail fast on a bad timeframe
            h = SymbolHealth(symbol=symbol, timeframe=timeframe)
            self.health[key] = h
            self._recovery[key] = _Recovery()
        return h

    # ── the hot path ─────────────────────────────────────────────────────

    def accept(self, bar: Bar, *, now: datetime | None = None) -> bool:
        """Validate and record one closed bar. True if it should be stored."""
        tf = bar.timeframe or "1h"
        h = self.track(bar.symbol, tf)
        now = now or bar.ts
        h.last_update = now

        bad = self._implausible(bar)
        if bad is not None:
            h.implausible_dropped += 1
            self._degrade(h, FaultKind.IMPLAUSIBLE, bad, now)
            return False

        if h.last_bar_ts is not None:
            if bar.ts == h.last_bar_ts:
                h.duplicates_dropped += 1
                return False                     # routine on reconnect
            if bar.ts < h.last_bar_ts:
                h.out_of_order_dropped += 1
                self._degrade(h, FaultKind.OUT_OF_ORDER,
                              f"bar {bar.ts.isoformat()} arrived after "
                              f"{h.last_bar_ts.isoformat()}", now)
                return False

            step = timeframe_seconds(tf)
            elapsed = (bar.ts - h.last_bar_ts).total_seconds()
            missing = int(elapsed // step) - 1
            if missing > 0:
                h.gaps_detected += 1
                h.missing_bars += missing
                if missing > self.config.max_autofill_gap_bars:
                    self._degrade(h, FaultKind.GAP,
                                  f"{missing} bars missing between "
                                  f"{h.last_bar_ts.isoformat()} and "
                                  f"{bar.ts.isoformat()}", now)
                    # This bar is stored and is the newest we have, but it must
                    # NOT count toward recovery: it is the bar that *revealed*
                    # the fault. Counting it made every settling period one bar
                    # shorter than configured — a weaker guarantee than the one
                    # advertised, which is the worst kind of off-by-one.
                    h.last_bar_ts = bar.ts
                    h.bars_seen += 1
                    return True
                log.info("%s %s: %d bar gap, backfillable", bar.symbol, tf,
                         missing)

        h.last_bar_ts = bar.ts
        h.bars_seen += 1
        self._on_clean_bar(h, now)
        return True

    def mark_backfilled(self, symbol: str, timeframe: str, newest: datetime,
                        *, now: datetime | None = None) -> None:
        """Advance the cursor after a REST backfill repaired the stored data.

        Without this the monitor keeps the pre-disconnect `last_bar_ts`, so the
        first live bar after a reconnect is measured against a hole that has
        already been filled — and the feed degrades on a gap that no longer
        exists. A **false halt**, and the worst kind: it fires exactly when the
        system has just successfully recovered.

        Deliberately does NOT run fault detection over the backfilled bars.
        Backfill covers a known hole, and re-deriving "there was a gap" from
        data fetched *because* there was a gap would degrade on every recovery.
        """
        h = self.track(symbol, timeframe)
        if h.last_bar_ts is None or newest > h.last_bar_ts:
            h.last_bar_ts = newest
        h.last_update = now or newest
        # A successful repair is progress toward recovery, but only one bar's
        # worth — the settling period still has to be earned from live data.
        self._on_clean_bar(h, h.last_update)

    def check_staleness(self, now: datetime) -> list[Fault]:
        """Call on a timer. Detects the fault a quiet feed cannot report itself.

        A websocket that stops delivering looks exactly like a market with no
        trades. Only elapsed time distinguishes them, so this cannot be driven
        by incoming bars — it has to be driven by the clock.
        """
        new: list[Fault] = []
        for h in self.health.values():
            if h.last_bar_ts is None:
                continue
            age = h.age_bars(now)
            if age > self.config.max_stale_bars:
                if h.state is not IntegrityState.DEGRADED or (
                        h.fault and h.fault.kind is not FaultKind.STALE):
                    f = self._degrade(
                        h, FaultKind.STALE,
                        f"newest bar is {age} intervals old "
                        f"({h.last_bar_ts.isoformat()})", now)
                    new.append(f)
        return new

    # ── state ────────────────────────────────────────────────────────────

    def _implausible(self, bar: Bar) -> str | None:
        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            return f"non-positive price in OHLC {bar.open}/{bar.high}/{bar.low}/{bar.close}"
        if bar.high < bar.low:
            return f"high {bar.high} < low {bar.low}"
        if not (bar.low <= bar.open <= bar.high):
            return f"open {bar.open} outside [{bar.low}, {bar.high}]"
        if not (bar.low <= bar.close <= bar.high):
            return f"close {bar.close} outside [{bar.low}, {bar.high}]"
        if bar.volume < 0:
            return f"negative volume {bar.volume}"
        move = (bar.high - bar.low) / bar.low if bar.low > 0 else 0.0
        if move > self.config.max_bar_move_pct:
            return (f"intrabar move {move:.1%} exceeds "
                    f"{self.config.max_bar_move_pct:.0%}; treating as bad data")
        return None

    def _degrade(self, h: SymbolHealth, kind: FaultKind, detail: str,
                 now: datetime) -> Fault:
        fault = Fault(kind, h.symbol, h.timeframe, detail, now)
        was = h.state
        h.state = IntegrityState.DEGRADED
        h.fault = fault
        h.healthy_since = None
        self._recovery[(h.symbol, h.timeframe)] = _Recovery()
        self.faults.append(fault)
        if was is not IntegrityState.DEGRADED:
            log.warning("DATA INTEGRITY DEGRADED — %s", fault)
        return fault

    def _on_clean_bar(self, h: SymbolHealth, now: datetime) -> None:
        key = (h.symbol, h.timeframe)
        rec = self._recovery.setdefault(key, _Recovery())
        if h.state is IntegrityState.OK:
            return
        rec.clean_bars += 1
        if rec.clean_bars >= self.config.recovery_bars:
            previous = h.state
            h.state = IntegrityState.OK
            h.fault = None
            h.healthy_since = now
            rec.clean_bars = 0
            if previous is IntegrityState.DEGRADED:
                log.info("data integrity RESTORED — %s %s after %d clean bars",
                         h.symbol, h.timeframe, self.config.recovery_bars)

    # ── what the engine asks ─────────────────────────────────────────────

    def tradeable(self, symbol: str, timeframe: str) -> bool:
        h = self.health.get((symbol, timeframe))
        return h is not None and h.is_tradeable

    def degraded_symbols(self) -> list[str]:
        """Symbols with an actual fault. **Not** the same as "not OK".

        `UNKNOWN` means no live bar has arrived on some timeframe yet, which
        after a restart is the normal state for up to a full interval — a whole
        day on `1d`. Reporting that as *degraded* made a freshly restarted,
        perfectly healthy collector look broken on the dashboard. The two states
        are deliberately distinct in `IntegrityState`; collapsing them here threw
        that distinction away at the only place anybody reads it.
        """
        return sorted({h.symbol for h in self.health.values()
                       if h.state is IntegrityState.DEGRADED})

    def unknown_symbols(self) -> list[str]:
        """Symbols still waiting for their first live bar on some timeframe."""
        return sorted({h.symbol for h in self.health.values()
                       if h.state is IntegrityState.UNKNOWN})

    def data_age_bars(self, now: datetime) -> dict[str, int]:
        """Exactly the shape `AccountState.data_age_bars` wants, so the risk
        engine's STALE_DATA gate is fed by the same numbers reported here
        rather than a second, subtly different calculation."""
        out: dict[str, int] = {}
        for h in self.health.values():
            age = h.age_bars(now)
            out[h.symbol] = max(out.get(h.symbol, 0), age)
        return out

    def summary(self) -> str:
        if not self.health:
            return "no symbols tracked"
        rows = []
        for h in sorted(self.health.values(), key=lambda x: (x.symbol, x.timeframe)):
            flag = {IntegrityState.OK: "ok",
                    IntegrityState.DEGRADED: "DEGRADED",
                    IntegrityState.UNKNOWN: "unknown"}[h.state]
            rows.append(f"{h.symbol} {h.timeframe}: {flag} "
                        f"bars={h.bars_seen} gaps={h.gaps_detected} "
                        f"dupes={h.duplicates_dropped} bad={h.implausible_dropped}"
                        + (f" — {h.fault.detail}" if h.fault else ""))
        return "\n".join(rows)


__all__ = [
    "TIMEFRAME_SECONDS",
    "DataIntegrityMonitor",
    "Fault",
    "FaultKind",
    "IntegrityConfig",
    "IntegrityState",
    "SymbolHealth",
    "timeframe_seconds",
]
