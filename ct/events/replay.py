"""Replay and projections over the event log.

The log is append-only and totally ordered, which makes three things possible
that are otherwise guesswork after the fact:

  * **Trade replay** — every event of one lifecycle, keyed by `correlation_id`,
    from the signal that proposed it to the position that closed.
  * **Execution quality** — the latencies between those events are the raw
    material for "what did execution actually cost us?", which is the only way
    to check whether the maker-first thesis (ARCHITECTURE.md §0.1) is real.
  * **An independent position reconstruction** — `rebuild_positions` replays
    fills from the log alone. It shares no code with the live position
    bookkeeping, so agreement between them is real evidence rather than the
    same bug twice. It is a genuine third opinion for the reconciliation
    invariant in §6.3.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from ct.events.log import EventLog
from ct.events.types import Event, EventType

_ORDER_EVENTS = frozenset({
    EventType.ORDER_SUBMITTED, EventType.ORDER_ACCEPTED, EventType.ORDER_FILLED,
    EventType.ORDER_REJECTED, EventType.ORDER_CANCELLED, EventType.ORDER_EXPIRED,
})


@dataclass
class TradeLifecycle:
    """Every event sharing one `correlation_id`, in sequence order."""

    correlation_id: str
    events: list[Event] = field(default_factory=list)

    # ── phases ──
    def first(self, kind: EventType) -> Event | None:
        return next((e for e in self.events if e.type is kind), None)

    def all_of(self, kind: EventType) -> list[Event]:
        return [e for e in self.events if e.type is kind]

    @property
    def signal(self) -> Event | None:
        return self.first(EventType.SIGNAL_GENERATED)

    @property
    def rejection(self) -> Event | None:
        return self.first(EventType.SIGNAL_REJECTED)

    @property
    def submitted(self) -> Event | None:
        return self.first(EventType.ORDER_SUBMITTED)

    @property
    def fills(self) -> list[Event]:
        return self.all_of(EventType.ORDER_FILLED)

    @property
    def closed(self) -> Event | None:
        return self.first(EventType.POSITION_CLOSED)

    @property
    def outcome(self) -> str:
        if self.rejection is not None:
            return f"rejected:{self.rejection.payload.get('code', '?')}"
        if self.closed is not None:
            return "closed"
        if self.fills:
            return "open"
        if self.submitted is not None:
            return "submitted"
        return "signalled"

    # ── execution quality ──
    @property
    def decision_latency(self) -> float | None:
        """Signal → order submitted, in seconds. Our own thinking time."""
        s, o = self.signal, self.submitted
        return None if s is None or o is None else (o.ts - s.ts).total_seconds()

    @property
    def fill_latency(self) -> float | None:
        """Order submitted → first fill, in seconds. The venue's time plus our
        queue position — the number that decides whether resting as maker to
        save 15 bps was worth it."""
        o = self.submitted
        f = self.fills
        return None if o is None or not f else (f[0].ts - o.ts).total_seconds()

    @property
    def filled_qty(self) -> float:
        return sum(float(e.payload.get("qty") or 0) for e in self.fills)

    @property
    def fees_paid(self) -> float:
        return sum(float(e.payload.get("fee") or 0) for e in self.fills)

    @property
    def maker_ratio(self) -> float | None:
        """Fraction of filled quantity that rested as maker. Directly measures
        the largest quantified edge in the project."""
        f = self.fills
        if not f:
            return None
        total = sum(float(e.payload.get("qty") or 0) for e in f)
        if total <= 0:
            return None
        maker = sum(float(e.payload.get("qty") or 0) for e in f
                    if e.payload.get("is_maker"))
        return maker / total

    @property
    def slippage_bps(self) -> float | None:
        """Signal reference price → volume-weighted fill price, in bps, signed
        so positive always means *worse for us*."""
        s = self.signal
        f = self.fills
        if s is None or not f:
            return None
        ref = s.payload.get("ref_price")
        if not ref:
            return None
        qty = sum(float(e.payload.get("qty") or 0) for e in f)
        if qty <= 0:
            return None
        vwap = sum(float(e.payload.get("price") or 0) * float(e.payload.get("qty") or 0)
                   for e in f) / qty
        direction = 1 if str(f[0].payload.get("side")) == "buy" else -1
        return (vwap - float(ref)) / float(ref) * 10_000.0 * direction

    def timeline(self) -> list[tuple[float, str, str]]:
        """(seconds since first event, type, one-line summary) — the shape the
        dashboard's execution timeline renders."""
        if not self.events:
            return []
        t0 = self.events[0].ts
        return [((e.ts - t0).total_seconds(), e.type.value, _summarize(e))
                for e in self.events]


def _summarize(e: Event) -> str:
    p = e.payload
    if e.type is EventType.SIGNAL_GENERATED:
        return f"{p.get('strategy')} {p.get('symbol')} dir={p.get('direction')}"
    if e.type is EventType.SIGNAL_REJECTED:
        return f"{p.get('code')}: {p.get('detail')}"
    if e.type is EventType.ORDER_SUBMITTED:
        return f"{p.get('side')} {p.get('qty')} {p.get('symbol')} ({p.get('order_type')})"
    if e.type is EventType.ORDER_FILLED:
        tag = "maker" if p.get("is_maker") else "taker"
        return f"{p.get('qty')} @ {p.get('price')} ({tag}, fee {p.get('fee')})"
    if e.type is EventType.POSITION_CLOSED:
        return f"{p.get('exit_reason')} pnl={p.get('net_pnl')}"
    return ", ".join(f"{k}={v}" for k, v in list(p.items())[:3])


# ── entry points ─────────────────────────────────────────────────────────

def lifecycle(source: EventLog | Iterable[Event], correlation_id: str) -> TradeLifecycle:
    events = _events(source, correlation_id=correlation_id)
    return TradeLifecycle(correlation_id, sorted(events, key=lambda e: e.seq))


def lifecycles(
    source: EventLog | Iterable[Event],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[TradeLifecycle]:
    """Every correlated lifecycle in the window, oldest first.

    Events with no `correlation_id` (heartbeats, portfolio marks, ops) are not
    part of any lifecycle and are excluded by design.
    """
    grouped: dict[str, list[Event]] = defaultdict(list)
    for e in _events(source, start=start, end=end):
        if e.correlation_id:
            grouped[e.correlation_id].append(e)
    out = [TradeLifecycle(cid, sorted(evs, key=lambda e: e.seq))
           for cid, evs in grouped.items()]
    out.sort(key=lambda lc: lc.events[0].seq if lc.events else 0)
    return out


def timeline(
    source: EventLog | Iterable[Event],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    types: frozenset[EventType] | None = None,
    limit: int | None = None,
) -> list[Event]:
    return sorted(_events(source, start=start, end=end, types=types, limit=limit),
                  key=lambda e: e.seq)


def causal_chain(source: EventLog | Iterable[Event], seq: int) -> list[Event]:
    """Walk `causation_id` back from `seq` to the root cause, oldest first."""
    by_seq = {e.seq: e for e in _events(source)}
    chain: list[Event] = []
    cur = by_seq.get(seq)
    seen: set[int] = set()
    while cur is not None and cur.seq not in seen:
        seen.add(cur.seq)
        chain.append(cur)
        cur = by_seq.get(cur.causation_id) if cur.causation_id else None
    return list(reversed(chain))


def rebuild_positions(source: EventLog | Iterable[Event]) -> dict[str, Decimal]:
    """Net quantity per symbol, replayed from ORDER_FILLED events alone.

    Shares no code with the live position bookkeeping — that is the entire
    value. Agreement between this and `positions` is evidence; a mismatch is a
    `RECONCILIATION_MISMATCH` waiting to happen.
    """
    net: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    for e in _events(source, types=frozenset({EventType.ORDER_FILLED})):
        sym = str(e.payload.get("symbol"))
        qty = Decimal(str(e.payload.get("qty") or 0))
        sign = 1 if str(e.payload.get("side")) == "buy" else -1
        net[sym] += qty * sign
    return {s: q for s, q in net.items() if q != 0}


def summarize(source: EventLog | Iterable[Event]) -> dict[str, int]:
    """Counts by event type — the cheapest health check there is."""
    return dict(Counter(e.type.value for e in _events(source)))


def execution_quality(lcs: Iterable[TradeLifecycle]) -> dict[str, float | int]:
    """Aggregate the numbers that say whether execution is working.

    `maker_ratio` is the headline: at Kraken's tier, resting as maker instead
    of crossing saves 15 bps per side. If this sits near zero in production,
    the maker-first design is not being realised and the whole low-frequency
    thesis needs re-examining.
    """
    lcs = [lc for lc in lcs if lc.fills]
    if not lcs:
        return {"lifecycles": 0}
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "lifecycles": len(lcs),
        "mean_decision_latency_s": _mean([v for lc in lcs
                                          if (v := lc.decision_latency) is not None]),
        "mean_fill_latency_s": _mean([v for lc in lcs
                                      if (v := lc.fill_latency) is not None]),
        "mean_slippage_bps": _mean([v for lc in lcs
                                    if (v := lc.slippage_bps) is not None]),
        "maker_ratio": _mean([v for lc in lcs if (v := lc.maker_ratio) is not None]),
        "total_fees": sum(lc.fees_paid for lc in lcs),
    }


def _events(source: EventLog | Iterable[Event], **kw) -> Iterator[Event]:
    if isinstance(source, EventLog):
        return source.read(**kw)
    from ct.events.log import _filter
    return _filter(list(source), **kw)


__all__ = [
    "TradeLifecycle",
    "causal_chain",
    "execution_quality",
    "lifecycle",
    "lifecycles",
    "rebuild_positions",
    "summarize",
    "timeline",
]
