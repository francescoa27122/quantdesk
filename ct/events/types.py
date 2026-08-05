"""Event taxonomy.

**Events are an OUTPUT, not control flow.** The order path
(`Signal → Risk → Order → Broker → Fill → Position`) stays a synchronous
in-process call chain; every state transition *emits* an event afterwards.
Nothing in the trading path ever waits on, or reads from, the bus. See
PHASE2-AUDIT.md §7.1 for why: at-least-once delivery and reordering on the one
path where a duplicate means a double-sized position is a reliability
regression, not an architecture.

Consequences of that decision, encoded here:

  * `Event` is frozen and carries no callbacks, futures or handles — an event
    cannot be used to *ask* for anything, only to report that something
    happened. It is past tense by construction, which is why every `EventType`
    is named in the past tense.
  * Payloads are plain JSON-able dicts, so the log is readable with `jq` on the
    NAS and does not need this codebase to be interpreted.

Construction goes through the factory functions at the bottom rather than raw
`Event(...)` calls. Forty payload dataclasses would be forty things to keep in
sync with a JSON column; the factories give the same guarantee at the point
that actually matters — the call site — while keeping the wire format one shape.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ct.models import utcnow


class EventType(str, Enum):
    """Every event the platform can emit. Values are stable — they are written
    to the log and to the database, and old logs must stay readable."""

    # ── market data ──
    MARKET_DATA_RECEIVED = "market_data.received"
    BAR_CLOSED = "market_data.bar_closed"
    DATA_GAP_DETECTED = "market_data.gap_detected"
    DATA_INTEGRITY_DEGRADED = "market_data.integrity_degraded"
    DATA_INTEGRITY_RESTORED = "market_data.integrity_restored"

    # ── strategy ──
    SIGNAL_GENERATED = "strategy.signal_generated"
    SIGNAL_REJECTED = "strategy.signal_rejected"
    STRATEGY_QUARANTINED = "strategy.quarantined"
    STRATEGY_PAUSED = "strategy.paused"
    STRATEGY_RESUMED = "strategy.resumed"
    STRATEGY_PROMOTED = "strategy.promoted"
    STRATEGY_RETIRED = "strategy.retired"

    # ── risk ──
    RISK_APPROVED = "risk.approved"
    RISK_LIMIT_EXCEEDED = "risk.limit_exceeded"
    CIRCUIT_BREAKER_TRIPPED = "risk.circuit_breaker_tripped"
    KILL_SWITCH_ENGAGED = "risk.kill_switch_engaged"
    KILL_SWITCH_RELEASED = "risk.kill_switch_released"

    # ── execution ──
    ORDER_SUBMITTED = "execution.order_submitted"
    ORDER_ACCEPTED = "execution.order_accepted"
    ORDER_FILLED = "execution.order_filled"
    ORDER_REJECTED = "execution.order_rejected"
    ORDER_CANCELLED = "execution.order_cancelled"
    ORDER_EXPIRED = "execution.order_expired"

    # ── portfolio ──
    POSITION_OPENED = "portfolio.position_opened"
    POSITION_CHANGED = "portfolio.position_changed"
    POSITION_CLOSED = "portfolio.position_closed"
    PORTFOLIO_UPDATED = "portfolio.updated"

    # ── arming (see ARCHITECTURE.md §8.4) ──
    ARMING_CREATED = "arming.created"
    ARMING_RENEWED = "arming.renewed"
    ARMING_DISARMED = "arming.disarmed"
    ARMING_RENEWAL_REFUSED = "arming.renewal_refused"

    # ── operations ──
    HEARTBEAT = "ops.heartbeat"
    ENGINE_STARTED = "ops.engine_started"
    ENGINE_STOPPING = "ops.engine_stopping"
    ENGINE_HALTED = "ops.engine_halted"
    ENGINE_RESUMED = "ops.engine_resumed"
    EXCHANGE_CONNECTED = "ops.exchange_connected"
    EXCHANGE_DISCONNECTED = "ops.exchange_disconnected"
    RECONCILIATION_OK = "ops.reconciliation_ok"
    RECONCILIATION_MISMATCH = "ops.reconciliation_mismatch"
    CLOCK_DRIFT_DETECTED = "ops.clock_drift_detected"

    # ── the bus reporting on itself ──
    SUBSCRIBER_LAGGED = "bus.subscriber_lagged"
    SUBSCRIBER_FAILED = "bus.subscriber_failed"
    SUBSCRIBER_QUARANTINED = "bus.subscriber_quarantined"


class Severity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Events whose loss would leave the system unable to answer "what did we do
# with the money?". These are fsync'd on write rather than batched. Everything
# else is durable within the flush interval, which is the right trade on a
# 4-core NAS — fsyncing a heartbeat is pure cost.
_CRITICAL_TYPES = frozenset({
    EventType.ORDER_SUBMITTED,
    EventType.ORDER_ACCEPTED,
    EventType.ORDER_FILLED,
    EventType.ORDER_REJECTED,
    EventType.ORDER_CANCELLED,
    EventType.POSITION_OPENED,
    EventType.POSITION_CLOSED,
    EventType.RISK_LIMIT_EXCEEDED,
    EventType.CIRCUIT_BREAKER_TRIPPED,
    EventType.KILL_SWITCH_ENGAGED,
    EventType.ARMING_CREATED,
    EventType.ARMING_RENEWED,
    EventType.ARMING_DISARMED,
    EventType.RECONCILIATION_MISMATCH,
})


def new_correlation_id() -> str:
    """Groups every event of one trade lifecycle: signal → risk → order → fill
    → position close. This is what makes trade replay and the execution
    timeline possible at all."""
    return f"cor-{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened. Immutable, past tense, JSON-able.

    `seq` is 0 until the bus assigns it — the bus is the only sequencer, which
    is what makes the log totally ordered.
    """
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=utcnow)
    seq: int = 0
    source: str = "unknown"
    severity: Severity = Severity.INFO
    # Groups a whole lifecycle. Set once at the signal and threaded through.
    correlation_id: str | None = None
    # The event that caused this one — lets replay build a causal tree rather
    # than just a flat timeline.
    causation_id: int | None = None

    @property
    def is_critical(self) -> bool:
        return self.type in _CRITICAL_TYPES or self.severity is Severity.CRITICAL

    def with_seq(self, seq: int) -> Event:
        return replace(self, seq=seq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts.isoformat(),
            "type": self.type.value,
            "source": self.source,
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": self.payload,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Event:
        return Event(
            type=EventType(d["type"]),
            payload=d.get("payload") or {},
            ts=datetime.fromisoformat(d["ts"]),
            seq=int(d.get("seq", 0)),
            source=d.get("source", "unknown"),
            severity=Severity(d.get("severity", "info")),
            correlation_id=d.get("correlation_id"),
            causation_id=d.get("causation_id"),
        )


def _num(x: Decimal | float | int | None) -> float | None:
    """Decimals do not survive JSON. Stored as float for the log, which is an
    observability artifact — the authoritative money figures live in the
    `fills` and `positions` tables, not here."""
    return None if x is None else float(x)


# ── factories ────────────────────────────────────────────────────────────
# One per meaningful transition. They exist so payload keys are decided once,
# here, rather than at forty call sites that will drift apart.

def bar_closed(symbol: str, timeframe: str, close: float, ts: datetime,
               **extra: Any) -> Event:
    return Event(EventType.BAR_CLOSED, {"symbol": symbol, "timeframe": timeframe,
                                        "close": close, "bar_ts": ts.isoformat(),
                                        **extra}, source="collector")


def data_gap_detected(symbol: str, timeframe: str, missing_bars: int,
                      from_ts: datetime, to_ts: datetime) -> Event:
    return Event(EventType.DATA_GAP_DETECTED,
                 {"symbol": symbol, "timeframe": timeframe,
                  "missing_bars": missing_bars, "from": from_ts.isoformat(),
                  "to": to_ts.isoformat()},
                 severity=Severity.WARNING, source="collector")


def signal_generated(strategy: str, symbol: str, direction: int, strength: float,
                     correlation_id: str, **extra: Any) -> Event:
    return Event(EventType.SIGNAL_GENERATED,
                 {"strategy": strategy, "symbol": symbol, "direction": direction,
                  "strength": strength, **extra},
                 correlation_id=correlation_id, source="engine")


def signal_rejected(strategy: str, symbol: str, code: str, detail: str,
                    correlation_id: str, causation_id: int | None = None,
                    **extra: Any) -> Event:
    """Rejections are events too. Without them you cannot tell 'the strategy is
    bad' from 'risk is strangling it'."""
    return Event(EventType.SIGNAL_REJECTED,
                 {"strategy": strategy, "symbol": symbol, "code": code,
                  "detail": detail, **extra},
                 correlation_id=correlation_id, causation_id=causation_id,
                 source="risk")


def risk_approved(strategy: str, symbol: str, qty: Decimal, binding_sizer: str,
                  correlation_id: str, causation_id: int | None = None) -> Event:
    return Event(EventType.RISK_APPROVED,
                 {"strategy": strategy, "symbol": symbol, "qty": _num(qty),
                  "binding_sizer": binding_sizer},
                 correlation_id=correlation_id, causation_id=causation_id,
                 source="risk")


def risk_limit_exceeded(code: str, detail: str, **extra: Any) -> Event:
    return Event(EventType.RISK_LIMIT_EXCEEDED, {"code": code, "detail": detail,
                                                 **extra},
                 severity=Severity.CRITICAL, source="risk")


def order_submitted(client_order_id: str, symbol: str, side: str, qty: Decimal,
                    order_type: str, limit_price: Decimal | None,
                    correlation_id: str, causation_id: int | None = None) -> Event:
    return Event(EventType.ORDER_SUBMITTED,
                 {"client_order_id": client_order_id, "symbol": symbol,
                  "side": side, "qty": _num(qty), "order_type": order_type,
                  "limit_price": _num(limit_price)},
                 correlation_id=correlation_id, causation_id=causation_id,
                 source="execution")


def order_filled(client_order_id: str, symbol: str, side: str, qty: Decimal,
                 price: Decimal, fee: Decimal, is_maker: bool,
                 correlation_id: str | None = None,
                 causation_id: int | None = None) -> Event:
    return Event(EventType.ORDER_FILLED,
                 {"client_order_id": client_order_id, "symbol": symbol,
                  "side": side, "qty": _num(qty), "price": _num(price),
                  "fee": _num(fee), "is_maker": is_maker},
                 correlation_id=correlation_id, causation_id=causation_id,
                 source="execution")


def order_rejected(client_order_id: str, symbol: str, error: str,
                   correlation_id: str | None = None) -> Event:
    return Event(EventType.ORDER_REJECTED,
                 {"client_order_id": client_order_id, "symbol": symbol,
                  "error": error},
                 severity=Severity.WARNING, correlation_id=correlation_id,
                 source="execution")


def position_opened(symbol: str, qty: Decimal, avg_entry: Decimal, strategy: str,
                    correlation_id: str | None = None) -> Event:
    return Event(EventType.POSITION_OPENED,
                 {"symbol": symbol, "qty": _num(qty),
                  "avg_entry": _num(avg_entry), "strategy": strategy},
                 correlation_id=correlation_id, source="engine")


def position_closed(symbol: str, strategy: str, net_pnl: Decimal,
                    return_pct: float, exit_reason: str,
                    correlation_id: str | None = None) -> Event:
    return Event(EventType.POSITION_CLOSED,
                 {"symbol": symbol, "strategy": strategy,
                  "net_pnl": _num(net_pnl), "return_pct": return_pct,
                  "exit_reason": exit_reason},
                 correlation_id=correlation_id, source="engine")


def portfolio_updated(equity: Decimal, cash: Decimal, gross_exposure: float,
                      drawdown: float, open_positions: int) -> Event:
    return Event(EventType.PORTFOLIO_UPDATED,
                 {"equity": _num(equity), "cash": _num(cash),
                  "gross_exposure": gross_exposure, "drawdown": drawdown,
                  "open_positions": open_positions},
                 severity=Severity.DEBUG, source="engine")


def heartbeat(component: str, **extra: Any) -> Event:
    return Event(EventType.HEARTBEAT, {"component": component, **extra},
                 severity=Severity.DEBUG, source=component)


def exchange_disconnected(exchange: str, reason: str) -> Event:
    return Event(EventType.EXCHANGE_DISCONNECTED,
                 {"exchange": exchange, "reason": reason},
                 severity=Severity.CRITICAL, source="collector")


def reconciliation_mismatch(symbol: str, ours: Decimal, theirs: Decimal) -> Event:
    return Event(EventType.RECONCILIATION_MISMATCH,
                 {"symbol": symbol, "ours": _num(ours), "theirs": _num(theirs),
                  "delta": _num(ours - theirs)},
                 severity=Severity.CRITICAL, source="engine")


def strategy_promoted(strategy: str, from_state: str, to_state: str,
                      metrics: dict[str, Any]) -> Event:
    return Event(EventType.STRATEGY_PROMOTED,
                 {"strategy": strategy, "from": from_state, "to": to_state,
                  "metrics": metrics}, source="worker")


def strategy_retired(strategy: str, reason: str, metrics: dict[str, Any]) -> Event:
    return Event(EventType.STRATEGY_RETIRED,
                 {"strategy": strategy, "reason": reason, "metrics": metrics},
                 severity=Severity.WARNING, source="worker")


__all__ = [
    "Event",
    "EventType",
    "Severity",
    "bar_closed",
    "data_gap_detected",
    "exchange_disconnected",
    "heartbeat",
    "new_correlation_id",
    "order_filled",
    "order_rejected",
    "order_submitted",
    "portfolio_updated",
    "position_closed",
    "position_opened",
    "reconciliation_mismatch",
    "risk_approved",
    "risk_limit_exceeded",
    "signal_generated",
    "signal_rejected",
    "strategy_promoted",
    "strategy_retired",
]
