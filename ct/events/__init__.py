"""Event bus, durable log, and replay.

Events are an **output**, never the order-path control flow — see
`ct/events/types.py` and PHASE2-AUDIT.md §7.1.

    from ct.events import EventBus, JsonlEventLog, order_filled

    bus = EventBus(JsonlEventLog("/app/state/events"), source="engine")
    bus.publish(order_filled(cid, "BTC/USD", "buy", qty, px, fee, is_maker=True,
                             correlation_id=cor))
"""

from ct.events.bus import BusStats, EventBus, Handler, NullEventBus, SubscriberStats
from ct.events.log import EventLog, InMemoryEventLog, JsonlEventLog
from ct.events.replay import (
    TradeLifecycle,
    causal_chain,
    execution_quality,
    lifecycle,
    lifecycles,
    rebuild_positions,
    summarize,
    timeline,
)
from ct.events.types import (
    Event,
    EventType,
    Severity,
    bar_closed,
    data_gap_detected,
    exchange_disconnected,
    heartbeat,
    new_correlation_id,
    order_filled,
    order_rejected,
    order_submitted,
    portfolio_updated,
    position_closed,
    position_opened,
    reconciliation_mismatch,
    risk_approved,
    risk_limit_exceeded,
    signal_generated,
    signal_rejected,
    strategy_promoted,
    strategy_retired,
)

__all__ = [
    # bus
    "EventBus", "NullEventBus", "BusStats", "SubscriberStats", "Handler",
    # log
    "EventLog", "JsonlEventLog", "InMemoryEventLog",
    # types
    "Event", "EventType", "Severity", "new_correlation_id",
    # factories
    "bar_closed", "data_gap_detected", "signal_generated", "signal_rejected",
    "risk_approved", "risk_limit_exceeded", "order_submitted", "order_filled",
    "order_rejected", "position_opened", "position_closed", "portfolio_updated",
    "heartbeat", "exchange_disconnected", "reconciliation_mismatch",
    "strategy_promoted", "strategy_retired",
    # replay
    "TradeLifecycle", "lifecycle", "lifecycles", "timeline", "causal_chain",
    "rebuild_positions", "summarize", "execution_quality",
]
