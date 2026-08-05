"""Event bus, log and replay tests.

The load-bearing ones are the isolation and durability tests: `publish()` is
called from the order path, so a bus that can raise, block, or lose a fill is a
bus that makes the system less reliable than having no observability at all.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ct.events import (
    EventBus,
    EventType,
    InMemoryEventLog,
    JsonlEventLog,
    NullEventBus,
    Severity,
    causal_chain,
    execution_quality,
    lifecycle,
    lifecycles,
    new_correlation_id,
    order_filled,
    order_submitted,
    position_closed,
    rebuild_positions,
    signal_generated,
    signal_rejected,
    summarize,
    timeline,
)
from ct.events.types import Event, heartbeat

TS = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _bus(tmp_path=None, **kw) -> EventBus:
    log = JsonlEventLog(tmp_path) if tmp_path else InMemoryEventLog()
    return EventBus(log, source="test", **kw)


# ══════════════════════════════════════════════════════════════════════════
# The contract: publish never raises, never blocks
# ══════════════════════════════════════════════════════════════════════════

def test_publish_survives_a_subscriber_that_raises():
    bus = _bus()
    calls = {"good": 0}

    def explodes(_e):
        raise RuntimeError("boom")

    def good(_e):
        calls["good"] += 1

    bus.subscribe(explodes, name="explodes")
    bus.subscribe(good, name="good")

    for _ in range(3):
        bus.publish(heartbeat("engine"))     # must not raise

    # The healthy subscriber kept receiving throughout.
    assert calls["good"] >= 3
    assert bus.stats.subscribers["explodes"].failures >= 3


def test_repeatedly_failing_subscriber_is_quarantined_not_retried_forever():
    """Degrade to no observability, never to a broken engine."""
    bus = _bus(max_handler_failures=3)
    seen = {"n": 0}

    def explodes(_e):
        seen["n"] += 1
        raise ValueError("nope")

    bus.subscribe(explodes, name="bad")
    for _ in range(10):
        bus.publish(heartbeat("engine"))

    assert "bad" in bus.stats.quarantined()
    # Called at most until quarantine, then never again.
    assert seen["n"] == 3
    q = [e for e in bus.log.read()
         if e.type is EventType.SUBSCRIBER_QUARANTINED]
    assert len(q) == 1 and q[0].payload["subscriber"] == "bad"


def test_publish_survives_a_dead_log():
    """A failed disk must cost observability, not the ability to place a stop."""
    class DeadLog(InMemoryEventLog):
        def append(self, event):
            raise OSError("disk gone")

    bus = EventBus(DeadLog(), source="test")
    got = bus.publish(heartbeat("engine"))    # must not raise
    assert got.seq == 1
    assert bus.stats.log_failures == 1
    assert bus.stats.published == 1


def test_sequence_is_monotonic_and_gapless_under_threads():
    import threading
    bus = _bus()

    def worker():
        for _ in range(50):
            bus.publish(heartbeat("t"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    seqs = sorted(e.seq for e in bus.log.read())
    assert seqs == list(range(1, 201)), "sequence must be gapless and unique"


def test_publish_returns_the_sequenced_event_for_causation_chaining():
    bus = _bus()
    a = bus.publish(heartbeat("engine"))
    b = bus.publish(Event(EventType.ORDER_SUBMITTED, {}, causation_id=a.seq))
    assert b.causation_id == a.seq
    assert b.seq == a.seq + 1


# ══════════════════════════════════════════════════════════════════════════
# Durability: the log is the authority
# ══════════════════════════════════════════════════════════════════════════

def test_jsonl_log_round_trips_every_field(tmp_path):
    log = JsonlEventLog(tmp_path)
    cor = new_correlation_id()
    original = order_filled("cid-1", "BTC/USD", "buy", Decimal("0.5"),
                            Decimal(60_000), Decimal("12.5"), is_maker=True,
                            correlation_id=cor, causation_id=7).with_seq(42)
    log.append(original)
    log.close()

    back = list(JsonlEventLog(tmp_path).read())
    assert len(back) == 1
    r = back[0]
    assert r.seq == 42
    assert r.type is EventType.ORDER_FILLED
    assert r.correlation_id == cor
    assert r.causation_id == 7
    assert r.payload["qty"] == 0.5
    assert r.payload["is_maker"] is True


def test_sequence_continues_across_a_restart(tmp_path):
    """Restarting at 1 would put two different events at the same sequence and
    silently corrupt every replay that spans the restart."""
    bus1 = EventBus(JsonlEventLog(tmp_path), source="test")
    for _ in range(5):
        bus1.publish(heartbeat("engine"))
    bus1.close()

    bus2 = EventBus(JsonlEventLog(tmp_path), source="test")
    assert bus2.sequence == 5
    e = bus2.publish(heartbeat("engine"))
    assert e.seq == 6
    bus2.close()

    seqs = sorted(x.seq for x in JsonlEventLog(tmp_path).read())
    assert seqs == [1, 2, 3, 4, 5, 6]


def test_non_critical_events_become_visible_to_another_reader_on_a_timer(tmp_path):
    """`qd-web` reads the file `qd-engine` writes, from a different container.

    Count-based flushing alone bounds lag by event *volume*, which on a quiet
    system is no bound at all — a written heartbeat could sit in the writer's
    buffer indefinitely and the dashboard would show a frozen system. Found by
    testing across two log instances rather than reading back through the
    writer's own handle.
    """
    writer = JsonlEventLog(tmp_path, flush_every=1_000_000,
                           flush_interval_seconds=1_000_000)
    writer.append(heartbeat("engine").with_seq(1))
    assert list(JsonlEventLog(tmp_path).read()) == [], \
        "precondition: with no flush policy the event is invisible to a reader"

    # With a time bound, the next append carries the flush.
    prompt = JsonlEventLog(tmp_path, flush_every=1_000_000,
                           flush_interval_seconds=0.0)
    prompt.append(heartbeat("engine").with_seq(2))
    seqs = sorted(e.seq for e in JsonlEventLog(tmp_path).read())
    assert 2 in seqs, "a time-bounded flush must make the write visible"
    writer.close()
    prompt.close()


def test_critical_events_are_visible_to_another_reader_immediately(tmp_path):
    """A fill must never be sitting in a buffer when the dashboard asks."""
    writer = JsonlEventLog(tmp_path, flush_every=1_000_000,
                           flush_interval_seconds=1_000_000)
    writer.append(order_filled("c1", "BTC/USD", "buy", Decimal(1), Decimal(10),
                               Decimal(0), is_maker=False).with_seq(1))
    seen = list(JsonlEventLog(tmp_path).read())
    assert [e.seq for e in seen] == [1]
    writer.close()


def test_truncated_final_line_is_skipped_not_fatal(tmp_path):
    """`docker kill` mid-write is normal operation, not an exception. A log that
    refuses to be read after a crash is not a log."""
    log = JsonlEventLog(tmp_path)
    for _ in range(3):
        log.append(heartbeat("engine").with_seq(_ + 1))
    log.close()

    path = sorted(tmp_path.glob("events-*.jsonl"))[0]
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write('{"seq": 4, "ts": "2026-07-27T12:00:00+00:00", "ty')  # torn

    reopened = JsonlEventLog(tmp_path)
    events = list(reopened.read())
    assert len(events) == 3
    assert reopened.malformed_lines == 1
    # And the sequence high-water mark still recovers from the intact records.
    assert reopened.last_sequence() == 3


def test_unknown_event_type_from_an_older_log_is_skipped(tmp_path):
    log = JsonlEventLog(tmp_path)
    log.append(heartbeat("engine").with_seq(1))
    log.close()
    path = sorted(tmp_path.glob("events-*.jsonl"))[0]
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps({"seq": 2, "ts": TS.isoformat(),
                             "type": "from.the.future", "payload": {}}) + "\n")

    reopened = JsonlEventLog(tmp_path)
    assert len(list(reopened.read())) == 1
    assert reopened.malformed_lines == 1


def test_critical_events_are_fsynced_immediately(tmp_path):
    """A fill must survive a power cut; a heartbeat need not."""
    log = JsonlEventLog(tmp_path, flush_every=1_000_000)

    log.append(heartbeat("engine").with_seq(1))
    path = sorted(tmp_path.glob("events-*.jsonl"))[0]
    assert path.stat().st_size == 0, "a heartbeat should still be buffered"

    log.append(order_filled("c", "BTC/USD", "buy", Decimal(1), Decimal(10),
                            Decimal(0), is_maker=False).with_seq(2))
    assert path.stat().st_size > 0, "a fill must be on disk immediately"
    log.close()


def test_log_rotates_by_date_and_reads_in_order(tmp_path):
    log = JsonlEventLog(tmp_path)
    for i, day in enumerate([TS, TS + timedelta(days=1), TS + timedelta(days=2)]):
        log.append(Event(EventType.HEARTBEAT, {"i": i}, ts=day).with_seq(i + 1))
    log.close()

    assert len(sorted(tmp_path.glob("events-*.jsonl"))) == 3
    seqs = [e.seq for e in JsonlEventLog(tmp_path).read()]
    assert seqs == [1, 2, 3]


def test_in_memory_log_bound_drops_oldest():
    log = InMemoryEventLog(max_events=10)
    for i in range(25):
        log.append(heartbeat("x").with_seq(i + 1))
    assert len(log) == 10
    assert log.dropped == 15
    assert next(e.seq for e in log.read()) == 16


# ══════════════════════════════════════════════════════════════════════════
# Backpressure: bounded, drop-oldest, and the drop is visible
# ══════════════════════════════════════════════════════════════════════════

def test_slow_stream_drops_oldest_and_reports_lag():
    bus = _bus()
    stream = bus.stream(name="slow", maxsize=5)

    for _ in range(20):
        bus.publish(heartbeat("engine"))

    assert stream.stats.dropped == 15
    assert len(stream.buffer) == 5
    lagged = [e for e in bus.log.read() if e.type is EventType.SUBSCRIBER_LAGGED]
    assert lagged, "a silent drop is worse than a drop you can see"
    assert lagged[0].payload["subscriber"] == "slow"


def test_lag_events_do_not_cascade():
    """A lagging subscriber must not generate lag events about the lag events."""
    bus = _bus()
    bus.stream(name="tiny", maxsize=1)
    for _ in range(50):
        bus.publish(heartbeat("engine"))
    lagged = [e for e in bus.log.read() if e.type is EventType.SUBSCRIBER_LAGGED]
    # Far fewer than 50, and crucially finite — no runaway.
    assert 0 < len(lagged) < 50


def test_log_keeps_everything_a_dropping_stream_lost():
    """The log is the authority; queues are a convenience."""
    bus = _bus()
    stream = bus.stream(name="s", maxsize=3)
    for _ in range(30):
        bus.publish(heartbeat("engine"))
    hb = [e for e in bus.log.read() if e.type is EventType.HEARTBEAT]
    assert len(hb) == 30
    assert stream.stats.dropped == 27


def test_async_stream_delivers_in_order():
    async def run():
        bus = _bus()
        stream = bus.stream(name="s", maxsize=100)
        for i in range(5):
            bus.publish(Event(EventType.HEARTBEAT, {"i": i}))
        got = []
        async for e in stream:
            got.append(e.payload["i"])
            if len(got) == 5:
                bus.close_stream(stream)
                break
        return got

    assert asyncio.run(run()) == [0, 1, 2, 3, 4]


def test_type_filtered_subscription_sees_only_its_types():
    bus = _bus()
    seen: list[EventType] = []
    bus.subscribe(lambda e: seen.append(e.type), name="fills",
                  types=frozenset({EventType.ORDER_FILLED}))
    bus.publish(heartbeat("engine"))
    bus.publish(order_filled("c", "BTC/USD", "buy", Decimal(1), Decimal(10),
                             Decimal(0), is_maker=True))
    assert seen == [EventType.ORDER_FILLED]


# ══════════════════════════════════════════════════════════════════════════
# Replay
# ══════════════════════════════════════════════════════════════════════════

def _one_trade(bus: EventBus, symbol="BTC/USD", maker=True, fee=Decimal("15")):
    cor = new_correlation_id()
    sig = bus.publish(signal_generated("ema_cross", symbol, 1, 0.8, cor,
                                       ref_price=60_000.0))
    sub = bus.publish(order_submitted("cid-1", symbol, "buy", Decimal("0.1"),
                                      "limit", Decimal(60_000), cor,
                                      causation_id=sig.seq))
    bus.publish(order_filled("cid-1", symbol, "buy", Decimal("0.1"),
                             Decimal(60_060), fee, is_maker=maker,
                             correlation_id=cor, causation_id=sub.seq))
    bus.publish(position_closed(symbol, "ema_cross", Decimal("120"), 0.02,
                                "target", correlation_id=cor))
    return cor


def test_lifecycle_groups_a_whole_trade():
    bus = _bus()
    cor = _one_trade(bus)
    bus.publish(heartbeat("engine"))            # noise, must not appear
    lc = lifecycle(bus.log, cor)
    assert [e.type for e in lc.events] == [
        EventType.SIGNAL_GENERATED, EventType.ORDER_SUBMITTED,
        EventType.ORDER_FILLED, EventType.POSITION_CLOSED,
    ]
    assert lc.outcome == "closed"
    assert lc.filled_qty == pytest.approx(0.1)
    assert lc.maker_ratio == pytest.approx(1.0)


def test_rejected_lifecycle_reports_its_reason():
    bus = _bus()
    cor = new_correlation_id()
    sig = bus.publish(signal_generated("bb", "ETH/USD", 1, 0.5, cor))
    bus.publish(signal_rejected("bb", "ETH/USD", "EDGE_BELOW_FEES",
                                "30bps < 90bps", cor, causation_id=sig.seq))
    lc = lifecycle(bus.log, cor)
    assert lc.outcome == "rejected:EDGE_BELOW_FEES"
    assert lc.fills == []


def test_slippage_is_signed_so_positive_always_means_worse():
    bus = _bus()
    cor = _one_trade(bus)                       # bought 60,000 → filled 60,060
    lc = lifecycle(bus.log, cor)
    assert lc.slippage_bps == pytest.approx(10.0, abs=0.01)


def test_causal_chain_walks_back_to_the_root():
    bus = _bus()
    _one_trade(bus)
    fill = next(e for e in bus.log.read() if e.type is EventType.ORDER_FILLED)
    chain = [e.type for e in causal_chain(bus.log, fill.seq)]
    assert chain == [EventType.SIGNAL_GENERATED, EventType.ORDER_SUBMITTED,
                     EventType.ORDER_FILLED]


def test_rebuild_positions_from_fills_alone():
    """An independent reconstruction — shares no code with the live position
    bookkeeping, which is what makes agreement meaningful."""
    bus = _bus()
    bus.publish(order_filled("a", "BTC/USD", "buy", Decimal("0.5"), Decimal(60_000),
                             Decimal(0), is_maker=True))
    bus.publish(order_filled("b", "BTC/USD", "buy", Decimal("0.25"), Decimal(61_000),
                             Decimal(0), is_maker=True))
    bus.publish(order_filled("c", "BTC/USD", "sell", Decimal("0.35"), Decimal(62_000),
                             Decimal(0), is_maker=False))
    bus.publish(order_filled("d", "ETH/USD", "buy", Decimal(2), Decimal(3_000),
                             Decimal(0), is_maker=True))
    assert rebuild_positions(bus.log) == {"BTC/USD": Decimal("0.40"),
                                          "ETH/USD": Decimal(2)}


def test_fully_closed_symbol_disappears_from_the_rebuild():
    bus = _bus()
    bus.publish(order_filled("a", "BTC/USD", "buy", Decimal(1), Decimal(60_000),
                             Decimal(0), is_maker=True))
    bus.publish(order_filled("b", "BTC/USD", "sell", Decimal(1), Decimal(61_000),
                             Decimal(0), is_maker=False))
    assert rebuild_positions(bus.log) == {}


def test_execution_quality_surfaces_the_maker_ratio():
    """The headline number for whether the maker-first thesis is being realised."""
    bus = _bus()
    _one_trade(bus, symbol="BTC/USD", maker=True)
    _one_trade(bus, symbol="ETH/USD", maker=False)
    q = execution_quality(lifecycles(bus.log))
    assert q["lifecycles"] == 2
    assert q["maker_ratio"] == pytest.approx(0.5)
    assert q["total_fees"] == pytest.approx(30.0)


def test_lifecycles_excludes_uncorrelated_noise():
    bus = _bus()
    _one_trade(bus)
    for _ in range(5):
        bus.publish(heartbeat("engine"))
    assert len(lifecycles(bus.log)) == 1


def test_timeline_and_summarize():
    bus = _bus()
    _one_trade(bus)
    bus.publish(heartbeat("engine"))
    tl = timeline(bus.log)
    assert [e.seq for e in tl] == sorted(e.seq for e in tl)
    counts = summarize(bus.log)
    assert counts["execution.order_filled"] == 1
    assert counts["ops.heartbeat"] == 1


def test_lifecycle_timeline_is_relative_to_its_first_event():
    bus = _bus()
    cor = _one_trade(bus)
    rows = lifecycle(bus.log, cor).timeline()
    assert rows[0][0] == 0.0
    assert all(t >= 0 for t, _, _ in rows)
    assert "ema_cross" in rows[0][2]


# ══════════════════════════════════════════════════════════════════════════
# NullEventBus — same type, no persistence
# ══════════════════════════════════════════════════════════════════════════

def test_null_bus_sequences_and_fans_out_but_stores_nothing():
    """Same type as the real bus on purpose: `if bus is not None` scattered
    through the engine is how backtest and live drift apart again."""
    bus = NullEventBus(source="backtest")
    seen: list[Event] = []
    bus.subscribe(seen.append, name="spy")
    for _ in range(1000):
        bus.publish(heartbeat("backtest"))
    assert bus.sequence == 1000
    assert len(seen) == 1000
    assert len(list(bus.log.read())) == 0     # nothing retained
    assert bus.stats.published == 1000


def test_severity_critical_forces_critical_handling():
    e = Event(EventType.HEARTBEAT, {}, severity=Severity.CRITICAL)
    assert e.is_critical
    assert not heartbeat("x").is_critical
