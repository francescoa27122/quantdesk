"""In-process event bus.

## The one contract

**`publish()` never raises and never blocks.** Everything else here follows from
that. The order path calls `publish()` after each state transition, so a bus
that can throw, stall, or apply backpressure is a bus that can break trading —
which would make the whole observability layer a net liability.

Concretely, all of these are swallowed, counted and reported rather than
propagated:

  * a subscriber that raises
  * a subscriber that is slower than the publisher
  * the log being full, locked, or on a dead disk
  * an asyncio loop that has already closed

The corollary is that **the log, not a subscriber queue, is the authority.**
Queues are bounded and drop under pressure by design; if you need every event,
read the log. `SUBSCRIBER_LAGGED` tells you when a queue dropped, so a gap is
always visible rather than silent.

## Two delivery modes

  * `subscribe(handler)` — synchronous callback, invoked inline. Fast handlers
    only: metrics counters, in-memory projections. A handler that raises three
    times is quarantined and stops being called, because a broken subscriber
    must degrade to *no* observability, never to a broken engine.
  * `stream()` — async iterator over a bounded, drop-oldest buffer. For the
    dashboard SSE feed and anything that might be slow.

## What this bus deliberately is NOT

It is not a message broker, has no delivery guarantee, no acknowledgements, no
retries, and no way to reply to an event. That is the point: none of the
machinery that would let it creep into the order path exists. See
PHASE2-AUDIT.md §7.1.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from ct.events.log import EventLog, InMemoryEventLog
from ct.events.types import Event, EventType, Severity

log = logging.getLogger(__name__)

Handler = Callable[[Event], None]

# Bus meta-events. Emitting one of these must never be able to trigger another,
# or a single lagging subscriber becomes an infinite cascade.
_META = frozenset({
    EventType.SUBSCRIBER_LAGGED,
    EventType.SUBSCRIBER_FAILED,
    EventType.SUBSCRIBER_QUARANTINED,
})


def _meta_subject(event: Event) -> str | None:
    """Which subscriber a bus meta-event is *about*, if any.

    A meta-event is never delivered to its own subject, and that rule is what
    keeps the bus stable rather than merely non-recursive:

      * Pushing "your buffer overflowed" into the buffer that just overflowed
        evicts a real event to report an eviction. It doubled every drop count
        and told you nothing.
      * Delivering "your handler raised" to the handler that just raised makes
        it raise again inside the failure path, so failures accrue at 2× and
        the quarantine notice gets swallowed by the re-entrancy guard.

    The `_in_meta` flag remains as a backstop, but with this rule it should
    never actually fire.
    """
    if event.type not in _META:
        return None
    subject = event.payload.get("subscriber")
    return str(subject) if subject is not None else None


@dataclass
class SubscriberStats:
    name: str
    delivered: int = 0
    dropped: int = 0
    failures: int = 0
    quarantined: bool = False
    total_seconds: float = 0.0
    slowest_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.delivered if self.delivered else 0.0


@dataclass
class BusStats:
    published: int = 0
    log_failures: int = 0
    subscribers: dict[str, SubscriberStats] = field(default_factory=dict)

    def total_dropped(self) -> int:
        return sum(s.dropped for s in self.subscribers.values())

    def quarantined(self) -> list[str]:
        return [n for n, s in self.subscribers.items() if s.quarantined]


@dataclass
class _Subscription:
    name: str
    handler: Handler
    types: frozenset[EventType] | None
    stats: SubscriberStats


class _Stream:
    """Bounded drop-oldest buffer feeding an async iterator."""

    def __init__(self, name: str, types: frozenset[EventType] | None, maxsize: int):
        self.name = name
        self.types = types
        self.buffer: deque[Event] = deque(maxlen=maxsize)
        self.stats = SubscriberStats(name)
        self.closed = False
        self._wakeup = asyncio.Event()
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def offer(self, event: Event) -> bool:
        """Returns False if this push evicted an older event."""
        evicted = len(self.buffer) == self.buffer.maxlen
        self.buffer.append(event)          # deque(maxlen) drops the oldest itself
        self._signal()
        if evicted:
            self.stats.dropped += 1
            return False
        self.stats.delivered += 1
        return True

    def _signal(self) -> None:
        if self._loop is None or self._loop.is_closed():
            self._wakeup.set()
            return
        # Loop shut down between the check and the call. Nothing to wake.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._wakeup.set)

    async def __aiter__(self) -> AsyncIterator[Event]:
        while not self.closed:
            while self.buffer:
                yield self.buffer.popleft()
            if self.closed:
                return
            self._wakeup.clear()
            await self._wakeup.wait()

    def close(self) -> None:
        self.closed = True
        self._signal()


class EventBus:
    """The only sequencer. Assigning `seq` here is what makes the log ordered."""

    def __init__(
        self,
        event_log: EventLog | None = None,
        *,
        source: str = "engine",
        max_handler_failures: int = 3,
        slow_handler_seconds: float = 0.05,
    ) -> None:
        self.log: EventLog = event_log if event_log is not None else InMemoryEventLog()
        self.source = source
        self.max_handler_failures = max_handler_failures
        self.slow_handler_seconds = slow_handler_seconds

        self._seq_lock = threading.Lock()
        # Continue the total order across restarts rather than restarting at 1
        # and producing two events with the same sequence.
        self._seq = self.log.last_sequence()
        self._subs: list[_Subscription] = []
        self._streams: list[_Stream] = []
        self._subs_lock = threading.Lock()
        self._in_meta = threading.local()
        self.stats = BusStats()

    # ── publishing ───────────────────────────────────────────────────────

    def publish(self, event: Event) -> Event:
        """Assign a sequence, persist, fan out. Never raises.

        Returns the sequenced event so callers can use `.seq` as a
        `causation_id` for whatever they emit next.
        """
        with self._seq_lock:
            self._seq += 1
            sequenced = event.with_seq(self._seq)
            if not sequenced.source or sequenced.source == "unknown":
                sequenced = Event(
                    type=sequenced.type, payload=sequenced.payload, ts=sequenced.ts,
                    seq=sequenced.seq, source=self.source, severity=sequenced.severity,
                    correlation_id=sequenced.correlation_id,
                    causation_id=sequenced.causation_id)

        try:
            self.log.append(sequenced)
        except Exception:
            # A dead disk must not stop the engine. Count it; the health check
            # surfaces it. Losing observability is survivable; losing the
            # ability to place a stop is not.
            self.stats.log_failures += 1
            log.exception("event log append failed (seq=%s)", sequenced.seq)

        self.stats.published += 1
        self._fan_out(sequenced)
        return sequenced

    def _fan_out(self, event: Event) -> None:
        with self._subs_lock:
            subs = list(self._subs)
            streams = list(self._streams)

        subject = _meta_subject(event)

        for sub in subs:
            if sub.stats.quarantined:
                continue
            if sub.name == subject:
                continue          # never tell a subscriber about itself
            if sub.types is not None and event.type not in sub.types:
                continue
            started = time.perf_counter()
            try:
                sub.handler(event)
            except Exception as exc:
                sub.stats.failures += 1
                log.exception("subscriber %s raised on %s", sub.name, event.type.value)
                self._emit_meta(Event(
                    EventType.SUBSCRIBER_FAILED,
                    {"subscriber": sub.name, "event_type": event.type.value,
                     "error": repr(exc), "failures": sub.stats.failures},
                    severity=Severity.WARNING, source="bus"))
                if sub.stats.failures >= self.max_handler_failures:
                    sub.stats.quarantined = True
                    self._emit_meta(Event(
                        EventType.SUBSCRIBER_QUARANTINED,
                        {"subscriber": sub.name, "failures": sub.stats.failures},
                        severity=Severity.WARNING, source="bus"))
                continue
            elapsed = time.perf_counter() - started
            sub.stats.delivered += 1
            sub.stats.total_seconds += elapsed
            sub.stats.slowest_seconds = max(sub.stats.slowest_seconds, elapsed)

        for stream in streams:
            if stream.closed:
                continue
            if stream.name == subject:
                continue          # do not evict a real event to report an eviction
            if stream.types is not None and event.type not in stream.types:
                continue
            if not stream.offer(event) and event.type not in _META:
                self._emit_meta(Event(
                    EventType.SUBSCRIBER_LAGGED,
                    {"subscriber": stream.name, "dropped": stream.stats.dropped,
                     "buffer_size": stream.buffer.maxlen},
                    severity=Severity.WARNING, source="bus"))

    def _emit_meta(self, event: Event) -> None:
        """Publish a bus meta-event without letting it trigger another one."""
        if getattr(self._in_meta, "active", False):
            return
        self._in_meta.active = True
        try:
            self.publish(event)
        finally:
            self._in_meta.active = False

    # ── subscribing ──────────────────────────────────────────────────────

    def subscribe(
        self,
        handler: Handler,
        *,
        name: str | None = None,
        types: frozenset[EventType] | None = None,
    ) -> str:
        """Register a synchronous callback. Returns its name.

        Handlers run inline on the publisher's thread, so they must be fast and
        must not do I/O. One that raises `max_handler_failures` times is
        quarantined permanently — degrade to no observability, never to a
        broken engine.
        """
        sub_name = name or getattr(handler, "__name__", f"sub-{len(self._subs)}")
        with self._subs_lock:
            stats = SubscriberStats(sub_name)
            self._subs.append(_Subscription(sub_name, handler, types, stats))
            self.stats.subscribers[sub_name] = stats
        return sub_name

    def unsubscribe(self, name: str) -> bool:
        with self._subs_lock:
            before = len(self._subs)
            self._subs = [s for s in self._subs if s.name != name]
            return len(self._subs) != before

    def stream(
        self,
        *,
        name: str | None = None,
        types: frozenset[EventType] | None = None,
        maxsize: int = 1000,
    ) -> _Stream:
        """A bounded, drop-oldest async feed. Drop-oldest because a dashboard
        wants *current* state; the log remains the authority for completeness."""
        stream_name = name or f"stream-{len(self._streams)}"
        s = _Stream(stream_name, types, maxsize)
        with self._subs_lock:
            self._streams.append(s)
            self.stats.subscribers[stream_name] = s.stats
        return s

    def close_stream(self, stream: _Stream) -> None:
        stream.close()
        with self._subs_lock:
            self._streams = [s for s in self._streams if s is not stream]

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def sequence(self) -> int:
        return self._seq

    def flush(self) -> None:
        try:
            self.log.flush()
        except Exception:
            self.stats.log_failures += 1
            log.exception("event log flush failed")

    def close(self) -> None:
        with self._subs_lock:
            streams = list(self._streams)
        for s in streams:
            s.close()
        try:
            self.log.close()
        except Exception:
            self.stats.log_failures += 1
            log.exception("event log close failed")


class NullEventBus(EventBus):
    """Sequences and fans out but never persists.

    For the backtester, which emits millions of events per run. Keeping the
    same type means strategy and risk code has exactly one code path whether or
    not anything is listening — the alternative, `if bus is not None` scattered
    through the engine, is how the backtest and live paths drift apart again.
    """

    def __init__(self, **kw) -> None:
        super().__init__(InMemoryEventLog(max_events=0), **kw)

    def publish(self, event: Event) -> Event:
        with self._seq_lock:
            self._seq += 1
            sequenced = event.with_seq(self._seq)
        self.stats.published += 1
        self._fan_out(sequenced)
        return sequenced


__all__ = ["BusStats", "EventBus", "Handler", "NullEventBus", "SubscriberStats"]
