"""Postgres event sink — the queryable projection of the event log.

## Why this is not simply "the event log in Postgres"

`EventBus.publish()` must never block (see `ct/events/bus.py`). A synchronous
INSERT in the order path would break that outright: a slow or restarting
database would stall the engine between deciding to trade and trading.

So the split is:

  * **`JsonlEventLog` is the durable authority.** Local file, fsync on
    money-relevant events, survives a power cut, readable with `jq` when the
    database is down and you are trying to work out why.
  * **`PostgresEventSink` is the queryable projection.** `append()` puts the
    event on a bounded in-memory queue and returns immediately; a background
    task batches it into the database. If Postgres is unavailable, trading
    continues and nothing is lost — the JSONL file still has everything, and
    `backfill_from_jsonl` replays the gap.

Inserts are `ON CONFLICT (seq) DO NOTHING`, so replay is idempotent and a
backfill can safely overlap with live writes.

`TeeEventLog` composes the two so the bus sees a single `EventLog`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Iterator, Sequence
from typing import Any

from ct.events.log import EventLog, JsonlEventLog
from ct.events.types import Event, EventType
from ct.store.db import Database

log = logging.getLogger(__name__)

_INSERT = """
INSERT INTO events (seq, ts, type, source, severity, correlation_id,
                    causation_id, payload)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
ON CONFLICT (seq) DO NOTHING
"""


class PostgresEventSink:
    """Non-blocking write-behind sink. Implements the `EventLog` protocol."""

    def __init__(self, db: Database, *, queue_size: int = 10_000,
                 batch_size: int = 200, flush_interval: float = 1.0) -> None:
        self.db = db
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        # Bounded and drop-oldest. An unbounded queue in front of a database
        # that is down does not preserve events, it exhausts a 7.7 GB NAS and
        # takes the engine with it. Dropping is visible; OOM is not.
        self._queue: deque[Event] = deque(maxlen=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.written = 0
        self.dropped = 0
        self.write_failures = 0

    # ── EventLog protocol ────────────────────────────────────────────────

    def append(self, event: Event) -> None:
        """Enqueue. Never blocks, never raises, never touches the database."""
        if len(self._queue) == self._queue.maxlen:
            self.dropped += 1
        self._queue.append(event)

    def read(self, **kw: Any) -> Iterator[Event]:
        """Not supported synchronously — use `fetch()`.

        Raising rather than returning empty is deliberate: silently returning
        nothing would make a caller believe the log is empty when it is merely
        the wrong access path.
        """
        raise NotImplementedError(
            "PostgresEventSink is write-behind; use `await sink.fetch(...)` or "
            "read the JSONL log for synchronous access")

    def last_sequence(self) -> int:
        return 0    # the bus recovers its high-water mark from the JSONL log

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    # ── the drain ────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task = asyncio.create_task(self._run(), name="pg-event-sink")

    async def stop(self, *, drain: bool = True) -> None:
        self._stopping.set()
        if self._task is not None:
            await asyncio.wait([self._task], timeout=10)
            self._task = None
        if drain:
            await self.drain_once()

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.drain_once()
            except Exception:
                self.write_failures += 1
                log.exception("event sink drain failed; will retry")
            # Timeout is the normal path: it means the interval elapsed with
            # no shutdown request, so loop round and drain again.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), self.flush_interval)

    async def drain_once(self) -> int:
        """Write up to one batch. Returns rows attempted."""
        if not self._queue or not self.db.is_connected:
            return 0
        batch: list[Event] = []
        while self._queue and len(batch) < self.batch_size:
            batch.append(self._queue.popleft())
        if not batch:
            return 0
        try:
            await self.db.executemany(_INSERT, [_row(e) for e in batch])
            self.written += len(batch)
            return len(batch)
        except Exception:
            # Put them back at the FRONT so ordering survives a transient
            # failure. If the queue is full they fall off the other end, which
            # is the same drop-oldest policy applied consistently.
            self.write_failures += 1
            for e in reversed(batch):
                self._queue.appendleft(e)
            raise

    # ── reading ──────────────────────────────────────────────────────────

    async def fetch(self, *, since_seq: int = 0,
                    types: Sequence[EventType] | None = None,
                    correlation_id: str | None = None,
                    limit: int = 1000) -> list[Event]:
        rows = await self.db.fetch(
            """SELECT seq, ts, type, source, severity, correlation_id,
                      causation_id, payload
               FROM events
               WHERE seq > $1
                 AND ($2::text[] IS NULL OR type = ANY($2))
                 AND ($3::text IS NULL OR correlation_id = $3)
               ORDER BY seq LIMIT $4""",
            since_seq, [t.value for t in types] if types else None,
            correlation_id, limit)
        return [_event(r) for r in rows]

    async def max_sequence(self) -> int:
        v = await self.db.fetchval("SELECT max(seq) FROM events")
        return int(v) if v is not None else 0

    async def stats(self) -> dict[str, int]:
        return {"queued": len(self._queue), "written": self.written,
                "dropped": self.dropped, "write_failures": self.write_failures}


class TeeEventLog:
    """Writes to several logs; one failing never stops the others.

    The bus sees a single `EventLog`. The durability guarantee comes from the
    JSONL member, the queryability from the Postgres member, and neither can
    take the other down.
    """

    def __init__(self, *logs: EventLog) -> None:
        if not logs:
            raise ValueError("TeeEventLog needs at least one log")
        self.logs = logs
        self.failures: dict[int, int] = {i: 0 for i in range(len(logs))}

    def append(self, event: Event) -> None:
        for i, lg in enumerate(self.logs):
            try:
                lg.append(event)
            except Exception:
                self.failures[i] += 1
                log.exception("event log %d append failed", i)

    def read(self, **kw: Any) -> Iterator[Event]:
        """Reads come from the FIRST log that supports it — by construction the
        durable one, so a read never depends on the database being up."""
        for lg in self.logs:
            try:
                return lg.read(**kw)
            except NotImplementedError:
                continue
        return iter(())

    def last_sequence(self) -> int:
        return max((lg.last_sequence() for lg in self.logs), default=0)

    def flush(self) -> None:
        for lg in self.logs:
            try:
                lg.flush()
            except Exception:
                log.exception("event log flush failed")

    def close(self) -> None:
        for lg in self.logs:
            try:
                lg.close()
            except Exception:
                log.exception("event log close failed")


async def backfill_from_jsonl(sink: PostgresEventSink, jsonl: JsonlEventLog,
                              *, batch: int = 500) -> int:
    """Replay events the database is missing, from the durable file log.

    Run after any period when Postgres was down. Idempotent — inserts are
    `ON CONFLICT DO NOTHING` — so it is safe to run at every boot, and cheap
    when there is nothing to do.
    """
    have = await sink.max_sequence()
    written = 0
    pending: list[Event] = []
    for event in jsonl.read(since_seq=have):
        pending.append(event)
        if len(pending) >= batch:
            await sink.db.executemany(_INSERT, [_row(e) for e in pending])
            written += len(pending)
            pending.clear()
    if pending:
        await sink.db.executemany(_INSERT, [_row(e) for e in pending])
        written += len(pending)
    if written:
        log.info("backfilled %d events into Postgres from seq %d", written, have)
    return written


def _row(e: Event) -> tuple[Any, ...]:
    return (e.seq, e.ts, e.type.value, e.source, e.severity.value,
            e.correlation_id, e.causation_id, json.dumps(e.payload, default=str))


def _event(r: Any) -> Event:
    from ct.events.types import Severity
    payload = r["payload"]
    return Event(type=EventType(r["type"]),
                 payload=json.loads(payload) if isinstance(payload, str) else payload,
                 ts=r["ts"], seq=int(r["seq"]), source=r["source"],
                 severity=Severity(r["severity"]),
                 correlation_id=r["correlation_id"],
                 causation_id=r["causation_id"])


__all__ = ["PostgresEventSink", "TeeEventLog", "backfill_from_jsonl"]
