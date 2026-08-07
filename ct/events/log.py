"""Durable, append-only event log.

Two implementations, both complete — neither is a placeholder for the other:

  * `JsonlEventLog`  — one JSON object per line, date-rotated files on disk.
    This is the production log for now. It is deliberately readable with `jq`
    on the NAS with this codebase absent, which matters at 3 a.m.
  * `InMemoryEventLog` — same contract, no disk. For tests and for the
    backtester, which emits millions of events nobody wants on a spinning disk.

A Postgres-backed log lands with the persistence layer (Phase 2 item 2) as a
third implementation of the same protocol; the `events` hypertable is already
in `db/schema.sql`. Nothing above this module knows which one it has.

Three properties this file exists to guarantee:

1. **Total order.** The bus is the only sequencer; the log preserves what it
   assigns and can recover the high-water mark after a restart.
2. **Crash tolerance.** A process killed mid-write leaves a truncated final
   line. Reading MUST skip it and count it, never raise — a corrupted tail is
   the normal consequence of `docker kill`, not an exceptional condition.
3. **Tiered durability.** Money-relevant events are fsync'd immediately;
   heartbeats are not. fsyncing every heartbeat on a 4-core NAS is pure cost
   for no recoverable information.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

from ct.events.types import Event, EventType

log = logging.getLogger(__name__)


@runtime_checkable
class EventLog(Protocol):
    """What the bus needs from a log. Deliberately tiny."""

    def append(self, event: Event) -> None: ...

    def read(
        self,
        *,
        since_seq: int = 0,
        types: frozenset[EventType] | None = None,
        correlation_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterator[Event]: ...

    def last_sequence(self) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class InMemoryEventLog:
    """Complete implementation, no disk. Optionally bounded so a long backtest
    cannot exhaust RAM on a 7.7 GB NAS."""

    def __init__(self, max_events: int | None = None) -> None:
        self._events: list[Event] = []
        self._max = max_events
        self._lock = threading.Lock()
        self.dropped = 0

    def append(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)
            if self._max is not None and len(self._events) > self._max:
                # Drop oldest: the recent past is what an operator asks about.
                overflow = len(self._events) - self._max
                del self._events[:overflow]
                self.dropped += overflow

    def read(self, **kw: Any) -> Iterator[Event]:
        with self._lock:
            snapshot = list(self._events)
        yield from _filter(snapshot, **kw)

    def last_sequence(self) -> int:
        with self._lock:
            return max((e.seq for e in self._events), default=0)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._events)


class JsonlEventLog:
    """Date-rotated newline-delimited JSON on disk.

    Files are `events-YYYY-MM-DD.jsonl` in `directory`. Date rotation rather
    than size rotation because operators reason in days ("what happened
    Tuesday?"), retention is a `find -mtime` away, and it needs no rename dance
    that could lose a file mid-crash.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        fsync_critical: bool = True,
        flush_every: int = 32,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fsync_critical = fsync_critical
        self.flush_every = max(1, flush_every)
        # Count-based flushing alone bounds lag by *event volume*, which on a
        # quiet system is no bound at all — a written event can sit in this
        # process's buffer indefinitely, invisible to `qd-web` reading the same
        # file from another container. Time-based flushing bounds it by wall
        # clock instead. The engine's periodic HEARTBEAT guarantees there is
        # always a next append to carry the flush.
        self.flush_interval_seconds = flush_interval_seconds

        self._lock = threading.Lock()
        self._handle: IO[str] | None = None
        self._open_date: date | None = None
        self._since_flush = 0
        self._last_flush = time.monotonic()
        self.malformed_lines = 0

    # ── writing ──────────────────────────────────────────────────────────

    def _file_for(self, when: date) -> Path:
        return self.directory / f"events-{when.isoformat()}.jsonl"

    def _ensure_open(self, when: date) -> None:
        if self._handle is not None and self._open_date == when:
            return
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
        # newline="" keeps Windows from turning "\n" into "\r\n", so a line is
        # exactly one record on every platform the log might be read on.
        #
        # SIM115 is suppressed deliberately: the handle is long-lived. Opening
        # per append would cost a syscall pair on every event and lose the
        # buffering the tiered flush policy depends on. Ownership is explicit —
        # `close()` and this method's rotation branch are the only releases.
        self._handle = open(  # noqa: SIM115
            self._file_for(when), "a", encoding="utf-8", newline="")
        self._open_date = when

    def append(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"),
                          default=str) + "\n"
        with self._lock:
            self._ensure_open(event.ts.date())
            assert self._handle is not None
            self._handle.write(line)
            self._since_flush += 1

            now = time.monotonic()
            if event.is_critical and self.fsync_critical:
                # Money-relevant: make it survive a power cut, now.
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._since_flush = 0
                self._last_flush = now
            elif (self._since_flush >= self.flush_every
                  or now - self._last_flush >= self.flush_interval_seconds):
                self._handle.flush()      # to the OS; survives a process crash
                self._since_flush = 0     # (not a power cut — that's the trade)
                self._last_flush = now    # and makes it visible to other readers

    def flush(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._since_flush = 0
                self._last_flush = time.monotonic()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()
                self._handle = None
                self._open_date = None

    # ── reading ──────────────────────────────────────────────────────────

    def files(self) -> list[Path]:
        """Chronological. Filenames sort lexicographically because ISO dates do."""
        return sorted(self.directory.glob("events-*.jsonl"))

    def read(
        self,
        *,
        since_seq: int = 0,
        types: frozenset[EventType] | None = None,
        correlation_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterator[Event]:
        self.flush()
        emitted = 0
        for path in self.files():
            for raw in self._read_file(path):
                if raw.seq <= since_seq:
                    continue
                if types is not None and raw.type not in types:
                    continue
                if correlation_id is not None and raw.correlation_id != correlation_id:
                    continue
                if start is not None and raw.ts < start:
                    continue
                if end is not None and raw.ts > end:
                    continue
                yield raw
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _read_file(self, path: Path) -> Iterator[Event]:
        with open(path, encoding="utf-8", newline="") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Event.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    # A truncated final line is the ordinary result of a kill,
                    # and an unknown EventType is what reading an older log
                    # looks like. Count both; never raise. A log that refuses
                    # to be read after a crash is not a log.
                    self.malformed_lines += 1
                    log.warning("skipping malformed event %s:%d (%s)",
                                path.name, lineno, exc)

    def last_sequence(self) -> int:
        """Highest sequence on disk, so a restart continues the total order.

        Scans only the newest file: sequences are monotonic across the whole
        log, so the maximum can only live in the last-written file.
        """
        files = self.files()
        if not files:
            return 0
        self.flush()
        best = 0
        for event in self._read_file(files[-1]):
            best = max(best, event.seq)
        if best == 0 and len(files) > 1:
            # Newest file is empty or entirely corrupt — fall back one.
            for event in self._read_file(files[-2]):
                best = max(best, event.seq)
        return best


def _filter(
    events: list[Event],
    *,
    since_seq: int = 0,
    types: frozenset[EventType] | None = None,
    correlation_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> Iterator[Event]:
    emitted = 0
    for e in events:
        if e.seq <= since_seq:
            continue
        if types is not None and e.type not in types:
            continue
        if correlation_id is not None and e.correlation_id != correlation_id:
            continue
        if start is not None and e.ts < start:
            continue
        if end is not None and e.ts > end:
            continue
        yield e
        emitted += 1
        if limit is not None and emitted >= limit:
            return


__all__ = ["EventLog", "InMemoryEventLog", "JsonlEventLog"]
