"""`qd-collector` — the market data service.

Wires together the four pieces that each do one job:

    feed.py       websocket, reconnect, backoff        -> raw bars
    integrity.py  gap/stale/plausibility detection     -> tradeable or not
    backfill.py   REST catch-up over the retained window
    repository.py idempotent persistence

The sequence that matters is the **reconnect handshake**. A dropped websocket
always loses messages, so on every reconnect the collector REST-backfills from
the last stored bar before resuming the stream. Without it each disconnect
leaves a permanent notch that no later run will ever fill.

Isolation note: this container holds **no private credentials**. It reads public
market data and writes candles. A compromise here cannot trade.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ct.data.backfill import Backfiller
from ct.data.feed import FeedConfig, MarketDataFeed, Watcher
from ct.data.integrity import DataIntegrityMonitor, IntegrityConfig, IntegrityState
from ct.data.repository import CandleRepository
from ct.events import (
    EventBus,
    NullEventBus,
    bar_closed,
    data_gap_detected,
    exchange_disconnected,
    heartbeat,
)
from ct.events.types import Event, EventType, Severity
from ct.models import Bar

log = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    timeframes: tuple[str, ...] = ("1h", "4h", "1d")
    heartbeat_seconds: float = 30.0
    staleness_check_seconds: float = 30.0
    # Cap on how far a catch-up will backfill. None = per-timeframe: the venue
    # retains ~720 BARS per timeframe (A11), so the honest cap is bar-count-
    # shaped, not wall-clock-shaped. The old default of a flat 7 DAYS silently
    # starved every 4h strategy: 7 days is 42 bars against donchian's 205-bar
    # warmup, so a freshly initialised database could not feed the engine for
    # a month and the strategy was never consulted at all (found 2026-07-30,
    # by the operator asking why the Positions page was empty).
    max_catchup: timedelta | None = None


@dataclass
class CollectorStats:
    bars_stored: int = 0
    bars_rejected: int = 0
    # Bars the stream re-delivered that storage already had (websocket
    # snapshots replay history on every (re)connect). Not faults — see
    # `_handle_bar` — but counted so a flood of them stays visible.
    bars_already_known: int = 0
    backfills: int = 0
    backfilled_bars: int = 0
    reconnects: int = 0


class MarketDataCollector:
    """Owns one feed per (symbol, timeframe) plus the shared monitor."""

    def __init__(
        self,
        watcher: Watcher,
        backfiller: Backfiller,
        repository: CandleRepository | None,
        *,
        config: CollectorConfig | None = None,
        integrity: IntegrityConfig | None = None,
        feed_config: FeedConfig | None = None,
        bus: EventBus | None = None,
        clock=None,
    ) -> None:
        self.config = config or CollectorConfig()
        self.watcher = watcher
        self.backfiller = backfiller
        self.repo = repository
        self.monitor = DataIntegrityMonitor(integrity)
        self.feed_config = feed_config or FeedConfig()
        self.bus = bus if bus is not None else NullEventBus(source="collector")
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self.stats = CollectorStats()
        self.feeds: list[MarketDataFeed] = []
        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()

        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                self.monitor.track(symbol, tf)

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        self.bus.publish(Event(EventType.ENGINE_STARTED,
                               {"component": "collector",
                                "symbols": list(self.config.symbols),
                                "timeframes": list(self.config.timeframes)},
                               source="collector"))
        # Catch up BEFORE streaming. Starting the stream first would leave the
        # gap between shutdown and restart unfilled — and it is the one gap
        # guaranteed to exist on every restart.
        await self.catch_up_all()

        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                feed = MarketDataFeed(
                    self.watcher, symbol, tf,
                    on_bar=self._handle_bar,
                    on_reconnect=self._handle_reconnect,
                    config=self.feed_config, clock=self._now)
                self.feeds.append(feed)
                self._tasks.append(asyncio.create_task(
                    feed.run(), name=f"feed-{symbol}-{tf}"))
        self._tasks.append(asyncio.create_task(self._monitor_loop(),
                                               name="collector-monitor"))

    async def stop(self) -> None:
        self._stopping.set()
        for feed in self.feeds:
            feed.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self.bus.publish(Event(EventType.ENGINE_STOPPING,
                               {"component": "collector"}, source="collector"))

    async def run_forever(self) -> None:
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self._stopping.set)
        await self._stopping.wait()
        await self.stop()

    # ── the hot path ─────────────────────────────────────────────────────

    async def _handle_bar(self, bar: Bar) -> None:
        now = self._now()
        health = self.monitor.health[(bar.symbol, bar.timeframe)]
        # Websocket snapshots replay recent history on every (re)connect, and
        # boot backfills have usually stored those bars already — the monitor's
        # cursor is past them. A bar we already hold is not a fault and needs
        # no recovery; feeding it to the monitor flagged OUT_OF_ORDER on every
        # restart, and on 1d the 3-clean-bar settling period kept the symbol
        # reading "degraded" for up to three days. Crying wolf on every
        # restart is how a real degradation gets ignored.
        if health.last_bar_ts is not None and bar.ts <= health.last_bar_ts:
            self.stats.bars_already_known += 1
            return
        before = health.gaps_detected
        if not self.monitor.accept(bar, now=now):
            self.stats.bars_rejected += 1
            return

        health = self.monitor.health[(bar.symbol, bar.timeframe)]
        if health.gaps_detected > before:
            # A gap the stream itself revealed — fill it now rather than
            # leaving a notch nothing will ever revisit.
            self.bus.publish(data_gap_detected(
                bar.symbol, bar.timeframe, health.missing_bars,
                health.last_bar_ts or bar.ts, bar.ts))
            await self._backfill(bar.symbol, bar.timeframe)

        if self.repo is not None:
            await self.repo.upsert_many([bar])
        self.stats.bars_stored += 1
        self.bus.publish(bar_closed(bar.symbol, bar.timeframe, bar.close, bar.ts,
                                    volume=bar.volume))

    async def _handle_reconnect(self, symbol: str, timeframe: str,
                                last_seen: datetime | None) -> None:
        """A disconnect ALWAYS loses messages — backfill, never just resume."""
        self.stats.reconnects += 1
        self.bus.publish(exchange_disconnected(
            "kraken", f"{symbol} {timeframe} stream dropped; backfilling"))
        await self._backfill(symbol, timeframe)

    async def _backfill(self, symbol: str, timeframe: str) -> int:
        last = None
        if self.repo is not None:
            last = await self.repo.latest_ts(symbol, timeframe)
        if last is None:
            h = self.monitor.health.get((symbol, timeframe))
            last = h.last_bar_ts if h else None

        result = await self.backfiller.catch_up(
            symbol, timeframe, last_known=last, now=self._now(),
            max_lookback=self.config.max_catchup)
        self.stats.backfills += 1
        if not result.bars:
            return 0
        if self.repo is not None:
            await self.repo.upsert_many(result.bars, is_gap_fill=True)
        self.stats.backfilled_bars += len(result.bars)
        # Tell the monitor the hole is filled. Without this its cursor stays at
        # the pre-disconnect bar, the next live bar measures a gap that no
        # longer exists, and the feed degrades on its own successful recovery.
        self.monitor.mark_backfilled(symbol, timeframe, result.bars[-1].ts,
                                     now=self._now())
        log.info("backfilled %d bars for %s %s", len(result.bars), symbol, timeframe)
        return len(result.bars)

    async def catch_up_all(self) -> int:
        total = 0
        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                total += await self._backfill(symbol, tf)
        return total

    # ── the timer that catches silence ───────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Staleness and heartbeat. Driven by the clock, because a feed that
        has stopped delivering cannot report its own absence."""
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(),
                                       self.config.staleness_check_seconds)
                return
            except TimeoutError:
                pass
            now = self._now()
            for fault in self.monitor.check_staleness(now):
                self.bus.publish(Event(
                    EventType.DATA_INTEGRITY_DEGRADED,
                    {"symbol": fault.symbol, "timeframe": fault.timeframe,
                     "kind": fault.kind.value, "detail": fault.detail},
                    severity=Severity.CRITICAL, source="collector"))
            self.bus.publish(heartbeat(
                "collector",
                bars_stored=self.stats.bars_stored,
                bars_already_known=self.stats.bars_already_known,
                degraded=self.monitor.degraded_symbols(),
                awaiting_first_bar=self.monitor.unknown_symbols(),
                reconnects=self.stats.reconnects))

    # ── what the engine asks ─────────────────────────────────────────────

    def tradeable_symbols(self) -> list[str]:
        """Symbols healthy on EVERY tracked timeframe.

        Deliberately strict: a strategy reading 4h while the 1h regime feed is
        stale is trading on half a picture, and the conservative reading is the
        only safe one when the question is whether to risk money.
        """
        by_symbol: dict[str, list[IntegrityState]] = {}
        for (sym, _tf), h in self.monitor.health.items():
            by_symbol.setdefault(sym, []).append(h.state)
        return sorted(s for s, states in by_symbol.items()
                      if all(st is IntegrityState.OK for st in states))

    def health_summary(self) -> dict[str, object]:
        return {
            "bars_stored": self.stats.bars_stored,
            "bars_rejected": self.stats.bars_rejected,
            "bars_already_known": self.stats.bars_already_known,
            "backfills": self.stats.backfills,
            "backfilled_bars": self.stats.backfilled_bars,
            "reconnects": self.stats.reconnects,
            "degraded": self.monitor.degraded_symbols(),
            "awaiting_first_bar": self.monitor.unknown_symbols(),
            "tradeable": self.tradeable_symbols(),
            "feeds_given_up": [f"{f.symbol} {f.timeframe}" for f in self.feeds
                               if f.stats.given_up],
        }


async def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for `python -m ct.data.collector` (see docker-compose.yml)."""
    import os
    import sys

    from ct.logging import configure as _configure_logging

    _configure_logging()
    argv = list(argv if argv is not None else sys.argv[1:])

    if "--healthcheck" in argv:
        # The compose healthcheck. Deliberately only proves the process can
        # import and reach the database — claiming more would make the check
        # itself a source of false confidence.
        dsn = os.environ.get("QD_DB_DSN")
        if not dsn:
            print("QD_DB_DSN not set")
            return 1
        from ct.store import Database
        db = Database(dsn, max_size=1)
        try:
            await db.connect(retries=1)
            ok = await db.healthy()
        finally:
            await db.close()
        print("ok" if ok else "database unreachable")
        return 0 if ok else 1

    import ccxt.pro as ccxtpro

    from ct.data.backfill import ccxt_fetcher
    from ct.data.feed import ccxt_watcher
    from ct.events import JsonlEventLog
    from ct.store import Database, PostgresEventSink, TeeEventLog

    dsn = os.environ["QD_DB_DSN"]
    symbols = tuple(s.strip() for s in
                    os.environ.get("QD_SYMBOLS", "BTC/USD,ETH/USD").split(",") if s.strip())
    timeframes = tuple(s.strip() for s in
                       os.environ.get("QD_TIMEFRAMES", "1h,4h,1d").split(",") if s.strip())

    db = Database(dsn)
    await db.connect()
    sink = PostgresEventSink(db)
    await sink.start()
    bus = EventBus(TeeEventLog(JsonlEventLog("/app/logs/events"), sink),
                   source="collector")

    exchange = ccxtpro.kraken({"enableRateLimit": True})
    collector = MarketDataCollector(
        ccxt_watcher(exchange), Backfiller(ccxt_fetcher(exchange)),
        CandleRepository(db, exchange=os.environ.get("QD_EXCHANGE", "kraken")),
        config=CollectorConfig(symbols=symbols, timeframes=timeframes), bus=bus)
    try:
        await collector.run_forever()
    finally:
        await collector.stop()
        await exchange.close()
        await sink.stop()
        bus.close()
        await db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
