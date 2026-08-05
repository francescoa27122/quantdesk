"""Websocket market-data feed — reconnect, heartbeat, and giving up honestly.

Wraps `ccxt.pro`'s `watch_ohlcv` (verified 2026-07-28: ccxt 4.5.69, MIT, Kraken
supports it). The exchange client is injected, so every behaviour worth testing
— backoff, jitter, heartbeat, the reconnect→backfill handshake — is testable
without a network.

Three things this gets right that a naive `while True: await watch()` does not:

1. **Jittered exponential backoff.** Reconnecting immediately after a drop, on
   a fixed schedule, means every symbol retries in lockstep and hits the venue
   as one synchronised burst — the classic thundering herd, at exactly the
   moment the venue is already unhappy.
2. **A heartbeat that measures silence, not liveness.** A websocket that stops
   delivering looks identical to a market with no trades. Only elapsed time
   distinguishes them, so staleness is driven by a timer, never by arrivals.
3. **Reconnect implies a gap.** Messages are lost while disconnected, always.
   The feed therefore signals reconnection so the collector can REST-backfill
   the hole; a feed that silently resumes leaves a permanent notch in the data.

`stop()` is cooperative and idempotent — `qd-engine` has a 30 s
`stop_grace_period` and a feed that ignores SIGTERM burns all of it.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from ct.data.integrity import timeframe_seconds
from ct.models import Bar

log = logging.getLogger(__name__)

# (symbol, timeframe) -> raw ccxt OHLCV rows. Mirrors `exchange.watch_ohlcv`.
Watcher = Callable[[str, str], Awaitable[Sequence[Sequence[float]]]]
BarHandler = Callable[[Bar], Awaitable[None]]
ReconnectHandler = Callable[[str, str, datetime | None], Awaitable[None]]


@dataclass
class FeedConfig:
    # Backoff ceiling. Beyond ~a minute the venue is down, not busy, and
    # retrying faster helps nobody.
    base_backoff: float = 1.0
    max_backoff: float = 60.0
    # Full jitter. Without it, N symbols reconnect in lockstep.
    jitter: float = 0.5
    # No message at all for this many intervals ⇒ treat the socket as dead and
    # reconnect, even though it has not errored. Silent sockets are the common
    # failure, not noisy ones.
    silence_intervals: float = 3.0
    min_silence_seconds: float = 30.0
    # Consecutive failures before the feed stops retrying and says so. A feed
    # that retries forever looks identical to one that is working.
    max_consecutive_failures: int = 20


@dataclass
class FeedStats:
    connects: int = 0
    reconnects: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    bars_emitted: int = 0
    last_message_at: datetime | None = None
    last_error: str = ""
    given_up: bool = False

    def silence_seconds(self, now: datetime) -> float | None:
        if self.last_message_at is None:
            return None
        return (now - self.last_message_at).total_seconds()


class MarketDataFeed:
    """One (symbol, timeframe) stream with its own reconnect loop."""

    def __init__(
        self,
        watcher: Watcher,
        symbol: str,
        timeframe: str,
        *,
        on_bar: BarHandler,
        on_reconnect: ReconnectHandler | None = None,
        config: FeedConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.watcher = watcher
        self.symbol = symbol
        self.timeframe = timeframe
        self.on_bar = on_bar
        self.on_reconnect = on_reconnect
        self.config = config or FeedConfig()
        # Injected so tests do not sleep and do not depend on wall-clock time.
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleeper or asyncio.sleep
        self.stats = FeedStats()
        self._stopping = asyncio.Event()
        self._last_emitted_ts: datetime | None = None

    @property
    def silence_limit(self) -> float:
        return max(self.config.min_silence_seconds,
                   self.config.silence_intervals
                   * timeframe_seconds(self.timeframe))

    def stop(self) -> None:
        """Cooperative and idempotent — safe to call from a signal handler."""
        self._stopping.set()

    async def run(self) -> None:
        """Watch until stopped or until the failure ceiling is reached."""
        attempt = 0
        while not self._stopping.is_set():
            try:
                rows = await asyncio.wait_for(
                    self.watcher(self.symbol, self.timeframe),
                    timeout=self.silence_limit)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                # Not an error the socket reported — an absence. This is the
                # failure mode a naive loop never notices.
                self.stats.failures += 1
                self.stats.consecutive_failures += 1
                self.stats.last_error = (
                    f"no message for {self.silence_limit:.0f}s")
                log.warning("%s %s: silent socket, reconnecting", self.symbol,
                            self.timeframe)
                attempt = await self._reconnect(attempt)
                continue
            except Exception as exc:
                self.stats.failures += 1
                self.stats.consecutive_failures += 1
                self.stats.last_error = repr(exc)
                log.warning("%s %s: feed error %r", self.symbol, self.timeframe, exc)
                attempt = await self._reconnect(attempt)
                continue

            if self.stats.consecutive_failures:
                self.stats.consecutive_failures = 0
                attempt = 0
            self.stats.last_message_at = self._now()
            if self.stats.connects == 0:
                self.stats.connects = 1

            for bar in self._to_bars(rows):
                self.stats.bars_emitted += 1
                self._last_emitted_ts = bar.ts
                await self.on_bar(bar)

            if self.stats.given_up:
                return

    async def _reconnect(self, attempt: int) -> int:
        if self.stats.consecutive_failures >= self.config.max_consecutive_failures:
            # Stop pretending. A feed that retries forever is indistinguishable
            # from one that is working, and the engine needs to know the
            # difference to halt.
            self.stats.given_up = True
            self._stopping.set()
            log.error("%s %s: giving up after %d consecutive failures (%s)",
                      self.symbol, self.timeframe,
                      self.stats.consecutive_failures, self.stats.last_error)
            return attempt

        delay = min(self.config.base_backoff * (2 ** attempt),
                    self.config.max_backoff)
        delay *= 1.0 - self.config.jitter * random.random()
        await self._sleep(delay)

        self.stats.reconnects += 1
        self.stats.connects += 1
        if self.on_reconnect is not None:
            # A disconnect ALWAYS loses messages. Announcing it lets the
            # collector REST-backfill the hole; a feed that silently resumes
            # leaves a permanent notch.
            await self.on_reconnect(self.symbol, self.timeframe,
                                    self._last_emitted_ts)
        return attempt + 1

    def _to_bars(self, rows: Sequence[Sequence[float]]) -> list[Bar]:
        """Convert ccxt OHLCV rows, dropping the still-forming final candle.

        `watch_ohlcv` streams the *current* interval, which mutates until it
        closes. Emitting it would hand strategies a bar that is not final —
        look-ahead's mirror image, and just as wrong.
        """
        if not rows:
            return []
        step = timeframe_seconds(self.timeframe)
        now_ms = int(self._now().timestamp() * 1000)
        out: list[Bar] = []
        for r in rows:
            ts_ms = int(r[0])
            if ts_ms + step * 1000 > now_ms:
                continue                       # still forming
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if self._last_emitted_ts is not None and ts <= self._last_emitted_ts:
                continue                       # already emitted
            out.append(Bar(ts=ts, open=float(r[1]), high=float(r[2]),
                           low=float(r[3]), close=float(r[4]),
                           volume=float(r[5]), symbol=self.symbol,
                           timeframe=self.timeframe))
        out.sort(key=lambda b: b.ts)
        return out


def ccxt_watcher(exchange) -> Watcher:
    """Adapt a `ccxt.pro` exchange. Thin by design — the logic is in the feed."""
    async def watch(symbol: str, timeframe: str) -> Sequence[Sequence[float]]:
        return await exchange.watch_ohlcv(symbol, timeframe)
    return watch


__all__ = [
    "BarHandler",
    "FeedConfig",
    "FeedStats",
    "MarketDataFeed",
    "ReconnectHandler",
    "Watcher",
    "ccxt_watcher",
]
