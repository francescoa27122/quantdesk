"""REST backfill — cursor-based, because `since` works inside the window.

Measured 2026-07-28: Kraken honours `since` as a forward cursor (10 days ago →
240 candles) but retains only ~720 intervals per timeframe. So:

  * **Filling a gap after a reconnect works.** That is what this module is for,
    and it is the collector's whole recovery story.
  * **Fetching deep history does not.** No amount of paging reaches 2018.
    That comes from `ct/data/history.py` and a bulk CSV download.

The paging loop is written to terminate on three independent conditions —
cursor stopped advancing, no rows returned, or the requested end reached —
because a pagination loop that only checks one of those is how a rate-limited
backfill turns into an infinite loop against an exchange that will eventually
ban you for it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from ct.data.integrity import timeframe_seconds
from ct.models import Bar

log = logging.getLogger(__name__)

# (symbol, timeframe, since_ms, limit) -> raw ccxt OHLCV rows
Fetcher = Callable[[str, str, int | None, int], Awaitable[Sequence[Sequence[float]]]]


@dataclass
class BackfillResult:
    symbol: str
    timeframe: str
    bars: list[Bar] = field(default_factory=list)
    requests: int = 0
    truncated: bool = False
    reason: str = ""

    @property
    def covered(self) -> tuple[datetime, datetime] | None:
        if not self.bars:
            return None
        return self.bars[0].ts, self.bars[-1].ts

    def summary(self) -> str:
        if not self.bars:
            return (f"{self.symbol} {self.timeframe}: nothing fetched"
                    + (f" ({self.reason})" if self.reason else ""))
        a, b = self.covered            # type: ignore[misc]
        return (f"{self.symbol} {self.timeframe}: {len(self.bars)} bars "
                f"{a.isoformat()[:16]} → {b.isoformat()[:16]} "
                f"in {self.requests} requests"
                + (f" (TRUNCATED: {self.reason})" if self.truncated else ""))


class Backfiller:
    """Pages `fetch_ohlcv` forward from a cursor.

    `fetcher` is injected rather than an exchange being constructed here, so the
    paging logic — the part with the infinite-loop risk — is testable without a
    network.
    """

    def __init__(
        self,
        fetcher: Fetcher,
        *,
        page_limit: int = 720,
        max_requests: int = 200,
        request_pause: float = 1.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.fetcher = fetcher
        # Injected so "is this bar closed yet?" is testable without sleeping.
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self.page_limit = page_limit
        # Hard ceiling. Kraken retains ~720 intervals, so any run needing
        # hundreds of pages is asking for history that does not exist — and the
        # correct outcome is to stop and say so, not to hammer the venue.
        self.max_requests = max_requests
        # Kraken's ccxt rateLimit is 1000 ms. Sleeping between pages costs
        # nothing on a gap fill and is the difference between a good citizen and
        # an IP ban.
        self.request_pause = request_pause

    async def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> BackfillResult:
        step = timedelta(seconds=timeframe_seconds(timeframe))
        step_ms = int(step.total_seconds() * 1000)
        result = BackfillResult(symbol, timeframe)
        cursor = int(since.timestamp() * 1000) if since else None
        end_ms = int(until.timestamp() * 1000) if until else None
        # Completeness is measured against the REAL clock, never against
        # `until`. Those are different questions: `until` bounds the range you
        # asked for, while "has this bar closed?" is about wall time. Folding
        # them together drops the final bar of every historical range, which is
        # a legitimate closed bar.
        now_ms = int(self._now().timestamp() * 1000)
        seen: set[int] = set()

        while result.requests < self.max_requests:
            rows = await self.fetcher(symbol, timeframe, cursor, self.page_limit)
            result.requests += 1

            if not rows:
                result.reason = "exchange returned no rows"
                break

            fresh = 0
            for r in rows:
                ts_ms = int(r[0])
                if ts_ms in seen:
                    continue                       # pages overlap by design
                if end_ms is not None and ts_ms > end_ms:
                    continue                       # outside the range asked for
                # Drop the still-forming candle, exactly as `MarketDataFeed`
                # does: a bar is complete only once its interval has ELAPSED,
                # so the test is on its close (`ts + step`), not its open.
                #
                # This one cost a deployment. Without it the current, mutating
                # candle was stored; `mark_backfilled` then advanced the
                # integrity cursor to it, and every genuinely-closed bar the
                # websocket delivered afterwards looked out-of-order. The
                # collector stored **zero** live bars and sat DEGRADED, which
                # would have held the engine's STALE_DATA gate shut. Found by
                # running the stack on the NAS, not by reading it.
                if ts_ms + step_ms > now_ms:
                    continue
                seen.add(ts_ms)
                fresh += 1
                result.bars.append(Bar(
                    ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                    open=float(r[1]), high=float(r[2]), low=float(r[3]),
                    close=float(r[4]), volume=float(r[5]),
                    symbol=symbol, timeframe=timeframe))

            newest = max(int(r[0]) for r in rows)
            if fresh == 0:
                # Every row was already seen. The cursor is not advancing —
                # this is the loop that would otherwise never end.
                result.reason = "cursor stopped advancing (window exhausted)"
                break
            if end_ms is not None and newest >= end_ms:
                break
            if len(rows) < self.page_limit:
                result.reason = "short page — reached the end of available data"
                break

            cursor = newest + int(step.total_seconds() * 1000)
            if self.request_pause:
                await asyncio.sleep(self.request_pause)
        else:
            result.truncated = True
            result.reason = (f"hit the {self.max_requests}-request ceiling; the "
                             f"exchange does not retain this much history")
            log.warning("backfill truncated for %s %s", symbol, timeframe)

        result.bars.sort(key=lambda b: b.ts)
        log.info("backfill: %s", result.summary())
        return result

    async def fill_gap(
        self, symbol: str, timeframe: str, *, after: datetime, before: datetime,
    ) -> BackfillResult:
        """Fill the hole between two known-good bars.

        `after` is the last bar we have; the fetch starts one interval later so
        the known bar is not re-fetched. Overlap would be harmless — storage
        upserts — but a backfill that always re-reads its own starting point is
        one that never makes progress if the interval is ever miscomputed.
        """
        step = timedelta(seconds=timeframe_seconds(timeframe))
        return await self.fetch_range(symbol, timeframe,
                                      since=after + step, until=before)

    async def catch_up(
        self, symbol: str, timeframe: str, *, last_known: datetime | None,
        now: datetime, max_lookback: timedelta | None = None,
    ) -> BackfillResult:
        """Fetch everything since the last stored bar. The reconnect path.

        With no stored data, looks back `max_lookback` rather than to the
        beginning of time. `None` means "the venue's actual window": Kraken
        serves ~720 BARS per timeframe whatever the clock span (A11), so the
        default is bar-count-shaped. The old flat 7-day default was 42 bars
        on 4h — less than any strategy's warmup — so a fresh database never
        accumulated enough history for the engine to consult a strategy.
        """
        step = timedelta(seconds=timeframe_seconds(timeframe))
        if max_lookback is None:
            max_lookback = step * 750       # venue window (~720) + margin
        if last_known is None:
            since = now - max_lookback
        else:
            since = last_known + step
            if now - since > max_lookback:
                log.warning("%s %s is %s behind; capping lookback at %s",
                            symbol, timeframe, now - since, max_lookback)
                since = now - max_lookback
        if since >= now:
            return BackfillResult(symbol, timeframe, reason="already current")
        return await self.fetch_range(symbol, timeframe, since=since, until=now)


def ccxt_fetcher(exchange: Any) -> Fetcher:
    """Adapt a ccxt async exchange to the `Fetcher` signature.

    Thin on purpose: everything worth testing lives in `Backfiller`, and this
    is the seam that keeps the network out of it.
    """
    async def fetch(symbol: str, timeframe: str, since: int | None,
                    limit: int) -> Sequence[Sequence[float]]:
        # ccxt ships no type information; the cast records the documented
        # OHLCV row shape rather than letting Any leak into the backfiller.
        return cast("Sequence[Sequence[float]]",
                    await exchange.fetch_ohlcv(symbol, timeframe, since=since,
                                               limit=limit))
    return fetch


__all__ = ["BackfillResult", "Backfiller", "Fetcher", "ccxt_fetcher"]
