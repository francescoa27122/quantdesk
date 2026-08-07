"""Candle persistence.

Writes are idempotent (`ON CONFLICT ... DO UPDATE`) because every path that
produces a bar can produce it twice: a websocket reconnect replays, a REST
backfill overlaps its own pages, and the CSV importer may be re-run over an
archive that already landed. Duplicate-tolerance is the normal case here, not
an error path.

`is_gap_fill` distinguishes bars that arrived live from ones fetched or imported
afterwards, so a later question of "was this price streamed or reconstructed?"
has an answer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import cast

from ct.data.integrity import timeframe_seconds
from ct.models import Bar
from ct.store.db import Database

log = logging.getLogger(__name__)

_UPSERT = """
INSERT INTO candles (exchange, symbol, timeframe, ts, open, high, low, close,
                     volume, is_gap_fill)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
ON CONFLICT (exchange, symbol, timeframe, ts) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume
"""


class CandleRepository:
    def __init__(self, db: Database, *, exchange: str = "kraken") -> None:
        self.db = db
        self.exchange = exchange

    async def upsert_many(self, bars: Sequence[Bar], *,
                          is_gap_fill: bool = False) -> int:
        if not bars:
            return 0
        rows = [(self.exchange, b.symbol, b.timeframe, b.ts,
                 b.open, b.high, b.low, b.close, b.volume, is_gap_fill)
                for b in bars]
        await self.db.executemany(_UPSERT, rows)
        return len(rows)

    async def upsert_stream(self, bars: Iterable[Bar], *, batch: int = 2000,
                            is_gap_fill: bool = True) -> int:
        """Batched insert for the CSV importer.

        Streams rather than materialising: the full Kraken archive is multiple
        gigabytes and `qd-worker` has 1 GB.
        """
        total = 0
        chunk: list[Bar] = []
        for bar in bars:
            chunk.append(bar)
            if len(chunk) >= batch:
                total += await self.upsert_many(chunk, is_gap_fill=is_gap_fill)
                chunk.clear()
                if total % (batch * 25) == 0:
                    log.info("imported %d candles so far", total)
        if chunk:
            total += await self.upsert_many(chunk, is_gap_fill=is_gap_fill)
        return total

    async def latest_ts(self, symbol: str, timeframe: str) -> datetime | None:
        """The cursor the collector resumes from after a restart."""
        # `fetchval` is Any because asyncpg ships no stubs. `ts` is timestamptz
        # and asyncpg decodes that to an aware datetime, so the cast states a
        # fact about the schema rather than papering over an unknown.
        return cast("datetime | None", await self.db.fetchval(
            """SELECT max(ts) FROM candles
               WHERE exchange=$1 AND symbol=$2 AND timeframe=$3""",
            self.exchange, symbol, timeframe))

    async def earliest_ts(self, symbol: str, timeframe: str) -> datetime | None:
        return cast("datetime | None", await self.db.fetchval(
            """SELECT min(ts) FROM candles
               WHERE exchange=$1 AND symbol=$2 AND timeframe=$3""",
            self.exchange, symbol, timeframe))

    async def count(self, symbol: str, timeframe: str) -> int:
        return int(await self.db.fetchval(
            """SELECT count(*) FROM candles
               WHERE exchange=$1 AND symbol=$2 AND timeframe=$3""",
            self.exchange, symbol, timeframe) or 0)

    async def load(self, symbol: str, timeframe: str, *,
                   since: datetime | None = None,
                   until: datetime | None = None,
                   limit: int | None = None) -> list[Bar]:
        rows = await self.db.fetch(
            """SELECT ts, open, high, low, close, volume FROM candles
               WHERE exchange=$1 AND symbol=$2 AND timeframe=$3
                 AND ($4::timestamptz IS NULL OR ts >= $4)
                 AND ($5::timestamptz IS NULL OR ts <= $5)
               ORDER BY ts
               LIMIT $6""",
            self.exchange, symbol, timeframe, since, until, limit)
        return [Bar(ts=r["ts"], open=float(r["open"]), high=float(r["high"]),
                    low=float(r["low"]), close=float(r["close"]),
                    volume=float(r["volume"]), symbol=symbol,
                    timeframe=timeframe) for r in rows]

    async def find_gaps(self, symbol: str, timeframe: str, *,
                        since: datetime | None = None,
                        max_gaps: int = 100,
                        ) -> list[tuple[datetime, datetime, int]]:
        """Holes in stored data, as (after, before, missing_bars).

        Uses a window function rather than pulling every timestamp into Python:
        eight years of 1m bars is four million rows, and the answer is usually
        "no gaps".

        **Only meaningful for live-collected data.** Kraken's CSV archive omits
        intervals with no trades, so on imported history a "gap" is usually a
        quiet market — which is why the importer marks bars `is_gap_fill` and
        this filters them out.
        """
        step = timeframe_seconds(timeframe)
        rows = await self.db.fetch(
            f"""
            WITH ordered AS (
                SELECT ts, lag(ts) OVER (ORDER BY ts) AS prev
                FROM candles
                WHERE exchange=$1 AND symbol=$2 AND timeframe=$3
                  AND NOT is_gap_fill
                  AND ($4::timestamptz IS NULL OR ts >= $4)
            )
            SELECT prev, ts,
                   (EXTRACT(EPOCH FROM (ts - prev))::bigint / {step}) - 1 AS missing
            FROM ordered
            WHERE prev IS NOT NULL
              AND EXTRACT(EPOCH FROM (ts - prev)) > {step}
            ORDER BY prev
            LIMIT $5
            """,
            self.exchange, symbol, timeframe, since, max_gaps)
        return [(r["prev"], r["ts"], int(r["missing"])) for r in rows]

    async def coverage(self, symbol: str, timeframe: str) -> dict[str, object]:
        """What we actually have — the honest answer to "can I backtest this?"."""
        row = await self.db.fetchrow(
            """SELECT min(ts) lo, max(ts) hi, count(*) n,
                      count(*) FILTER (WHERE is_gap_fill) imported
               FROM candles WHERE exchange=$1 AND symbol=$2 AND timeframe=$3""",
            self.exchange, symbol, timeframe)
        if row is None or row["n"] == 0:
            return {"symbol": symbol, "timeframe": timeframe, "bars": 0,
                    "span_days": 0.0, "completeness": 0.0, "imported": 0}
        span = (row["hi"] - row["lo"]) if row["hi"] and row["lo"] else timedelta(0)
        expected = max(1, int(span.total_seconds() // timeframe_seconds(timeframe)) + 1)
        return {
            "symbol": symbol, "timeframe": timeframe,
            "bars": int(row["n"]),
            "earliest": row["lo"], "latest": row["hi"],
            "span_days": round(span.total_seconds() / 86_400, 1),
            "completeness": round(int(row["n"]) / expected, 4),
            "imported": int(row["imported"]),
        }


__all__ = ["CandleRepository"]
