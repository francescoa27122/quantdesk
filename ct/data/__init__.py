"""Market data — websocket ingest, integrity, backfill, history import.

    feed.py        websocket with reconnect + jittered backoff + silence timer
    integrity.py   gap / stale / out-of-order / implausibility detection
    backfill.py    REST catch-up over the exchange's retained window
    history.py     Kraken bulk OHLCVT CSV import (deep history; §7B)
    repository.py  idempotent candle persistence
    collector.py   the `qd-collector` entrypoint that wires them together

`ct.data.integrity` and `ct.data.backfill` are pure and have no database or
network dependency, which is why they carry the tests.
"""

from ct.data.backfill import Backfiller, BackfillResult, ccxt_fetcher
from ct.data.collector import CollectorConfig, MarketDataCollector
from ct.data.feed import FeedConfig, MarketDataFeed, ccxt_watcher
from ct.data.history import ImportStats, iter_archive, parse_filename
from ct.data.integrity import (
    DataIntegrityMonitor,
    Fault,
    FaultKind,
    IntegrityConfig,
    IntegrityState,
    SymbolHealth,
    timeframe_seconds,
)
from ct.data.repository import CandleRepository

__all__ = [
    "BackfillResult",
    "Backfiller",
    "CandleRepository",
    "CollectorConfig",
    "DataIntegrityMonitor",
    "Fault",
    "FaultKind",
    "FeedConfig",
    "ImportStats",
    "IntegrityConfig",
    "IntegrityState",
    "MarketDataCollector",
    "MarketDataFeed",
    "SymbolHealth",
    "ccxt_fetcher",
    "ccxt_watcher",
    "iter_archive",
    "parse_filename",
    "timeframe_seconds",
]
