"""Importer for Kraken's bulk OHLCVT CSV archive.

This is how deep history gets in, because the API cannot supply it (§7B). The
archive is a **manual Google Drive download** — the one step in this project
that cannot be automated — after which this module does the rest.

Format, per Kraken's support article: one CSV per pair per interval, named
`<PAIR>_<MINUTES>.csv` (e.g. `XBTUSD_240.csv`), **no header row**, columns:

    timestamp_seconds, open, high, low, close, volume, trades

Two things about the data that matter downstream:

  * **Only intervals with trades are present.** A missing row means no trades,
    not missing data. Gap detection must not treat archive holes as faults, so
    imported bars are marked `is_gap_fill` and the integrity monitor is not fed
    from this path.
  * **Kraken's pair codes are not our symbols.** `XBTUSD` is `BTC/USD`; the
    legacy `X`/`Z` asset prefixes appear inconsistently. `KRAKEN_PAIR_MAP`
    handles the ones we trade and the parser refuses anything it cannot map
    rather than inventing a symbol — a silently mis-mapped file would poison a
    backtest with another asset's prices.

Files are streamed, never read whole: the complete archive is multi-gigabyte and
`qd-worker` has 1 GB.
"""

from __future__ import annotations

import csv
import logging
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from ct.models import Bar

log = logging.getLogger(__name__)

# Minutes → our timeframe strings. Kraken ships 1/5/15/30/60/240/720/1440.
MINUTES_TO_TIMEFRAME = {
    1: "1m", 5: "5m", 15: "15m", 30: "30m",
    60: "1h", 240: "4h", 720: "12h", 1440: "1d",
}

# Kraken's pair codes → our symbols. Deliberately explicit: guessing at the
# X/Z prefix rules produces a plausible-looking symbol for the wrong asset.
KRAKEN_PAIR_MAP = {
    "XBTUSD": "BTC/USD", "XXBTZUSD": "BTC/USD",
    "ETHUSD": "ETH/USD", "XETHZUSD": "ETH/USD",
    "SOLUSD": "SOL/USD",
    "ADAUSD": "ADA/USD",
    "DOTUSD": "DOT/USD",
    "LTCUSD": "LTC/USD", "XLTCZUSD": "LTC/USD",
    "LINKUSD": "LINK/USD",
    "XBTEUR": "BTC/EUR", "XXBTZEUR": "BTC/EUR",
    "ETHEUR": "ETH/EUR", "XETHZEUR": "ETH/EUR",
    "XBTUSDT": "BTC/USDT",
    "ETHUSDT": "ETH/USDT",
}


class UnknownPairError(ValueError):
    """A filename whose pair code is not in `KRAKEN_PAIR_MAP`."""


@dataclass
class ImportStats:
    files: int = 0
    rows: int = 0
    bars: int = 0
    skipped_rows: int = 0
    skipped_files: list[str] = field(default_factory=list)
    symbols: set[str] = field(default_factory=set)
    earliest: datetime | None = None
    latest: datetime | None = None

    def note(self, bar: Bar) -> None:
        self.bars += 1
        self.symbols.add(bar.symbol)
        if self.earliest is None or bar.ts < self.earliest:
            self.earliest = bar.ts
        if self.latest is None or bar.ts > self.latest:
            self.latest = bar.ts

    def summary(self) -> str:
        # Both are set together in note(), so one being None means both are —
        # but that is an invariant mypy cannot see, and stating it is cheaper
        # than a cast.
        span = ("no bars" if self.earliest is None or self.latest is None else
                f"{self.earliest.isoformat()[:10]} → {self.latest.isoformat()[:10]}")
        out = (f"{self.bars:,} bars from {self.files} files across "
               f"{len(self.symbols)} symbols, {span}")
        if self.skipped_rows:
            out += f"; {self.skipped_rows:,} malformed rows skipped"
        if self.skipped_files:
            out += (f"; {len(self.skipped_files)} files skipped "
                    f"({', '.join(self.skipped_files[:5])}"
                    f"{'...' if len(self.skipped_files) > 5 else ''})")
        return out


def parse_filename(name: str) -> tuple[str, str]:
    """`XBTUSD_240.csv` → `("BTC/USD", "4h")`. Raises on anything unmapped."""
    stem = Path(name).stem
    if "_" not in stem:
        raise UnknownPairError(f"{name}: expected <PAIR>_<MINUTES>.csv")
    pair, _, minutes = stem.rpartition("_")
    try:
        tf = MINUTES_TO_TIMEFRAME[int(minutes)]
    except (ValueError, KeyError) as exc:
        raise UnknownPairError(f"{name}: unsupported interval {minutes!r}") from exc
    symbol = KRAKEN_PAIR_MAP.get(pair.upper())
    if symbol is None:
        raise UnknownPairError(
            f"{name}: pair {pair!r} is not in KRAKEN_PAIR_MAP. Add it "
            f"explicitly — guessing the X/Z prefix rules produces a valid-looking "
            f"symbol for the wrong asset.")
    return symbol, tf


def read_csv(handle: Iterable[str], symbol: str, timeframe: str,
             stats: ImportStats | None = None) -> Iterator[Bar]:
    """Stream one Kraken OHLCVT CSV. Malformed rows are skipped and counted.

    A single bad line must not abort the import of a multi-gigabyte archive —
    the correct behaviour is to import everything readable and report what was
    not, exactly as the event log does with a torn final line.
    """
    for lineno, row in enumerate(csv.reader(handle), 1):
        if not row or row[0].lstrip().startswith(("#", "time")):
            continue                      # tolerate a header if one appears
        try:
            ts = datetime.fromtimestamp(int(float(row[0])), tz=timezone.utc)
            o, h, l, c = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            v = float(row[5]) if len(row) > 5 else 0.0
        except (ValueError, IndexError):
            if stats is not None:
                stats.skipped_rows += 1
            if lineno <= 5:
                log.warning("%s %s: malformed row %d: %r", symbol, timeframe,
                            lineno, row[:8])
            continue
        if h < l or min(o, h, l, c) <= 0:
            if stats is not None:
                stats.skipped_rows += 1
            continue
        yield Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v,
                  symbol=symbol, timeframe=timeframe)


def iter_archive(
    path: str | Path,
    *,
    symbols: set[str] | None = None,
    timeframes: set[str] | None = None,
    stats: ImportStats | None = None,
) -> Iterator[Bar]:
    """Stream bars from a Kraken archive — a `.zip` or a directory of CSVs.

    Filter with `symbols`/`timeframes`: the full archive holds every pair at
    eight intervals, and importing all of it to trade four symbols on 4h bars
    would waste hours and gigabytes.
    """
    path = Path(path)
    stats = stats if stats is not None else ImportStats()

    if path.is_dir():
        members: list[tuple[str, object]] = [
            (p.name, p) for p in sorted(path.rglob("*.csv"))]
        # A factory, not a leak: the handle is closed by the consumer that
        # iterates it, and a `with` here would close the file before a single
        # row was read. Directory members and zip members go through this one
        # path, which is why it is a callable rather than an open file.
        opener: Callable[[Any], IO[str]]
        opener = lambda m: open(m, encoding="utf-8", newline="")   # noqa: E731, SIM115
    elif path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        members = [(n, n) for n in sorted(zf.namelist())
                   if n.lower().endswith(".csv")]
        import io
        opener = lambda m: io.TextIOWrapper(zf.open(m), encoding="utf-8")  # noqa: E731
    else:
        raise ValueError(f"{path}: expected a .zip archive or a directory of CSVs")

    if not members:
        raise ValueError(f"{path}: no CSV files found")

    for name, member in members:
        try:
            symbol, tf = parse_filename(name)
        except UnknownPairError as exc:
            stats.skipped_files.append(Path(name).name)
            log.debug("skipping %s: %s", name, exc)
            continue
        if symbols is not None and symbol not in symbols:
            continue
        if timeframes is not None and tf not in timeframes:
            continue

        stats.files += 1
        log.info("importing %s as %s %s", name, symbol, tf)
        with opener(member) as fh:
            for bar in read_csv(fh, symbol, tf, stats):
                stats.rows += 1
                stats.note(bar)
                yield bar


__all__ = [
    "KRAKEN_PAIR_MAP",
    "MINUTES_TO_TIMEFRAME",
    "ImportStats",
    "UnknownPairError",
    "iter_archive",
    "parse_filename",
    "read_csv",
]
