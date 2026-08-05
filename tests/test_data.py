"""Market-data service tests.

The load-bearing ones here are about *not trading on bad data*: staleness,
gaps, reconnect-implies-backfill, and the paging loop terminating. Every risk
gate downstream is computed from a price, so a feed that fails silently does
not degrade decisions — it invalidates all of them at once while everything
still looks healthy.

No network. `Watcher` and `Fetcher` are injected precisely so the reconnect and
paging logic — the parts with infinite-loop and thundering-herd risk — are
deterministic.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

from ct.data import (
    Backfiller,
    CollectorConfig,
    DataIntegrityMonitor,
    FaultKind,
    FeedConfig,
    IntegrityConfig,
    IntegrityState,
    MarketDataCollector,
    MarketDataFeed,
    iter_archive,
    parse_filename,
    timeframe_seconds,
)
from ct.data.history import ImportStats, UnknownPairError, read_csv
from ct.models import Bar

TS = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
H = timedelta(hours=1)


def bar(ts: datetime, close=100.0, symbol="BTC/USD", tf="1h", **kw) -> Bar:
    o = kw.pop("open", close)
    hi = kw.pop("high", max(o, close) * 1.001)
    lo = kw.pop("low", min(o, close) * 0.999)
    return Bar(ts=ts, open=o, high=hi, low=lo, close=close,
               volume=kw.pop("volume", 10.0), symbol=symbol, timeframe=tf)


# ══════════════════════════════════════════════════════════════════════════
# Integrity — the component that can stop trading
# ══════════════════════════════════════════════════════════════════════════

def test_unknown_symbol_is_not_tradeable_before_any_data():
    """UNKNOWN and OK must never collapse into the same answer. 'We have never
    seen a price' is not 'the price is fine'."""
    m = DataIntegrityMonitor()
    m.track("BTC/USD", "1h")
    assert m.health[("BTC/USD", "1h")].state is IntegrityState.UNKNOWN
    assert not m.tradeable("BTC/USD", "1h")
    assert not m.tradeable("NEVER/SEEN", "1h")


def test_clean_bars_bring_a_symbol_up_after_the_settling_period():
    m = DataIntegrityMonitor(IntegrityConfig(recovery_bars=3))
    for i in range(2):
        assert m.accept(bar(TS + i * H))
    assert not m.tradeable("BTC/USD", "1h"), "must not be tradeable before settling"
    assert m.accept(bar(TS + 2 * H))
    assert m.tradeable("BTC/USD", "1h")


def _bring_up(m: DataIntegrityMonitor, n: int = 3, start: datetime = TS) -> datetime:
    for i in range(n):
        m.accept(bar(start + i * H))
    return start + (n - 1) * H


def test_duplicate_bars_are_dropped_quietly_not_treated_as_faults():
    """Routine on reconnect. Counting them as faults would halt trading every
    time the socket blinked."""
    m = DataIntegrityMonitor()
    last = _bring_up(m)
    assert m.tradeable("BTC/USD", "1h")
    assert m.accept(bar(last)) is False
    h = m.health[("BTC/USD", "1h")]
    assert h.duplicates_dropped == 1
    assert h.state is IntegrityState.OK, "a duplicate must not degrade the feed"


def test_out_of_order_bar_degrades_and_is_dropped():
    m = DataIntegrityMonitor()
    last = _bring_up(m)
    assert m.accept(bar(last - 5 * H, close=101.0)) is False
    h = m.health[("BTC/USD", "1h")]
    assert h.out_of_order_dropped == 1
    assert h.state is IntegrityState.DEGRADED
    assert h.fault.kind is FaultKind.OUT_OF_ORDER
    assert not m.tradeable("BTC/USD", "1h")


def test_small_gap_is_recorded_but_stays_tradeable():
    """Small holes are backfillable; halting on them would halt constantly."""
    m = DataIntegrityMonitor(IntegrityConfig(max_autofill_gap_bars=10))
    last = _bring_up(m)
    assert m.accept(bar(last + 4 * H))
    h = m.health[("BTC/USD", "1h")]
    assert h.gaps_detected == 1 and h.missing_bars == 3
    assert h.state is IntegrityState.OK


def test_large_gap_degrades():
    m = DataIntegrityMonitor(IntegrityConfig(max_autofill_gap_bars=5))
    last = _bring_up(m)
    assert m.accept(bar(last + 50 * H))
    h = m.health[("BTC/USD", "1h")]
    assert h.state is IntegrityState.DEGRADED
    assert h.fault.kind is FaultKind.GAP
    assert "49 bars missing" in h.fault.detail


@pytest.mark.parametrize("bad,reason", [
    (dict(open=-1.0), "non-positive"),
    (dict(open=100.0, high=90.0, low=95.0, close=92.0), "high"),
    (dict(open=100.0, high=101.0, low=99.0, close=150.0), "outside"),
    (dict(open=100.0, high=200.0, low=99.0, close=150.0), "intrabar move"),
])
def test_implausible_bars_are_rejected(bad, reason):
    """One bad tick can trip every stop in the book."""
    m = DataIntegrityMonitor()
    _bring_up(m)
    b = Bar(ts=TS + 10 * H, volume=1.0, symbol="BTC/USD", timeframe="1h",
            **{**dict(open=100.0, high=101.0, low=99.0, close=100.0), **bad})
    assert m.accept(b) is False
    h = m.health[("BTC/USD", "1h")]
    assert h.implausible_dropped == 1
    assert reason in h.fault.detail


def test_staleness_is_detected_by_the_clock_not_by_arrivals():
    """A socket that stops delivering looks exactly like a market with no
    trades. Only elapsed time tells them apart."""
    m = DataIntegrityMonitor(IntegrityConfig(max_stale_bars=2))
    last = _bring_up(m)
    assert m.check_staleness(last + 1 * H) == []
    faults = m.check_staleness(last + 6 * H)
    assert len(faults) == 1 and faults[0].kind is FaultKind.STALE
    assert not m.tradeable("BTC/USD", "1h")


def test_recovery_requires_the_full_settling_period():
    """Resuming on the first good bar means flapping in and out of halt during
    exactly the conditions that caused the fault."""
    m = DataIntegrityMonitor(IntegrityConfig(recovery_bars=3, max_autofill_gap_bars=1))
    last = _bring_up(m)
    m.accept(bar(last + 50 * H))                     # degrade via a big gap
    assert not m.tradeable("BTC/USD", "1h")
    base = last + 50 * H
    m.accept(bar(base + 1 * H))
    assert not m.tradeable("BTC/USD", "1h")
    m.accept(bar(base + 2 * H))
    assert not m.tradeable("BTC/USD", "1h")
    m.accept(bar(base + 3 * H))
    assert m.tradeable("BTC/USD", "1h")


def test_data_age_matches_the_shape_the_risk_engine_wants():
    """Fed straight into `AccountState.data_age_bars`, so the STALE_DATA gate
    and this report cannot disagree."""
    m = DataIntegrityMonitor()
    last = _bring_up(m)
    ages = m.data_age_bars(last + 5 * H)
    assert ages["BTC/USD"] == 4


def test_unknown_timeframe_fails_loudly():
    with pytest.raises(ValueError, match="unknown timeframe"):
        timeframe_seconds("3h")
    with pytest.raises(ValueError, match="unknown timeframe"):
        DataIntegrityMonitor().track("BTC/USD", "7m")


# ══════════════════════════════════════════════════════════════════════════
# Backfill — the paging loop that must always terminate
# ══════════════════════════════════════════════════════════════════════════

def _rows(start: datetime, n: int, step_s: int = 3600):
    return [[int((start + timedelta(seconds=step_s * i)).timestamp() * 1000),
             100.0, 101.0, 99.0, 100.5, 5.0] for i in range(n)]


def test_backfill_pages_forward_until_a_short_page():
    calls: list[int | None] = []

    async def fetcher(symbol, tf, since, limit):
        calls.append(since)
        base = (datetime.fromtimestamp(since / 1000, tz=timezone.utc)
                if since else TS)
        n = limit if len(calls) < 3 else 10          # third page is short
        return _rows(base, n)

    b = Backfiller(fetcher, page_limit=100, request_pause=0)
    res = asyncio.run(b.fetch_range("BTC/USD", "1h", since=TS))
    assert res.requests == 3
    assert len(res.bars) == 210
    assert not res.truncated
    assert "short page" in res.reason
    assert res.bars == sorted(res.bars, key=lambda x: x.ts)


def test_backfill_stops_when_the_cursor_stops_advancing():
    """The loop that would otherwise never end: an exchange that keeps
    returning the same window."""
    async def stuck(symbol, tf, since, limit):
        return _rows(TS, limit)                      # always identical

    b = Backfiller(stuck, page_limit=50, max_requests=100, request_pause=0)
    res = asyncio.run(b.fetch_range("BTC/USD", "1h", since=TS))
    assert res.requests == 2, "must notice on the second page, not the hundredth"
    assert "cursor stopped advancing" in res.reason
    assert len(res.bars) == 50


def test_backfill_respects_the_request_ceiling():
    async def endless(symbol, tf, since, limit):
        base = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
        return _rows(base, limit)

    b = Backfiller(endless, page_limit=10, max_requests=5, request_pause=0)
    res = asyncio.run(b.fetch_range("BTC/USD", "1h", since=TS))
    assert res.requests == 5
    assert res.truncated and "ceiling" in res.reason


def test_backfill_honours_until():
    async def fetcher(symbol, tf, since, limit):
        base = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
        return _rows(base, limit)

    b = Backfiller(fetcher, page_limit=100, request_pause=0)
    res = asyncio.run(b.fetch_range("BTC/USD", "1h", since=TS, until=TS + 10 * H))
    assert all(x.ts <= TS + 10 * H for x in res.bars)
    assert len(res.bars) == 11


def test_backfill_deduplicates_overlapping_pages():
    async def overlapping(symbol, tf, since, limit):
        base = datetime.fromtimestamp(since / 1000, tz=timezone.utc) - 5 * H
        return _rows(base, limit)

    b = Backfiller(overlapping, page_limit=20, max_requests=4, request_pause=0)
    res = asyncio.run(b.fetch_range("BTC/USD", "1h", since=TS))
    stamps = [x.ts for x in res.bars]
    assert len(stamps) == len(set(stamps)), "overlapping pages produced duplicates"


def test_catch_up_from_nothing_uses_the_lookback_cap():
    """Kraken has nothing older than its window; asking burns request budget."""
    seen: list[int] = []

    async def fetcher(symbol, tf, since, limit):
        seen.append(since)
        return _rows(datetime.fromtimestamp(since / 1000, tz=timezone.utc), 5)

    b = Backfiller(fetcher, page_limit=100, request_pause=0)
    now = TS + timedelta(days=400)
    asyncio.run(b.catch_up("BTC/USD", "1h", last_known=None, now=now,
                           max_lookback=timedelta(days=7)))
    asked = datetime.fromtimestamp(seen[0] / 1000, tz=timezone.utc)
    assert (now - asked) <= timedelta(days=7, hours=1)


def test_catch_up_is_a_no_op_when_already_current():
    async def never(symbol, tf, since, limit):
        raise AssertionError("must not call the exchange when already current")

    b = Backfiller(never, request_pause=0)
    res = asyncio.run(b.catch_up("BTC/USD", "1h", last_known=TS, now=TS))
    assert res.bars == [] and res.reason == "already current"


# ══════════════════════════════════════════════════════════════════════════
# Feed — reconnect, backoff, silence
# ══════════════════════════════════════════════════════════════════════════

def test_feed_drops_the_still_forming_candle():
    """`watch_ohlcv` streams the CURRENT interval, which mutates until close.
    Emitting it hands strategies a non-final bar — look-ahead's mirror image."""
    now = TS + 5 * H + timedelta(minutes=30)
    emitted: list[Bar] = []

    async def watcher(symbol, tf):
        return _rows(TS, 6)          # the 6th (TS+5h) is still forming at `now`

    feed = MarketDataFeed(watcher, "BTC/USD", "1h", on_bar=_collect(emitted),
                          clock=lambda: now)
    bars = feed._to_bars(_rows(TS, 6))
    assert len(bars) == 5
    assert all(b.ts + H <= now for b in bars)


def test_feed_backs_off_with_jitter_and_announces_reconnects():
    delays: list[float] = []
    reconnects: list[tuple[str, str]] = []
    attempts = {"n": 0}

    async def flaky(symbol, tf):
        attempts["n"] += 1
        if attempts["n"] <= 4:
            raise ConnectionError("socket closed")
        feed.stop()
        return _rows(TS, 3)

    async def sleeper(d):
        delays.append(d)

    async def on_reconnect(sym, tf, last):
        reconnects.append((sym, tf))

    emitted: list[Bar] = []
    feed = MarketDataFeed(flaky, "BTC/USD", "1h", on_bar=_collect(emitted),
                          on_reconnect=on_reconnect,
                          config=FeedConfig(base_backoff=1.0, max_backoff=60.0,
                                            jitter=0.5),
                          clock=lambda: TS + 100 * H, sleeper=sleeper)
    asyncio.run(feed.run())

    assert len(delays) == 4
    assert delays == sorted(delays), "backoff must be non-decreasing"
    assert all(0 < d <= 60.0 for d in delays)
    # Jitter means delays are not the bare powers of two.
    assert delays != [1.0, 2.0, 4.0, 8.0]
    assert len(reconnects) == 4, "every reconnect must be announced for backfill"
    assert feed.stats.reconnects == 4


def test_feed_gives_up_loudly_rather_than_retrying_forever():
    """A feed that retries forever is indistinguishable from one that works."""
    async def always_fails(symbol, tf):
        raise ConnectionError("down")

    async def sleeper(d):
        return None

    feed = MarketDataFeed(always_fails, "BTC/USD", "1h", on_bar=_noop,
                          config=FeedConfig(max_consecutive_failures=5),
                          clock=lambda: TS, sleeper=sleeper)
    asyncio.run(feed.run())
    assert feed.stats.given_up
    assert feed.stats.consecutive_failures >= 5
    assert "down" in feed.stats.last_error


def test_feed_treats_silence_as_failure():
    """The common failure is a socket that goes quiet, not one that errors."""
    async def silent(symbol, tf):
        await asyncio.sleep(10)          # longer than the silence limit
        return []

    async def sleeper(d):
        return None

    feed = MarketDataFeed(silent, "BTC/USD", "1h", on_bar=_noop,
                          config=FeedConfig(max_consecutive_failures=2,
                                            min_silence_seconds=0.05,
                                            silence_intervals=0.0),
                          clock=lambda: TS, sleeper=sleeper)
    asyncio.run(feed.run())
    assert feed.stats.given_up
    assert "no message" in feed.stats.last_error


def _collect(sink: list[Bar]):
    async def handler(b: Bar) -> None:
        sink.append(b)
    return handler


async def _noop(b: Bar) -> None:
    return None


# ══════════════════════════════════════════════════════════════════════════
# Collector — the reconnect handshake
# ══════════════════════════════════════════════════════════════════════════

def test_reconnect_triggers_a_backfill():
    """A dropped socket ALWAYS loses messages. Resuming without backfilling
    leaves a permanent notch nothing will ever revisit."""
    calls: list[tuple[str, str]] = []

    async def fetcher(symbol, tf, since, limit):
        calls.append((symbol, tf))
        return []

    async def watcher(symbol, tf):
        return []

    collector = MarketDataCollector(
        watcher, Backfiller(fetcher, request_pause=0), None,
        config=CollectorConfig(symbols=("BTC/USD",), timeframes=("1h",)),
        clock=lambda: TS)
    asyncio.run(collector._handle_reconnect("BTC/USD", "1h", TS))
    assert calls == [("BTC/USD", "1h")]
    assert collector.stats.reconnects == 1


def test_collector_catches_up_before_streaming():
    """The gap between shutdown and restart is the one gap guaranteed to exist
    on every restart."""
    calls: list[tuple[str, str]] = []

    async def fetcher(symbol, tf, since, limit):
        calls.append((symbol, tf))
        return []

    async def watcher(symbol, tf):
        return []

    collector = MarketDataCollector(
        watcher, Backfiller(fetcher, request_pause=0), None,
        config=CollectorConfig(symbols=("BTC/USD", "ETH/USD"),
                               timeframes=("1h", "4h")),
        clock=lambda: TS)
    asyncio.run(collector.catch_up_all())
    assert set(calls) == {("BTC/USD", "1h"), ("BTC/USD", "4h"),
                          ("ETH/USD", "1h"), ("ETH/USD", "4h")}


def test_tradeable_requires_every_timeframe_healthy():
    """A strategy reading 4h while the 1h regime feed is stale is trading on
    half a picture."""
    async def fetcher(symbol, tf, since, limit):
        return []

    async def watcher(symbol, tf):
        return []

    c = MarketDataCollector(
        watcher, Backfiller(fetcher, request_pause=0), None,
        config=CollectorConfig(symbols=("BTC/USD",), timeframes=("1h", "4h")),
        integrity=IntegrityConfig(recovery_bars=1), clock=lambda: TS)

    for i in range(2):
        c.monitor.accept(bar(TS + i * H, tf="1h"))
    assert c.tradeable_symbols() == [], "4h has never been seen"

    for i in range(2):
        c.monitor.accept(bar(TS + i * timedelta(hours=4), tf="4h"))
    assert c.tradeable_symbols() == ["BTC/USD"]


# ══════════════════════════════════════════════════════════════════════════
# Kraken CSV archive import
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,symbol,tf", [
    ("XBTUSD_240.csv", "BTC/USD", "4h"),
    ("XXBTZUSD_60.csv", "BTC/USD", "1h"),
    ("ETHUSD_1440.csv", "ETH/USD", "1d"),
    ("SOLUSD_1.csv", "SOL/USD", "1m"),
])
def test_filename_parsing(name, symbol, tf):
    assert parse_filename(name) == (symbol, tf)


def test_unmapped_pair_is_refused_not_guessed():
    """A silently mis-mapped file poisons a backtest with another asset's
    prices — far worse than a loud failure."""
    with pytest.raises(UnknownPairError, match="KRAKEN_PAIR_MAP"):
        parse_filename("DOGEUSD_60.csv")
    with pytest.raises(UnknownPairError, match="interval"):
        parse_filename("XBTUSD_37.csv")


def test_csv_rows_parse_and_malformed_rows_are_skipped_not_fatal():
    """A single bad line must not abort a multi-gigabyte import."""
    csv_text = "\n".join([
        "1751328000,100.0,101.0,99.0,100.5,12.5,42",
        "not,a,number,at,all,x,y",
        "1751331600,100.5,102.0,100.0,101.5,8.0,31",
        "1751335200,-5,1,1,1,1,1",                    # non-positive
        "1751338800,101.5,103.0,101.0,102.0,9.0,28",
    ])
    stats = ImportStats()
    bars = list(read_csv(io.StringIO(csv_text), "BTC/USD", "1h", stats))
    assert len(bars) == 3
    assert stats.skipped_rows == 2
    assert bars[0].close == 100.5
    assert bars[0].ts.tzinfo is timezone.utc


def test_import_a_zip_archive_filtered_by_symbol_and_timeframe(tmp_path):
    zpath = tmp_path / "kraken.zip"
    rows = "\n".join(
        f"{int((TS + i * H).timestamp())},100,101,99,100.5,5,3" for i in range(4))
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("XBTUSD_60.csv", rows)
        zf.writestr("ETHUSD_60.csv", rows)
        zf.writestr("XBTUSD_1440.csv", rows)
        zf.writestr("DOGEUSD_60.csv", rows)        # unmapped → skipped
        zf.writestr("README.txt", "not a csv")

    stats = ImportStats()
    bars = list(iter_archive(zpath, symbols={"BTC/USD"}, timeframes={"1h"},
                             stats=stats))
    assert len(bars) == 4
    assert {b.symbol for b in bars} == {"BTC/USD"}
    assert {b.timeframe for b in bars} == {"1h"}
    assert stats.files == 1
    assert "DOGEUSD_60.csv" in stats.skipped_files
    assert "4 bars from 1 files" in stats.summary()


def test_import_a_directory_of_csvs(tmp_path):
    (tmp_path / "XBTUSD_240.csv").write_text(
        "\n".join(f"{int((TS + i * 4 * H).timestamp())},1,2,0.5,1.5,3,2"
                  for i in range(3)), encoding="utf-8")
    bars = list(iter_archive(tmp_path))
    assert len(bars) == 3
    assert bars[0].symbol == "BTC/USD" and bars[0].timeframe == "4h"


def test_empty_archive_is_an_error_not_a_silent_zero(tmp_path):
    with pytest.raises(ValueError, match="no CSV files"):
        list(iter_archive(tmp_path))


def test_backfill_advances_the_monitor_so_recovery_is_not_a_false_halt():
    """Found by running the collector against real Kraken.

    After a reconnect the collector REST-fills the hole, but the monitor kept
    its pre-disconnect cursor — so the first live bar afterwards measured a gap
    that had already been filled and degraded the feed. A false halt firing
    exactly when the system had just recovered.
    """
    m = DataIntegrityMonitor(IntegrityConfig(recovery_bars=1,
                                             max_autofill_gap_bars=2))
    last = _bring_up(m, 1)
    assert m.tradeable("BTC/USD", "1h")

    # 50 bars pass while disconnected; backfill repairs storage.
    m.mark_backfilled("BTC/USD", "1h", last + 50 * H)

    # The next live bar is contiguous with the backfill, so no gap.
    assert m.accept(bar(last + 51 * H))
    h = m.health[("BTC/USD", "1h")]
    assert h.gaps_detected == 0, "degraded on a gap that backfill had filled"
    assert m.tradeable("BTC/USD", "1h")


def test_mark_backfilled_never_moves_the_cursor_backwards():
    m = DataIntegrityMonitor(IntegrityConfig(recovery_bars=1))
    last = _bring_up(m, 1)
    m.mark_backfilled("BTC/USD", "1h", last - 10 * H)
    assert m.health[("BTC/USD", "1h")].last_bar_ts == last


def test_collector_backfill_marks_the_monitor():
    """End to end: the collector must wire the repair back into the monitor."""
    async def fetcher(symbol, tf, since, limit):
        base = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
        return [[int((base + timedelta(hours=i)).timestamp() * 1000),
                 100.0, 101.0, 99.0, 100.5, 5.0] for i in range(3)]

    async def watcher(symbol, tf):
        return []

    c = MarketDataCollector(
        watcher, Backfiller(fetcher, request_pause=0), None,
        config=CollectorConfig(symbols=("BTC/USD",), timeframes=("1h",)),
        integrity=IntegrityConfig(recovery_bars=1), clock=lambda: TS + 100 * H)
    asyncio.run(c._handle_reconnect("BTC/USD", "1h", None))
    assert c.monitor.health[("BTC/USD", "1h")].last_bar_ts is not None
    assert c.stats.backfilled_bars == 3


def test_backfill_never_stores_the_still_forming_candle():
    """Regression from the first real deployment.

    `MarketDataFeed` drops the forming candle; `Backfiller` did not. So
    `catch_up(until=now)` stored a bar that was still mutating,
    `mark_backfilled` advanced the integrity cursor to it, and every closed bar
    the websocket delivered afterwards looked out-of-order. Observed on the NAS:
    `bars_stored: 0`, `degraded: [BTC/USD, ETH/USD]`, permanently — which would
    have kept the engine's STALE_DATA gate shut.
    """
    now = TS + 10 * H + timedelta(minutes=25)      # 25 min into the 10:00 bar

    async def fetcher(symbol, timeframe, since, limit):
        # The exchange returns the forming candle as its last row, as Kraken does.
        return [[int((TS + i * H).timestamp() * 1000), 100.0, 101.0, 99.0,
                 100.5, 5.0] for i in range(11)]

    b = Backfiller(fetcher, page_limit=100, request_pause=0, clock=lambda: now)
    res = asyncio.run(b.catch_up("BTC/USD", "1h", last_known=None, now=now))

    assert res.bars, "nothing fetched"
    newest = res.bars[-1].ts
    assert newest == TS + 9 * H, f"stored an unclosed bar: {newest}"
    assert all(bar.ts + H <= now for bar in res.bars)


def test_backfill_still_returns_the_last_closed_bar_of_a_historical_range():
    """The completeness rule must not eat legitimate history.

    Range (`until`) and completeness (wall clock) are different questions;
    folding them together dropped the final bar of every past range.
    """
    now = TS + 100 * H                              # long after the range

    async def fetcher(symbol, timeframe, since, limit):
        return [[int((TS + i * H).timestamp() * 1000), 100.0, 101.0, 99.0,
                 100.5, 5.0] for i in range(11)]

    b = Backfiller(fetcher, page_limit=100, request_pause=0, clock=lambda: now)
    res = asyncio.run(b.fetch_range("BTC/USD", "1h", since=TS,
                                    until=TS + 10 * H))
    assert res.bars[-1].ts == TS + 10 * H, "dropped a closed bar inside the range"
    assert len(res.bars) == 11


def test_never_seen_is_reported_separately_from_faulted():
    """`UNKNOWN` is not `DEGRADED`.

    After a restart a symbol has no live bar on some timeframe for up to a full
    interval — a whole day on `1d`. Reporting that as degraded made a healthy
    collector look broken. The states are distinct in `IntegrityState`; the
    reporting must keep them distinct too.
    """
    m = DataIntegrityMonitor()
    m.track("BTC/USD", "1h")
    m.track("BTC/USD", "1d")
    assert m.unknown_symbols() == ["BTC/USD"]
    assert m.degraded_symbols() == [], "never-seen was reported as a fault"

    # One clean bar on 1h; 1d is still unseen, so the symbol is still unknown
    # but still not faulted.
    m.accept(Bar(ts=TS, open=100.0, high=101.0, low=99.0, close=100.5,
                 volume=5.0, symbol="BTC/USD", timeframe="1h"),
             now=TS + timedelta(minutes=1))
    assert m.degraded_symbols() == []


# ══════════════════════════════════════════════════════════════════════════
# 2026-07-30: the two defects that kept the Positions page empty and the
# feed crying wolf. Found in production: asking why he saw no positions.
# ══════════════════════════════════════════════════════════════════════════


def test_default_catch_up_lookback_is_bar_count_shaped():
    """The venue serves ~720 BARS per timeframe whatever the clock span (A11).
    The old flat 7-day default was 42 bars on 4h — less than any strategy's
    warmup — so a fresh database never fed the engine enough history and the
    strategy was never consulted."""
    seen: list[int] = []

    async def fetcher(symbol, tf, since, limit):
        seen.append(since)
        return _rows(datetime.fromtimestamp(since / 1000, tz=timezone.utc), 5)

    b = Backfiller(fetcher, page_limit=100, request_pause=0)
    now = TS + timedelta(days=400)
    asyncio.run(b.catch_up("BTC/USD", "4h", last_known=None, now=now))
    asked = datetime.fromtimestamp(seen[0] / 1000, tz=timezone.utc)
    lookback = now - asked
    assert lookback >= timedelta(hours=4) * 700, (
        f"4h lookback {lookback} is too shallow for a 205-bar warmup")
    assert lookback <= timedelta(hours=4) * 800, "and not absurdly deep"


def test_an_already_stored_bar_is_not_a_fault():
    """Websocket snapshots replay history on every (re)connect; boot backfill
    has stored those bars already. Feeding them to the monitor flagged
    OUT_OF_ORDER on every restart — on 1d the settling period kept the symbol
    reading 'degraded' for days. Crying wolf on every restart is how a real
    degradation gets ignored."""
    async def fetcher(symbol, tf, since, limit):
        return []

    async def watcher(symbol, tf):
        return []

    c = MarketDataCollector(
        watcher, Backfiller(fetcher, request_pause=0), None,
        config=CollectorConfig(symbols=("BTC/USD",), timeframes=("1h",)),
        clock=lambda: TS + timedelta(hours=20))

    fresh = [Bar(ts=TS + i * timedelta(hours=1), open=1.0, high=1.0, low=1.0,
                 close=1.0, volume=1.0, symbol="BTC/USD", timeframe="1h")
             for i in range(12)]
    for bar in fresh:
        asyncio.run(c._handle_bar(bar))
    assert c.stats.bars_stored == 12

    # The reconnect snapshot re-delivers older bars the cursor is past.
    replay = fresh[5]
    asyncio.run(c._handle_bar(replay))
    assert c.stats.bars_already_known == 1
    assert c.stats.bars_rejected == 0, "a known bar is not a rejection"
    assert c.monitor.degraded_symbols() == [], (
        "a replayed known bar must never degrade the feed")
    # A genuinely NEW bar older than the cursor would be out-of-order — that
    # path still exists for bars the cursor has not passed... and a next live
    # bar still lands cleanly:
    nxt = Bar(ts=TS + 12 * timedelta(hours=1), open=1.0, high=1.0, low=1.0,
              close=1.0, volume=1.0, symbol="BTC/USD", timeframe="1h")
    asyncio.run(c._handle_bar(nxt))
    assert c.stats.bars_stored == 13
