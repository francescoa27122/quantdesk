"""Chaos and failure injection — Phase 2 item 10.

Every other test file asks "does this component work?". This one asks "what
happens when something else breaks underneath it?", and it does so **without a
database**, deliberately: the outbox, reconciliation and backpressure claims are
currently only exercised in `tests/test_store.py`, which skips unless
`QD_TEST_DSN` is set. A reliability guarantee that is only checked in an
environment nobody runs by default is a rumour with a test file next to it.

The four claims under test here, each quoted from the design it comes from:

* **"`publish()` never raises and never blocks"** (ARCHITECTURE §8A). Losing
  telemetry must never cost a stop-loss.
* **"`positions` is derivable by replaying `fills`"** (§3). Replay shares no
  code with the live bookkeeping, so agreement is real evidence rather than the
  same bug counted twice.
* **"The engine starts halted and must reconcile clean before it will place an
  order"** (§6.3) — including across a restart.
* **"A duplicate is a normal outcome, not an error"** (`OrderExecutor`).
  Exchanges re-deliver and restarts replay; the position must move once.

`ct/backtest` is deliberately absent: a backtest that fails is a wrong number,
not an outage.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ct.engine import Allocation, TradingEngine
from ct.engine.execution import OrderExecutor
from ct.events import EventBus, EventType, InMemoryEventLog, NullEventBus
from ct.events.replay import rebuild_positions
from ct.events.types import Event, Severity
from ct.execution.broker import PaperBroker
from ct.models import Fill, Order, OrderType, Position, Side
from ct.risk.account import AccountStateBuilder
from ct.risk.engine import RiskLimits
from ct.strategies.base import Context, Strategy

TS = datetime(2026, 7, 1, tzinfo=timezone.utc)


class Quiet(Strategy):
    name = "quiet"
    family = "trend"
    DEFAULTS: dict = {}

    @property
    def warmup(self) -> int:
        return 1

    def on_bar(self, ctx: Context):
        return []


def _fill(order_id: str, symbol: str, side: Side, qty: str, price: str,
          strategy: str = "quiet", ts: datetime | None = None) -> Fill:
    return Fill(order_id=order_id, ts=ts or TS, symbol=symbol, side=side,
                qty=Decimal(qty), price=Decimal(price), fee=Decimal("0.1"),
                is_maker=False, strategy=strategy)


# ── the bus contract: never raises, never blocks ─────────────────────────


class _HostileLog(InMemoryEventLog):
    """A sink that is broken in the two ways a real one breaks: it raises, or
    it is slow. Both must be invisible to the caller."""

    def __init__(self, *, raises: bool = False, delay: float = 0.0) -> None:
        super().__init__()
        self.raises = raises
        self.delay = delay
        self.attempts = 0

    def append(self, event):
        self.attempts += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise OSError("no space left on device")
        return super().append(event)


def test_publish_survives_a_sink_that_raises_on_every_write():
    """A full or dead disk must not propagate into the order path."""
    log = _HostileLog(raises=True)
    bus = EventBus(log, source="chaos")
    for i in range(50):
        bus.publish(Event(EventType.HEARTBEAT, {"i": i}, source="chaos"))
    bus.flush()
    assert log.attempts >= 50, "the bus stopped trying"


def test_publish_does_not_block_on_a_slow_sink():
    """Measured, not asserted by inspection: 40 events through a sink that
    takes 5 ms each must not cost the caller 200 ms."""
    log = _HostileLog(delay=0.005)
    bus = EventBus(log, source="chaos")
    started = time.perf_counter()
    for i in range(40):
        bus.publish(Event(EventType.HEARTBEAT, {"i": i}, source="chaos"))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"publish() blocked for {elapsed:.3f}s on a slow sink"


def test_a_subscriber_that_raises_cannot_stop_the_others():
    bus = EventBus(InMemoryEventLog(), source="chaos")
    seen: list[int] = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(lambda e: seen.append(e.seq))
    for i in range(6):
        bus.publish(Event(EventType.HEARTBEAT, {"i": i}, source="chaos"))
    assert len(seen) == 6, "a broken subscriber starved a healthy one"


def test_throughput_ceiling_is_measured_rather_than_load_tested():
    """PHASE2-AUDIT §7.6 declined load testing and promised this instead: a
    recorded ceiling. This system processes single-digit events per second, so
    the useful fact is the order of magnitude of the headroom."""
    bus = EventBus(InMemoryEventLog(), source="chaos")
    n = 2000
    started = time.perf_counter()
    for i in range(n):
        bus.publish(Event(EventType.HEARTBEAT, {"i": i}, source="chaos"))
    bus.flush()
    rate = n / max(1e-9, time.perf_counter() - started)
    print(f"\n    bus throughput ceiling: {rate:,.0f} events/s")
    # The engine emits ~10 events per cycle at 20 s intervals. Anything above
    # a few hundred a second is four orders of magnitude of headroom; below
    # that, something has started doing I/O on the publish path.
    assert rate > 500, f"only {rate:,.0f} events/s — publish() has grown work"


# ── replay is a genuine third opinion ────────────────────────────────────


def test_replayed_positions_agree_with_the_live_book():
    """§3's invariant. `rebuild_positions` reads ORDER_FILLED events and shares
    no code with `AccountStateBuilder`, so agreement is evidence rather than
    the same arithmetic run twice."""
    from ct.events import order_filled

    bus = EventBus(InMemoryEventLog(), source="chaos")
    acct = AccountStateBuilder(Decimal(100_000), RiskLimits())

    script = [
        ("BTC/USD", Side.BUY, "0.5", "60000"),
        ("ETH/USD", Side.BUY, "4", "3000"),
        ("BTC/USD", Side.BUY, "0.25", "61000"),
        ("BTC/USD", Side.SELL, "0.75", "62000"),   # flat again
        ("ETH/USD", Side.SELL, "1", "3100"),
        ("SOL/USD", Side.BUY, "20", "180"),
    ]
    for i, (symbol, side, qty, price) in enumerate(script):
        fill = _fill(f"o{i}", symbol, side, qty, price)
        acct.apply_fill(fill)
        bus.publish(order_filled(fill.order_id, fill.symbol, fill.side.value,
                                 fill.qty, fill.price, fill.fee, fill.is_maker))
    bus.flush()

    replayed = rebuild_positions(bus.log)
    live: dict[str, Decimal] = {}
    for book in acct.positions.values():
        live[book.symbol] = live.get(book.symbol, Decimal(0)) + book.qty

    for symbol in set(live) | set(replayed):
        assert live.get(symbol, Decimal(0)) == replayed.get(symbol, Decimal(0)), \
            f"{symbol}: live {live.get(symbol)} vs replay {replayed.get(symbol)}"
    assert replayed.get("BTC/USD", Decimal(0)) == 0, "the round trip did not close"


# ── a database that dies underneath a running engine ─────────────────────


class _FlakyDB:
    """Fails every call while `down`, records them while up."""

    def __init__(self) -> None:
        self.down = False
        self.calls = 0

    def _check(self):
        self.calls += 1
        if self.down:
            raise ConnectionError("connection to server was lost")

    async def fetch(self, sql, *a):
        self._check()
        return []

    async def fetchrow(self, sql, *a):
        self._check()
        return None

    async def fetchval(self, sql, *a):
        self._check()
        # None, not 0: `CandleRepository.latest_ts` returns a datetime or None,
        # and a fake that answers with the wrong *type* fails the step it is
        # meant to be exercising — which cost a debugging round here.
        return None

    async def execute(self, sql, *a):
        self._check()
        return "OK"

    async def executemany(self, sql, rows):
        self._check()


def _engine(db=None, **kw) -> TradingEngine:
    return TradingEngine([Allocation(Quiet(), ("BTC/USD",))], PaperBroker(),
                         db=db, initial_equity=Decimal(10_000),
                         bus=NullEventBus(source="chaos"), **kw)


def test_a_cycle_never_raises_when_the_database_is_gone():
    """`run_forever` catches per-cycle exceptions, but a cycle that throws is
    still a cycle that did no protective work. This asserts the cycle itself
    absorbs a dead database."""
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    db.down = True
    asyncio.run(engine.cycle())          # must not raise
    assert engine.cycles == 1


def test_the_engine_recovers_when_the_database_comes_back():
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    db.down = True
    asyncio.run(engine.cycle())
    db.down = False
    before = db.calls
    asyncio.run(engine.cycle())
    assert db.calls > before, "the engine stopped talking to a healthy database"
    assert engine.cycles == 2


def test_a_dead_database_never_produces_a_trade():
    """The failure that matters is not the exception — it is trading blind."""
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    db.down = True
    asyncio.run(engine.cycle())
    assert engine.orders_placed == 0


# ── restart survival ─────────────────────────────────────────────────────


class _StoreStub:
    """The three repositories the engine wires, with the state a restart would
    have to find. Enough to drive `boot()`; the SQL itself is covered against
    real Postgres in tests/test_store.py."""

    def __init__(self, books: dict[str, Position], *, reconciles: bool = True,
                 open_orders: dict[str, str] | None = None) -> None:
        self.books = books
        self.reconciles = reconciles
        self.open_orders = open_orders or {}
        self.risk_events: list[tuple] = []

    # Database surface used by _risk_event and the equity write.
    async def fetchval(self, sql, *a):
        if "INSERT INTO risk_events" in sql:
            self.risk_events.append(a)
        return 1

    async def fetch(self, sql, *a):
        return []

    async def fetchrow(self, sql, *a):
        return None

    async def execute(self, sql, *a):
        return "OK"


def _boot_with(store: _StoreStub) -> TradingEngine:
    """Boot an engine whose repositories are stubs rather than Postgres."""
    engine = _engine(store)

    class _Result:
        def __init__(self, ok):
            self.ok = ok
            self.discrepancies = [] if ok else ["BTC/USD: 0.5 vs 0.4"]

        def summary(self):
            return "clean" if self.ok else "1 discrepancy beyond dust"

    class _Positions:
        async def load(self, account_id):
            return dict(store.books)

        async def save_protective_levels(self, *a):
            return None

    class _Recon:
        async def run(self, account_id, balances=None):
            return _Result(store.reconciles)

    async def _wire():
        engine.candles = None
        engine.positions_repo = _Positions()
        engine.reconciler = _Recon()
        engine.executor = _Exec()
        engine.control = None
        engine.arming = None
        engine.outbox = None

    class _Exec:
        async def reconcile_open_orders(self):
            return dict(store.open_orders)

        async def poll_fills(self, **kw):
            return []

        async def cancel_all_open(self):
            return 0

    engine._wire_repositories = _wire                     # type: ignore[assignment]
    return engine


def test_a_restart_reloads_its_books_before_it_considers_trading():
    """A restart that forgets an open position is the worst outcome in this
    layer — it would re-enter a trade it is already in."""
    book = Position(strategy="quiet", symbol="BTC/USD", qty=Decimal("0.5"),
                    avg_entry=Decimal(60_000))
    engine = _boot_with(_StoreStub({"quiet|BTC/USD": book}))
    assert asyncio.run(engine.boot()) is True
    assert engine.acct.position("quiet", "BTC/USD").qty == Decimal("0.5")


def test_a_restart_re_queries_orders_that_were_in_flight():
    """`client_order_id` exists so a restart never blind-resubmits (§6.3)."""
    store = _StoreStub({}, open_orders={"qd-1": "filled", "qd-2": "open"})
    engine = _boot_with(store)
    assert asyncio.run(engine.boot()) is True


def test_a_reconciliation_mismatch_keeps_the_engine_halted_and_records_it():
    """Disagreeing with the exchange about what you own invalidates every risk
    gate at once, because all of them are computed from a position number."""
    store = _StoreStub({}, reconciles=False)
    engine = _boot_with(store)
    assert asyncio.run(engine.boot()) is False
    assert engine.state.halted and not engine.state.reconciled
    assert not engine.state.may_open
    assert store.risk_events, "a mismatch left no risk_events row"
    assert store.risk_events[0][2] == "RECONCILE_MISMATCH"


def test_an_unreconciled_engine_refuses_to_resume():
    store = _StoreStub({}, reconciles=False)
    engine = _boot_with(store)
    asyncio.run(engine.boot())
    assert engine.state.resume() is False, "resumed while disagreeing with the venue"


# ── duplicate delivery ───────────────────────────────────────────────────


class _DedupingOrders:
    """Models the one thing that matters here: `fills` has a UNIQUE constraint
    on `(order_id, exchange_fill_id)`, so the second delivery inserts nothing."""

    def __init__(self, books: _Books | None = None) -> None:
        self.seen: set[tuple] = set()
        self.applied: list[Fill] = []
        # The real repository applies the fill to `positions` inside the same
        # transaction, which is what makes the book move. A stub that only
        # records the row would leave the executor unable to see any change.
        self.books = books

    async def record_fill(self, account_id, fill, notification=None):
        key = (fill.order_id, fill.symbol, str(fill.qty), str(fill.price))
        if key in self.seen:
            return False
        self.seen.add(key)
        self.applied.append(fill)
        if self.books is not None:
            self.books.acct.apply_fill(fill)
        return True


class _Books:
    def __init__(self) -> None:
        self.acct = AccountStateBuilder(Decimal(100_000), RiskLimits())

    async def load(self, account_id):
        return dict(self.acct.positions)


def test_a_redelivered_fill_moves_the_position_once():
    """Exchanges re-deliver and restarts replay. Applying a fill twice is a
    double-sized position — the failure the idempotency key exists to stop."""
    books = _Books()
    orders = _DedupingOrders(books)
    bus = EventBus(InMemoryEventLog(), source="chaos")
    ex = OrderExecutor(PaperBroker(), orders, books, bus, account_id=1)

    fill = _fill("qd-1", "BTC/USD", Side.BUY, "0.5", "60000")
    assert asyncio.run(ex.record_fill(fill)) is True
    assert asyncio.run(ex.record_fill(fill)) is False, "duplicate was accepted"
    assert len(orders.applied) == 1
    assert ex.duplicate_fills == 1

    opens = [e for e in bus.log.read() if e.type is EventType.POSITION_OPENED]
    assert len(opens) == 1, "a duplicate fill announced a second position"


def test_a_duplicate_is_not_reported_as_an_error():
    """It is a normal outcome. Counting it as a failure would make a healthy
    reconnect look like an incident."""
    books = _Books()
    orders = _DedupingOrders(books)
    bus = EventBus(InMemoryEventLog(), source="chaos")
    ex = OrderExecutor(PaperBroker(), orders, books, bus, account_id=1)
    fill = _fill("qd-1", "BTC/USD", Side.BUY, "0.5", "60000")
    asyncio.run(ex.record_fill(fill))
    asyncio.run(ex.record_fill(fill))
    errors = [e for e in bus.log.read() if e.severity is Severity.CRITICAL]
    assert not errors


# ── the order path outlives its observability ────────────────────────────


def test_losing_the_event_log_does_not_stop_a_fill_being_recorded():
    """§8A's whole point: telemetry is an output. If the log dies, the trade
    still happens and still lands in the database."""
    books = _Books()
    orders = _DedupingOrders(books)
    bus = EventBus(_HostileLog(raises=True), source="chaos")
    ex = OrderExecutor(PaperBroker(), orders, books, bus, account_id=1)
    fill = _fill("qd-9", "ETH/USD", Side.BUY, "2", "3000")
    assert asyncio.run(ex.record_fill(fill)) is True
    assert len(orders.applied) == 1


def test_protective_exits_still_fire_while_the_database_is_down():
    """Stops are the last line. They must not depend on bookkeeping being
    healthy — a database outage is exactly when you want them working."""
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())

    book = Position(strategy="quiet", symbol="BTC/USD", qty=Decimal("1"),
                    avg_entry=Decimal(60_000), stop_price=Decimal(59_000))
    engine.acct.positions["quiet|BTC/USD"] = book

    submitted: list[Order] = []

    class _Exec:
        async def submit(self, order, **kw):
            submitted.append(order)

        async def poll_fills(self, **kw):
            return []

    engine.executor = _Exec()
    db.down = True
    asyncio.run(engine._check_protective_exits({"BTC/USD": Decimal(58_000)},
                                               TS + timedelta(hours=1)))
    assert submitted, "the stop did not fire with the database down"
    assert submitted[0].reduce_only and submitted[0].type is OrderType.MARKET


# ── being blind is a state, not a silence ────────────────────────────────


def test_a_prolonged_outage_halts_the_engine_rather_than_trading_blind():
    """The other half of isolating the cycle steps.

    Once steps are individually fault-tolerant, a cycle can complete while
    knowing nothing — and an engine that cannot read `control_commands` cannot
    be halted from the dashboard or the CLI. So it halts itself. It **halts**
    rather than kills, because exits must keep working.
    """
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    assert not engine.state.halted

    db.down = True
    for _ in range(engine.max_degraded_cycles):
        asyncio.run(engine.cycle())

    assert engine.state.halted, "engine kept opening while blind"
    assert "degraded" in engine.state.halt_reason
    assert engine.state.may_close, "exits were blocked; only the kill switch may"
    assert engine.step_failures > 0
    assert engine.status()["degraded_cycles"] >= engine.max_degraded_cycles


def test_one_bad_cycle_is_not_enough_to_halt():
    """A single blip during a `qd-db` restart should not stop the desk."""
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    db.down = True
    asyncio.run(engine.cycle())
    assert not engine.state.halted
    assert engine.degraded_cycles == 1


def test_recovery_resets_the_degradation_counter():
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    db.down = True
    for _ in range(3):
        asyncio.run(engine.cycle())
    assert engine.degraded_cycles == 3
    db.down = False
    asyncio.run(engine.cycle())
    assert engine.degraded_cycles == 0, "a healthy cycle did not clear the count"
    assert not engine.state.halted


def test_degradation_is_reported_not_merely_logged():
    """`qd status` and the dashboard read `status()`. A cycle that did half its
    work must not look identical to one that did all of it."""
    db = _FlakyDB()
    engine = _engine(db)
    asyncio.run(engine.boot())
    db.down = True
    asyncio.run(engine.cycle())
    status = engine.status()
    assert status["degraded_cycles"] == 1
    assert status["step_failures"] >= 1
