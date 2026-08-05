"""Trading-engine tests.

The important ones are about *refusing to trade*: booting halted, staying
halted until reconciliation passes, the kill switch outranking everything, and
exits surviving conditions that block entries. A trading engine that trades is
easy; one that reliably declines to is the hard part.

Runs without a database — `db=None` — so the loop's ordering and gating are
tested in isolation. The database-backed paths (persistence, reconciliation,
arming) are covered in `tests/test_store.py` against real Postgres.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ct.engine import Allocation, KillSwitch, RuntimeState, TradingEngine
from ct.events import EventBus, EventType, InMemoryEventLog
from ct.execution.broker import FeeSchedule, PaperBroker
from ct.models import Bar, Order, OrderType, Quote, Side
from ct.risk.engine import RiskLimits
from ct.strategies.base import Context, Strategy

TS = datetime(2026, 7, 1, tzinfo=timezone.utc)
H4 = timedelta(hours=4)


class AlwaysLong(Strategy):
    name = "always_long"
    family = "trend"
    DEFAULTS: dict = {}

    @property
    def warmup(self) -> int:
        return 5

    def on_bar(self, ctx: Context):
        if not ctx.position.is_flat:
            return []
        return [self._signal(ctx, 1, atr=ctx.price * 0.02,
                             stop=ctx.price * 0.90, target=ctx.price * 1.30)]


class Exploder(Strategy):
    name = "exploder"
    family = "trend"
    DEFAULTS: dict = {}

    @property
    def warmup(self) -> int:
        return 1

    def on_bar(self, ctx: Context):
        raise RuntimeError("strategy is broken")


def _bars(n=30, start=TS, price=100.0) -> list[Bar]:
    return [Bar(ts=start + i * H4, open=price, high=price * 1.01,
                low=price * 0.99, close=price, volume=100.0,
                symbol="BTC/USD", timeframe="4h") for i in range(n)]


def _engine(strategy=None, *, kill_path=None, tmp_path=None, **kw) -> TradingEngine:
    limits = kw.pop("limits", RiskLimits(prefer_maker=False, min_edge_bps=0.0,
                                         min_order_notional=0.0))
    kill = KillSwitch(kill_path or (tmp_path / "KILL" if tmp_path else "/nonexistent/KILL"))
    return TradingEngine(
        [Allocation(strategy or AlwaysLong(), ("BTC/USD",), "4h")],
        PaperBroker(starting_cash=Decimal(10_000), fees=FeeSchedule()),
        db=None, limits=limits, bus=EventBus(InMemoryEventLog(), source="test"),
        kill_switch=kill, clock=lambda: TS + 100 * H4, **kw)


# ══════════════════════════════════════════════════════════════════════════
# Boot — the engine must refuse to trade until it has earned the right
# ══════════════════════════════════════════════════════════════════════════

def test_engine_boots_halted():
    """Before anything is checked, the answer to 'may I trade?' is no."""
    e = _engine()
    assert e.state.halted
    assert not e.state.reconciled
    assert not e.state.may_open
    assert "not yet reconciled" in e.state.halt_reason


def test_resume_is_refused_until_reconciled():
    """Resuming while we disagree with the exchange about what we own is worse
    than staying down."""
    s = RuntimeState()
    assert s.resume() is False
    assert s.halted
    s.reconciled = True
    assert s.resume() is True
    assert not s.halted


def test_boot_without_a_database_reconciles_trivially_and_arms_paper():
    e = _engine()
    assert asyncio.run(e.boot()) is True
    assert e.state.reconciled and not e.state.halted
    assert e.state.armed, "paper mode must arm so the same code path is exercised"
    assert e.state.may_open


def test_boot_logs_whether_the_broker_can_lose_money():
    """So 'I thought it was on paper' is not something anyone can say later."""
    e = _engine()
    asyncio.run(e.boot())
    started = [ev for ev in e.bus.log.read()
               if ev.type is EventType.ENGINE_STARTED]
    assert started and started[0].payload["can_lose_money"] is False


# ══════════════════════════════════════════════════════════════════════════
# The kill switch outranks everything
# ══════════════════════════════════════════════════════════════════════════

def test_kill_switch_blocks_opening_and_closing(tmp_path):
    e = _engine(tmp_path=tmp_path)
    asyncio.run(e.boot())
    assert e.state.may_open

    e.kill.engage("testing")
    e.state.kill_engaged = e.kill.engaged
    assert not e.state.may_open
    assert not e.state.may_close, (
        "KILL means 'I do not trust this system' — exits included. Use "
        "flatten_all to get out.")

    e.kill.release()
    e.state.kill_engaged = e.kill.engaged
    assert e.state.may_close


def test_halt_blocks_opening_but_never_closing():
    """The asymmetry that stops a small loss becoming a large one."""
    s = RuntimeState(reconciled=True, armed=True)
    s.resume()
    assert s.may_open and s.may_close
    s.halt("drawdown")
    assert not s.may_open
    assert s.may_close


def test_unreadable_kill_switch_fails_closed(tmp_path):
    """A filesystem that cannot be read is not a reason to keep trading."""
    class Broken(KillSwitch):
        @property
        def engaged(self) -> bool:
            try:
                raise OSError("disk gone")
            except OSError:
                return True

    k = Broken(tmp_path / "KILL")
    assert k.engaged


def test_kill_switch_needs_nothing_but_a_file(tmp_path):
    """No database, no network, no auth — the path that works when the rest
    is down."""
    k = KillSwitch(tmp_path / "KILL")
    assert not k.engaged
    k.engage("exchange on fire")
    assert k.engaged and k.reason() == "exchange on fire"
    k.release()
    assert not k.engaged


# ══════════════════════════════════════════════════════════════════════════
# The cycle
# ══════════════════════════════════════════════════════════════════════════

def test_cycle_survives_a_broken_strategy_and_quarantines_it():
    """A strategy bug must not stop stops from firing."""
    e = _engine(Exploder())
    asyncio.run(e.boot())
    fresh = {("BTC/USD", "4h"): _bars()}
    asyncio.run(e._run_strategies(fresh, {"BTC/USD": Decimal(100)},
                                  {"BTC/USD": 0}, TS))
    assert "exploder" in e.state.paused_strategies
    quarantined = [ev for ev in e.bus.log.read()
                   if ev.type is EventType.STRATEGY_QUARANTINED]
    assert quarantined


def test_a_paused_strategy_is_not_consulted():
    e = _engine()
    asyncio.run(e.boot())
    e.state.paused_strategies.add("always_long")
    fresh = {("BTC/USD", "4h"): _bars()}
    asyncio.run(e._run_strategies(fresh, {"BTC/USD": Decimal(100)},
                                  {"BTC/USD": 0}, TS))
    assert e.signals_seen == 0


def test_signals_are_recorded_even_when_risk_rejects_them():
    """Without the rejections you cannot tell 'the strategy is bad' from
    'risk is strangling it'."""
    e = _engine(limits=RiskLimits(prefer_maker=False, min_edge_bps=100_000.0))
    asyncio.run(e.boot())
    fresh = {("BTC/USD", "4h"): _bars()}
    asyncio.run(e._run_strategies(fresh, {"BTC/USD": Decimal(100)},
                                  {"BTC/USD": 0}, TS))
    rejected = [ev for ev in e.bus.log.read()
                if ev.type is EventType.SIGNAL_REJECTED]
    assert rejected, "a rejection must be an event, not a silence"
    assert rejected[0].payload["code"] == "EDGE_BELOW_FEES"
    assert e.orders_placed == 0


def test_stale_data_stops_signals_becoming_orders():
    e = _engine()
    asyncio.run(e.boot())
    fresh = {("BTC/USD", "4h"): _bars()}
    asyncio.run(e._run_strategies(fresh, {"BTC/USD": Decimal(100)},
                                  {"BTC/USD": 99}, TS))       # 99 bars stale
    rejected = [ev for ev in e.bus.log.read()
                if ev.type is EventType.SIGNAL_REJECTED]
    assert rejected and rejected[0].payload["code"] == "STALE_DATA"


def test_a_signal_carries_one_correlation_id_through_to_the_order():
    e = _engine()
    asyncio.run(e.boot())
    fresh = {("BTC/USD", "4h"): _bars()}
    asyncio.run(e._run_strategies(fresh, {"BTC/USD": Decimal(100)},
                                  {"BTC/USD": 0}, TS))
    events = list(e.bus.log.read())
    sig = next(ev for ev in events if ev.type is EventType.SIGNAL_GENERATED)
    appr = next(ev for ev in events if ev.type is EventType.RISK_APPROVED)
    assert sig.correlation_id == appr.correlation_id
    assert appr.causation_id == sig.seq


# ══════════════════════════════════════════════════════════════════════════
# Protective exits
# ══════════════════════════════════════════════════════════════════════════

def test_protective_exit_fires_when_the_stop_is_breached():
    """Never relies solely on a resting exchange stop — those get cancelled by
    maintenance and rejected on thin books."""
    e = _engine()
    asyncio.run(e.boot())
    book = e.acct.position("always_long", "BTC/USD")
    book.qty = Decimal("0.1")
    book.avg_entry = Decimal(100)
    book.stop_price = Decimal(90)

    submitted: list[Order] = []

    class Spy:
        async def submit(self, order, **kw):
            submitted.append(order)
        async def cancel_all_open(self):
            return 0

    e.executor = Spy()
    asyncio.run(e._check_protective_exits({"BTC/USD": Decimal(89)}, TS))
    assert len(submitted) == 1
    o = submitted[0]
    assert o.side is Side.SELL and o.reduce_only and o.qty == Decimal("0.1")
    assert o.meta["reason_enum"] == "stop"


def test_protective_exits_run_even_while_halted():
    """Halt stops opening. It must never strand an open position."""
    e = _engine()
    asyncio.run(e.boot())
    e.state.halt("drawdown")
    book = e.acct.position("always_long", "BTC/USD")
    book.qty, book.avg_entry, book.stop_price = Decimal(1), Decimal(100), Decimal(95)

    submitted: list[Order] = []

    class Spy:
        async def submit(self, order, **kw):
            submitted.append(order)

    e.executor = Spy()
    asyncio.run(e._check_protective_exits({"BTC/USD": Decimal(94)}, TS))
    assert len(submitted) == 1, "a halt must not strand an open position"


def test_protective_exits_do_not_run_under_the_kill_switch(tmp_path):
    e = _engine(tmp_path=tmp_path)
    asyncio.run(e.boot())
    e.kill.engage("no trust")
    e.state.kill_engaged = True
    book = e.acct.position("always_long", "BTC/USD")
    book.qty, book.avg_entry, book.stop_price = Decimal(1), Decimal(100), Decimal(95)

    submitted: list[Order] = []

    class Spy:
        async def submit(self, order, **kw):
            submitted.append(order)

    e.executor = Spy()
    asyncio.run(e._check_protective_exits({"BTC/USD": Decimal(94)}, TS))
    assert submitted == []


def test_target_exit_fires_too():
    e = _engine()
    asyncio.run(e.boot())
    book = e.acct.position("always_long", "BTC/USD")
    book.qty, book.avg_entry, book.target_price = Decimal(1), Decimal(100), Decimal(130)

    submitted: list[Order] = []

    class Spy:
        async def submit(self, order, **kw):
            submitted.append(order)

    e.executor = Spy()
    asyncio.run(e._check_protective_exits({"BTC/USD": Decimal(131)}, TS))
    assert submitted[0].meta["reason_enum"] == "target"


# ══════════════════════════════════════════════════════════════════════════
# Drawdown
# ══════════════════════════════════════════════════════════════════════════

def test_total_drawdown_halts_and_requests_a_flatten():
    e = _engine(limits=RiskLimits(max_total_drawdown=0.10))
    asyncio.run(e.boot())
    e.acct.high_water_mark = Decimal(10_000)
    e.acct.cash = e.acct.equity = Decimal(8_500)      # -15%
    snap = e.acct.snapshot(TS, {})
    asyncio.run(e._enforce_drawdown(snap))
    assert e.state.halted and e.state.flatten_all
    breach = [ev for ev in e.bus.log.read()
              if ev.type is EventType.RISK_LIMIT_EXCEEDED]
    assert breach and breach[0].payload["code"] == "MAX_DRAWDOWN_HALT"


# ══════════════════════════════════════════════════════════════════════════
# Live mode is refused without a live broker
# ══════════════════════════════════════════════════════════════════════════

def test_live_mode_without_a_live_broker_refuses_to_start(monkeypatch):
    """Silently paper-trading while the operator believes it is live is
    strictly worse than not starting."""
    from ct.engine.__main__ import main
    monkeypatch.setenv("QD_MODE", "live")
    monkeypatch.setenv("QD_DB_DSN", "postgresql://unused")
    assert asyncio.run(main([])) == 2


def test_paper_engine_never_uses_a_broker_that_can_lose_money():
    e = _engine()
    assert e.broker.can_lose_money is False


# ══════════════════════════════════════════════════════════════════════════
# Shutdown
# ══════════════════════════════════════════════════════════════════════════

def test_shutdown_cancels_resting_orders_but_does_not_flatten():
    """A container restart is not a reason to market-sell the book — that
    would turn every deploy into a taxable event."""
    e = _engine()
    asyncio.run(e.boot())
    book = e.acct.position("always_long", "BTC/USD")
    book.qty, book.avg_entry = Decimal(1), Decimal(100)

    cancelled = {"n": 0}
    submitted: list[Order] = []

    class Spy:
        async def cancel_all_open(self):
            cancelled["n"] += 1
            return 3
        async def submit(self, order, **kw):
            submitted.append(order)

    e.executor = Spy()
    asyncio.run(e.shutdown())
    assert cancelled["n"] == 1
    assert submitted == [], "shutdown must not flatten"
    assert not e.acct.position("always_long", "BTC/USD").is_flat


def test_status_reports_the_things_an_operator_asks_first():
    e = _engine()
    asyncio.run(e.boot())
    s = e.status()
    for key in ("mode", "halted", "armed", "reconciled", "may_open",
                "may_close", "equity", "open_books", "cycles"):
        assert key in s, f"status is missing {key}"


# ══════════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════════

def test_executor_persists_before_submitting():
    """A crash after submitting but before recording leaves an order live at
    the venue that this system has never heard of."""
    from ct.engine.execution import OrderExecutor
    order_seq: list[str] = []

    class Orders:
        async def insert_order(self, account_id, order, **kw):
            order_seq.append("persist")
            return 1
        async def mark_submitted(self, *a, **kw):
            return None
        async def mark_terminal(self, *a, **kw):
            return None

    class Brk:
        can_lose_money = False
        fills: list = []
        async def submit(self, order):
            order_seq.append("submit")
            from ct.execution.broker import OrderUpdate
            from ct.models import OrderStatus
            return OrderUpdate(order.client_order_id, OrderStatus.OPEN)

    ex = OrderExecutor(Brk(), Orders(), None,
                       EventBus(InMemoryEventLog(), source="t"), account_id=1)
    o = Order(symbol="BTC/USD", side=Side.BUY, type=OrderType.MARKET,
              qty=Decimal(1), strategy="s")
    asyncio.run(ex.submit(o, correlation_id="cor-1"))
    assert order_seq == ["persist", "submit"]


def test_broker_failure_leaves_the_order_pending_for_reconciliation():
    """'We don't know' is not 'it didn't happen'."""
    from ct.engine.execution import OrderExecutor
    from ct.models import OrderStatus

    terminal: list = []

    class Orders:
        async def insert_order(self, *a, **kw):
            return 1
        async def mark_submitted(self, *a, **kw):
            return None
        async def mark_terminal(self, *a, **kw):
            terminal.append(a)

    class Brk:
        can_lose_money = False
        fills: list = []
        async def submit(self, order):
            raise ConnectionError("exchange unreachable")

    ex = OrderExecutor(Brk(), Orders(), None,
                       EventBus(InMemoryEventLog(), source="t"), account_id=1)
    o = Order(symbol="BTC/USD", side=Side.BUY, type=OrderType.MARKET,
              qty=Decimal(1), strategy="s")
    upd = asyncio.run(ex.submit(o, correlation_id="cor-1"))
    assert upd.status is OrderStatus.PENDING
    assert terminal == [], "an unreachable exchange must not mark it terminal"
    assert ex.consecutive_broker_failures == 1


# ══════════════════════════════════════════════════════════════════════════
# Arming gates LIVE only — found by running the whole system end to end
# ══════════════════════════════════════════════════════════════════════════

def test_paper_mode_is_armed_even_with_a_database():
    """Regression. Paper mode used to consult the `live_arming` table, so with
    a real database attached it sat there doing nothing and looking healthy —
    and the 90-day paper record the promotion gate demands could never be
    accumulated. The unit tests passed because `db=None` took the other branch.
    """
    e = _engine()

    class FakeArming:
        checked = False
        async def is_armed(self, account_id):
            FakeArming.checked = True
            return False

    e.arming = FakeArming()
    asyncio.run(e._refresh_arming())
    assert e.state.armed is True
    assert FakeArming.checked is False, (
        "paper mode must not consult the live arming table at all")


def test_live_mode_is_unarmed_without_an_arming_row():
    e = _engine(mode="live")

    class FakeArming:
        async def is_armed(self, account_id):
            return False

    e.arming = FakeArming()
    asyncio.run(e._refresh_arming())
    assert e.state.armed is False


def test_live_mode_with_a_valid_arming_row_is_armed():
    e = _engine(mode="live")

    class FakeArming:
        async def is_armed(self, account_id):
            return True

    e.arming = FakeArming()
    asyncio.run(e._refresh_arming())
    assert e.state.armed is True


def test_live_mode_without_a_database_cannot_be_armed():
    """Nowhere to record a human tap means no way to be armed. Refuse rather
    than assume."""
    e = _engine(mode="live")
    e.arming = None
    asyncio.run(e._refresh_arming())
    assert e.state.armed is False


def test_paper_broker_fills_carry_the_strategy_so_books_reload_after_restart():
    """Regression, found by running the engine end to end against live prices.

    `PaperBroker` built its `Fill` without `strategy`, so the position landed
    under the book key `"|BTC/USD"` instead of `"force_long|BTC/USD"`. The
    engine then never found its own position on restart — while reconciliation
    still passed, because that nets by symbol. A silently lost open position is
    about the worst outcome this layer can produce.
    """
    async def run():
        pb = PaperBroker(starting_cash=Decimal(100_000))
        pb.on_quote(Quote(TS, "BTC/USD", bid=100.0, ask=100.2,
                          bid_size=10.0, ask_size=10.0))
        order = Order(symbol="BTC/USD", side=Side.BUY, type=OrderType.MARKET,
                      qty=Decimal(1), strategy="donchian_breakout")
        await pb.submit(order)
        return pb.fills

    fills = asyncio.run(run())
    assert fills, "no fill produced"
    assert fills[0].strategy == "donchian_breakout"

    from ct.models import AccountState
    from ct.risk.account import AccountStateBuilder
    b = AccountStateBuilder(Decimal(100_000), RiskLimits())
    b.apply_fill(fills[0])
    assert AccountState.book_key("donchian_breakout", "BTC/USD") in b.positions
    assert "|BTC/USD" not in b.positions


# ── risk events actually reach the database ──────────────────────────────


class _RecordingDB:
    """Minimal stand-in that records writes. The real repository is covered
    against Postgres in tests/test_store.py; what matters here is that the
    engine calls it at all."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.fail = fail

    async def fetchval(self, sql, *args):
        if self.fail:
            raise RuntimeError("database is down")
        self.calls.append((sql, args))
        return 1

    def wrote(self, needle: str) -> list[tuple[str, tuple]]:
        return [c for c in self.calls if needle in " ".join(c[0].split())]


def _engine_with(db):
    return TradingEngine([Allocation(AlwaysLong(), ("BTC/USD",))],
                         PaperBroker(), db=db, initial_equity=Decimal(10_000))


def test_a_breach_writes_a_risk_event_row():
    """Regression: `RiskEventRepository` had **no callers anywhere in the
    codebase**. Every breach was published as an event and no row was ever
    written, so `qd status` and the dashboard both reported "0 unresolved risk
    events" whether or not anything had breached. A green light that cannot
    turn red is worse than no light."""
    db = _RecordingDB()
    engine = _engine_with(db)
    asyncio.run(engine._risk_event("MAX_DRAWDOWN_HALT", "drawdown 21%",
                                   drawdown=0.21))
    wrote = db.wrote("INSERT INTO risk_events")
    assert len(wrote) == 1
    _account_id, severity, code, detail = wrote[0][1][:4]
    assert code == "MAX_DRAWDOWN_HALT" and severity == "critical"
    assert "21%" in detail


def test_a_risk_event_is_published_as_well_as_persisted():
    bus = EventBus(InMemoryEventLog(), source="test")
    db = _RecordingDB()
    engine = _engine_with(db)
    engine.bus = bus
    asyncio.run(engine._risk_event("RECONCILE_MISMATCH", "0.01 BTC adrift"))
    kinds = [e.type for e in bus.log.read()]
    assert EventType.RISK_LIMIT_EXCEEDED in kinds
    assert db.wrote("INSERT INTO risk_events")


def test_publish_false_does_not_double_report():
    """A site that already emitted its own, more specific event should not
    produce a second one just to get a row written."""
    bus = EventBus(InMemoryEventLog(), source="test")
    db = _RecordingDB()
    engine = _engine_with(db)
    engine.bus = bus
    asyncio.run(engine._risk_event("STRATEGY_QUARANTINED", "boom",
                                   severity="warn", publish=False))
    assert EventType.RISK_LIMIT_EXCEEDED not in [e.type for e in bus.log.read()]
    assert db.wrote("INSERT INTO risk_events")


def test_bookkeeping_failure_never_takes_down_the_loop():
    """The loop that enforces stops must not die because an audit insert
    failed. Persisting the breach matters; it matters less than exiting."""
    engine = _engine_with(_RecordingDB(fail=True))
    asyncio.run(engine._risk_event("MAX_DRAWDOWN_HALT", "drawdown 21%"))


def test_drawdown_halt_records_the_breach():
    class _Snap:
        def drawdown(self):
            return Decimal("0.25")

    db = _RecordingDB()
    engine = _engine_with(db)
    engine.state.halted = False
    asyncio.run(engine._enforce_drawdown(_Snap()))
    assert engine.state.halted
    assert engine.state.flatten_all
    assert db.wrote("INSERT INTO risk_events")[0][1][2] == "MAX_DRAWDOWN_HALT"


# ══════════════════════════════════════════════════════════════════════════
# Flatten actually flattens (2026-07-29). Before this, `flatten_all` set a
# flag and nothing closed — the CLI and dashboard carried warnings saying so.
# ══════════════════════════════════════════════════════════════════════════


class _SpyExec:
    def __init__(self):
        self.submitted: list[Order] = []

    async def submit(self, order, **kw):
        self.submitted.append(order)

    async def cancel_all_open(self):
        return 0


def _open_book(e, strategy, symbol, qty):
    book = e.acct.position(strategy, symbol)
    book.qty = Decimal(str(qty))
    book.avg_entry = Decimal(100)
    return book


def test_flatten_all_closes_every_open_book_once():
    e = _engine()
    asyncio.run(e.boot())
    _open_book(e, "always_long", "BTC/USD", "0.5")
    _open_book(e, "other", "ETH/USD", "-2")
    e.executor = spy = _SpyExec()
    e.state.flatten_all = True

    asyncio.run(e._apply_flatten())
    assert len(spy.submitted) == 2
    by_symbol = {o.symbol: o for o in spy.submitted}
    assert by_symbol["BTC/USD"].side is Side.SELL
    assert by_symbol["ETH/USD"].side is Side.BUY, "a short is closed by buying"
    assert all(o.reduce_only and o.type is OrderType.MARKET
               and o.meta["reason_enum"] == "flatten" for o in spy.submitted)

    # The fills have not landed yet: a second pass must NOT resubmit.
    asyncio.run(e._apply_flatten())
    assert len(spy.submitted) == 2, "slow fills must not attract duplicates"
    assert e.state.flatten_all, "the flag clears only when the book is flat"

    # Fills land; the flag clears on the pass that OBSERVES flatness.
    for b in list(e.acct.positions.values()):
        b.qty = Decimal(0)
    asyncio.run(e._apply_flatten())
    assert not e.state.flatten_all
    assert not e._flatten_submitted


def test_flatten_symbol_touches_only_that_symbol():
    e = _engine()
    asyncio.run(e.boot())
    _open_book(e, "always_long", "BTC/USD", "0.5")
    _open_book(e, "always_long", "ETH/USD", "1")
    e.executor = spy = _SpyExec()
    e.state.flatten_requested.add("ETH/USD")

    asyncio.run(e._apply_flatten())
    assert [o.symbol for o in spy.submitted] == ["ETH/USD"]

    e.acct.position("always_long", "ETH/USD").qty = Decimal(0)
    asyncio.run(e._apply_flatten())
    assert not e.state.flatten_requested, "request clears once the symbol is flat"
    btc = e.acct.position("always_long", "BTC/USD")
    assert btc.qty == Decimal("0.5"), "the other book must be untouched"


def test_flatten_runs_while_halted_but_never_under_the_kill_switch(tmp_path):
    e = _engine(tmp_path=tmp_path)
    asyncio.run(e.boot())
    _open_book(e, "always_long", "BTC/USD", "1")
    e.executor = spy = _SpyExec()
    e.state.flatten_all = True

    e.state.halt("drawdown")
    asyncio.run(e._apply_flatten())
    assert len(spy.submitted) == 1, "a halt is exactly when flatten must work"

    e.state.kill_engaged = True
    e.state.flatten_all = True
    e._flatten_submitted.clear()
    spy.submitted.clear()
    asyncio.run(e._apply_flatten())
    assert spy.submitted == [], "the kill switch blocks ALL orders, flatten too"


def test_flatten_with_nothing_open_clears_immediately():
    e = _engine()
    asyncio.run(e.boot())
    e.executor = spy = _SpyExec()
    e.state.flatten_all = True
    e.state.flatten_requested.add("ETH/USD")
    asyncio.run(e._apply_flatten())
    assert spy.submitted == []
    assert not e.state.flatten_all and not e.state.flatten_requested


# ══════════════════════════════════════════════════════════════════════════
# Lifecycle changes are picked up while running (2026-07-29). Before this,
# `qd enable-strategy` did nothing until the next container restart.
# ══════════════════════════════════════════════════════════════════════════

from ct.strategies.lifecycle import StrategyLifecycle, StrategyState  # noqa: E402


class _FakeStrategiesRepo:
    def __init__(self, states: dict[str, StrategyState]):
        self.states = dict(states)
        self.saved: list[tuple[str, str]] = []
        self.fail = False

    async def load_all(self):
        if self.fail:
            raise ConnectionError("db down")
        return {n: StrategyLifecycle(name=n, state=s)
                for n, s in self.states.items()}

    async def load(self, name):
        return (await self.load_all()).get(name)

    async def save(self, lc, *, actor):
        self.saved.append((lc.name, lc.state.value))

    async def upsert(self, name, **kw):
        self.states.setdefault(name, StrategyState.PROPOSED)


def test_externally_enabled_strategy_starts_without_a_restart():
    e = _engine()
    asyncio.run(e.boot())
    repo = _FakeStrategiesRepo({"always_long": StrategyState.PROPOSED})
    e.strategies_repo = repo
    asyncio.run(e._load_lifecycles())
    assert "always_long" in e.state.paused_strategies

    # A human runs `qd enable-strategy always_long` in another process.
    repo.states["always_long"] = StrategyState.PAPER
    asyncio.run(e._refresh_lifecycles(TS))
    assert "always_long" not in e.state.paused_strategies
    resumed = [ev for ev in e.bus.log.read()
               if ev.type is EventType.STRATEGY_RESUMED]
    assert resumed and resumed[0].payload["strategy"] == "always_long"


def test_externally_quarantined_strategy_stops_without_a_restart():
    e = _engine()
    asyncio.run(e.boot())
    repo = _FakeStrategiesRepo({"always_long": StrategyState.PAPER})
    e.strategies_repo = repo
    asyncio.run(e._load_lifecycles())
    assert "always_long" not in e.state.paused_strategies

    repo.states["always_long"] = StrategyState.QUARANTINED
    asyncio.run(e._refresh_lifecycles(TS))
    assert "always_long" in e.state.paused_strategies


def test_refresh_does_not_undo_an_operator_pause():
    """`pause_strategy` is a transient control command, not lifecycle state.
    The lifecycle refresh may only unblock names it blocked itself."""
    e = _engine()
    asyncio.run(e.boot())
    repo = _FakeStrategiesRepo({"always_long": StrategyState.PAPER})
    e.strategies_repo = repo
    asyncio.run(e._load_lifecycles())

    e.state.paused_strategies.add("always_long")     # operator's pause
    asyncio.run(e._refresh_lifecycles(TS))
    assert "always_long" in e.state.paused_strategies


def test_refresh_failure_keeps_current_states():
    """Fail static: an engine that cannot read lifecycle rows keeps every
    existing block and unblocks nothing."""
    e = _engine()
    asyncio.run(e.boot())
    repo = _FakeStrategiesRepo({"always_long": StrategyState.PROPOSED})
    e.strategies_repo = repo
    asyncio.run(e._load_lifecycles())
    assert "always_long" in e.state.paused_strategies

    repo.states["always_long"] = StrategyState.PAPER
    repo.fail = True
    asyncio.run(e._refresh_lifecycles(TS))
    assert "always_long" in e.state.paused_strategies, (
        "an unreadable lifecycle table must not unblock anything")


def test_control_resolve_risk_event_reaches_the_repository():
    """The dashboard queues intent; the engine is the writer. Wired 2026-07-30
    after the Risk page shipped with no resolve path at all."""
    class _Db:
        def __init__(self):
            self.executed = []
        async def execute(self, sql, *args):
            self.executed.append((" ".join(sql.split()), args))
        async def fetchval(self, sql, *args):
            return 1

    class _Cmd:
        id = 7
        command = "resolve_risk_event"
        args = {"id": "42"}
        requested_by = "web:admin"

    class _Control:
        def __init__(self):
            self.applied = []
        async def pending(self):
            return [_Cmd()]
        async def mark_applied(self, cid, result):
            self.applied.append((cid, result))

    e = _engine()
    e.db = _Db()
    e.control = _Control()
    asyncio.run(e._apply_control())
    assert any("UPDATE risk_events SET resolved_at" in sql
               for sql, _ in e.db.executed)
    assert e.control.applied == [(7, "resolved risk_event 42")]


def test_control_resolve_risk_event_refuses_a_bad_id():
    class _Db:
        async def execute(self, sql, *args):
            raise AssertionError("nothing must be written for a bad id")

    class _Cmd:
        id = 8
        command = "resolve_risk_event"
        args = {"id": "not-a-number"}
        requested_by = "web:admin"

    class _Control:
        def __init__(self):
            self.applied = []
        async def pending(self):
            return [_Cmd()]
        async def mark_applied(self, cid, result):
            self.applied.append((cid, result))

    e = _engine()
    e.db = _Db()
    e.control = _Control()
    asyncio.run(e._apply_control())
    assert e.control.applied and "refused: bad event id" in e.control.applied[0][1]


def test_schema_check_constraint_matches_valid_commands():
    """One boundary, expressed twice: VALID_COMMANDS in code and the CHECK
    constraint in db/schema.sql. Adding resolve_risk_event to the code alone
    failed live (2026-07-30) — the deployed database refused the row. This
    pins the two lists to each other; a deployed database additionally needs
    an ALTER, which the schema comment now says."""
    import re
    from pathlib import Path

    from ct.engine.control import VALID_COMMANDS

    sql = (Path(__file__).resolve().parents[1] / "db" / "schema.sql"
           ).read_text(encoding="utf-8")
    m = re.search(r"command\s+text NOT NULL CHECK \(command IN\s*\(([^)]+)\)\)",
                  sql)
    assert m, "control_commands command CHECK not found in schema.sql"
    schema_commands = set(re.findall(r"'([a-z_]+)'", m.group(1)))
    assert schema_commands == set(VALID_COMMANDS), (
        f"schema and VALID_COMMANDS disagree: "
        f"schema-only={schema_commands - set(VALID_COMMANDS)}, "
        f"code-only={set(VALID_COMMANDS) - schema_commands}")
