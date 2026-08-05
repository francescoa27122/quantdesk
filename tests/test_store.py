"""Persistence tests — against a REAL Postgres, or skipped.

Set `QD_TEST_DSN` to run these. There is no in-memory substitute: the things
being tested are `ON CONFLICT` semantics, `FOR UPDATE SKIP LOCKED`, CHECK
constraints, recursive CTEs and a plpgsql function. A fake database would test
the fake.

How these were actually run (the NAS blocks both inbound 55433 and SSH
forwarding, so the tests execute in a container beside the database):

    docker run -d --name qdtest -p 55433:5432 -e POSTGRES_USER=qd \\
        -e POSTGRES_PASSWORD=qdtest -e POSTGRES_DB=quantdesk \\
        timescale/timescaledb:2.17.2-pg16
    docker run --rm --network host -v /tmp/qdsrc:/src -w /src \\
        -e QD_TEST_DSN=postgresql://qd:qdtest@127.0.0.1:55433/quantdesk \\
        python:3.12-slim sh -c "pip install -q asyncpg pytest pytest-asyncio \\
        numpy && python -m pytest tests/test_store.py -q"
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ct.events import EventBus, InMemoryEventLog, JsonlEventLog, order_filled
from ct.models import Fill, Order, OrderStatus, OrderType, Side
from ct.store import (
    ArmingRepository,
    Database,
    IntegrityError,
    NotificationOutbox,
    OrderRepository,
    PositionRepository,
    PostgresEventSink,
    ReconciliationService,
    RiskEventRepository,
    SignalRepository,
    TeeEventLog,
    backfill_from_jsonl,
    envelope_hash,
)

DSN = os.environ.get("QD_TEST_DSN")
pytestmark = [
    pytest.mark.skipif(not DSN, reason="QD_TEST_DSN not set; needs a real Postgres"),
    pytest.mark.asyncio,
]

TS = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
SCHEMA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")


# Every data table, in one statement. TRUNCATE takes a lighter lock than
# DROP SCHEMA, which deadlocks against TimescaleDB's background workers holding
# AccessShareLock on the hypertable catalog — an intermittent failure that looks
# like flakiness and is really lock ordering. It is also ~10x faster than
# rebuilding the schema for every test.
_TRUNCATE = """
TRUNCATE events, orders, fills, positions, trades, equity_curve, signals,
         risk_events, control_commands, live_arming, notifications, audit_log,
         candles, book_snapshots, funding_rates, regimes, model_registry,
         promotion_status, strategies, risk_config, backtests, accounts
RESTART IDENTITY CASCADE
"""
_SEED = """
INSERT INTO accounts (name, mode, exchange, quote_ccy, initial_equity,
                      high_water_mark)
VALUES ('paper-main', 'paper', 'kraken', 'USD', 10000, 10000)
"""


@pytest.fixture
async def db():
    """A database with the schema present and every table empty.

    The schema is created once (whichever test runs first) and then reused;
    isolation comes from TRUNCATE. Recreate the container to pick up a changed
    `schema.sql` — the tests deliberately do not drop it, see `_TRUNCATE`.
    """
    d = Database(DSN, max_size=4)
    await d.connect()
    await d.bootstrap(SCHEMA)          # no-op once the spine exists
    await d.execute(_TRUNCATE)
    await d.execute(_SEED)
    yield d
    await d.close()


@pytest.fixture
async def account(db):
    return int(await db.fetchval(
        "SELECT id FROM accounts WHERE name = 'paper-main'"))


def _order(symbol="BTC/USD", qty="0.1", strategy="ema_cross",
           side=Side.BUY, cid=None) -> Order:
    kw = {"client_order_id": cid} if cid else {}
    return Order(symbol=symbol, side=side, type=OrderType.MARKET,
                 qty=Decimal(qty), strategy=strategy,
                 attached_stop=Decimal("54000"), **kw)


def _fill(order: Order, qty=None, price="60000", fee="24",
          exch_id="x1", ts=TS) -> Fill:
    return Fill(order_id=order.client_order_id, ts=ts, symbol=order.symbol,
                side=order.side, qty=Decimal(qty or order.qty),
                price=Decimal(price), fee=Decimal(fee), is_maker=False,
                exchange_fill_id=exch_id, strategy=order.strategy)


# ══════════════════════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════════════════════

async def test_bootstrap_creates_the_schema_and_is_idempotent(db):
    tables = await db.table_names()
    for required in ("orders", "fills", "positions", "events", "live_arming",
                     "notifications", "signals", "risk_events"):
        assert required in tables, f"{required} missing after bootstrap"
    # Running it again must be a no-op, not an error — containers restart.
    assert await db.bootstrap(SCHEMA) is False


async def test_health_and_stats(db):
    assert await db.healthy()
    s = await db.stats()
    assert s["connected"] and s["pool_size"] >= 1


# ══════════════════════════════════════════════════════════════════════════
# Idempotency — the property that makes restart-safety possible
# ══════════════════════════════════════════════════════════════════════════

async def test_resubmitting_an_order_is_a_no_op_not_a_second_position(db, account):
    repo = OrderRepository(db)
    o = _order()
    first = await repo.insert_order(account, o)
    second = await repo.insert_order(account, o)      # restart replay
    assert first == second
    assert await db.fetchval("SELECT count(*) FROM orders") == 1


async def test_duplicate_fill_delivery_is_ignored(db, account):
    repo = OrderRepository(db)
    o = _order()
    await repo.insert_order(account, o)
    f = _fill(o)
    assert await repo.record_fill(account, f) is True
    # Exchanges DO re-deliver. This must be a normal outcome, not an error.
    assert await repo.record_fill(account, f) is False
    assert await db.fetchval("SELECT count(*) FROM fills") == 1
    assert await db.fetchval(
        "SELECT qty FROM positions WHERE symbol='BTC/USD'") == Decimal("0.1")


async def test_fill_for_an_unknown_order_is_refused(db, account):
    repo = OrderRepository(db)
    with pytest.raises(ValueError, match="unknown order"):
        await repo.record_fill(account, _fill(_order()))


async def test_filled_qty_is_derived_not_incremented(db, account):
    """An increment that runs twice is a phantom position; a recomputation
    cannot drift."""
    repo = OrderRepository(db)
    o = _order(qty="1.0")
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o, qty="0.4", exch_id="a"))
    await repo.record_fill(account, _fill(o, qty="0.6", price="61000", exch_id="b"))
    row = await db.fetchrow("SELECT * FROM orders WHERE client_order_id=$1",
                            o.client_order_id)
    assert row["filled_qty"] == Decimal("1.0")
    assert row["status"] == "filled"
    assert row["avg_fill_price"] == pytest.approx(Decimal("60600"))


# ══════════════════════════════════════════════════════════════════════════
# Books — positions are per (strategy, symbol)
# ══════════════════════════════════════════════════════════════════════════

async def test_two_strategies_keep_independent_books_on_one_symbol(db, account):
    repo = OrderRepository(db)
    a = _order(strategy="alpha", qty="0.5")
    b = _order(strategy="beta", qty="0.2")
    await repo.insert_order(account, a)
    await repo.insert_order(account, b)
    await repo.record_fill(account, _fill(a, exch_id="a"))
    await repo.record_fill(account, _fill(b, exch_id="b"))

    books = await PositionRepository(db).load(account)
    assert books["alpha|BTC/USD"].qty == Decimal("0.5")
    assert books["beta|BTC/USD"].qty == Decimal("0.2")

    # Alpha exits entirely; beta must be untouched.
    close = _order(strategy="alpha", qty="0.5", side=Side.SELL)
    await repo.insert_order(account, close)
    await repo.record_fill(account, _fill(close, price="61000", exch_id="c"))
    books = await PositionRepository(db).load(account)
    assert books["alpha|BTC/USD"].is_flat
    assert books["beta|BTC/USD"].qty == Decimal("0.2")


async def test_symbol_view_nets_across_books(db, account):
    repo = OrderRepository(db)
    for strat in ("alpha", "beta"):
        o = _order(strategy=strat, qty="0.5")
        await repo.insert_order(account, o)
        await repo.record_fill(account, _fill(o, exch_id=strat))
    net = await PositionRepository(db).net_by_symbol(account)
    assert net["BTC/USD"] == Decimal("1.0")
    row = await db.fetchrow(
        "SELECT * FROM positions_by_symbol WHERE symbol='BTC/USD'")
    assert row["open_books"] == 2


async def test_position_flip_resets_entry_price_in_sql(db, account):
    """The case naive implementations corrupt — and the SQL path must match
    `Position.apply_fill` exactly."""
    repo = OrderRepository(db)
    buy = _order(qty="1")
    await repo.insert_order(account, buy)
    await repo.record_fill(account, _fill(buy, price="60000", exch_id="a"))
    sell = _order(qty="3", side=Side.SELL)
    await repo.insert_order(account, sell)
    await repo.record_fill(account, _fill(sell, price="65000", exch_id="b"))
    row = await db.fetchrow("SELECT * FROM positions WHERE symbol='BTC/USD'")
    assert row["qty"] == Decimal("-2")
    assert row["avg_entry"] == Decimal("65000")


async def test_protective_levels_survive_a_restart(db, account):
    """Losing a stop across a restart is the most expensive thing this layer
    can get wrong: the position survives, its protection does not."""
    repo = OrderRepository(db)
    o = _order()
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o))

    pos_repo = PositionRepository(db)
    books = await pos_repo.load(account)
    pos = books["ema_cross|BTC/USD"]
    pos.stop_price = Decimal("54000")
    pos.target_price = Decimal("72000")
    pos.trail_atr_mult = 3.0
    pos.highest_close = 61000.0
    await pos_repo.save_protective_levels(account, pos)

    reloaded = (await PositionRepository(db).load(account))["ema_cross|BTC/USD"]
    assert reloaded.stop_price == Decimal("54000")
    assert reloaded.target_price == Decimal("72000")
    assert reloaded.trail_atr_mult == 3.0
    assert reloaded.highest_close == 61000.0


# ══════════════════════════════════════════════════════════════════════════
# Reconciliation
# ══════════════════════════════════════════════════════════════════════════

async def test_reconciliation_passes_when_all_three_sources_agree(db, account):
    repo = OrderRepository(db)
    o = _order(qty="0.5")
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o))

    res = await ReconciliationService(db).run(
        account, {"BTC/USD": Decimal("0.5")})
    assert res.ok, res.summary()
    assert "OK" in res.summary()


async def test_reconciliation_detects_a_corrupted_positions_row(db, account):
    """positions vs fill-replay — catches a bug in our own bookkeeping."""
    repo = OrderRepository(db)
    o = _order(qty="0.5")
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o))
    await db.execute("UPDATE positions SET qty = qty * 2")     # corrupt it

    res = await ReconciliationService(db).run(account)
    assert not res.ok
    assert any(d.source_a == "positions" for d in res.discrepancies)
    assert res.positions_vs_fills["BTC/USD"] == Decimal("0.5")


async def test_reconciliation_detects_an_exchange_mismatch(db, account):
    """fill-replay vs exchange — catches a fill we never saw."""
    repo = OrderRepository(db)
    o = _order(qty="0.5")
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o))

    res = await ReconciliationService(db).run(account, {"BTC/USD": Decimal("0.8")})
    assert not res.ok
    assert any(d.source_b == "exchange" for d in res.discrepancies)


async def test_unreachable_exchange_is_reported_not_assumed_clean(db, account):
    res = await ReconciliationService(db).run(account, None)
    assert res.ok
    assert res.exchange_available is False
    assert "EXCHANGE NOT CHECKED" in res.summary()


async def test_repair_rebuilds_positions_from_the_fill_ledger(db, account):
    repo = OrderRepository(db)
    o = _order(qty="0.5")
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o))
    await db.execute("UPDATE positions SET qty = 999")

    svc = ReconciliationService(db)
    assert not (await svc.run(account)).ok
    assert await svc.repair_positions_from_fills(account) == 1
    assert (await svc.run(account)).ok


async def test_orphan_orders_are_listed_for_requery(db, account):
    repo = OrderRepository(db)
    o = _order()
    await repo.insert_order(account, o)
    await repo.mark_submitted(o.client_order_id, "EX-1")
    orphans = await ReconciliationService(db).orphan_orders(account)
    assert orphans == [o.client_order_id]
    await repo.mark_terminal(o.client_order_id, OrderStatus.CANCELLED)
    assert await ReconciliationService(db).orphan_orders(account) == []


# ══════════════════════════════════════════════════════════════════════════
# Outbox — "traded but never told you" must be impossible
# ══════════════════════════════════════════════════════════════════════════

async def test_fill_and_its_notification_are_one_transaction(db, account):
    repo = OrderRepository(db)
    o = _order()
    await repo.insert_order(account, o)
    await repo.record_fill(account, _fill(o),
                           notification=("telegram", "Filled", "BTC 0.1 @ 60000"))
    assert await db.fetchval("SELECT count(*) FROM notifications") == 1
    assert await db.fetchval("SELECT status FROM notifications") == "pending"


async def test_outbox_drains_and_marks_sent(db):
    box = NotificationOutbox(db)
    await box.enqueue("telegram", "hello")
    sent, failed = await box.drain(lambda c, s, b: _ok(True))
    assert (sent, failed) == (1, 0)
    assert (await box.backlog()).get("sent") == 1


async def test_outbox_retries_with_backoff_then_gives_up_loudly(db):
    box = NotificationOutbox(db)
    nid = await box.enqueue("telegram", "unreachable")
    # First failure schedules a retry in the future, so an immediate second
    # drain must find nothing — otherwise backoff is not backoff.
    await box.drain(lambda c, s, b: _ok(False))
    assert await box.drain(lambda c, s, b: _ok(False)) == (0, 0)

    from ct.store.outbox import MAX_ATTEMPTS
    for _ in range(MAX_ATTEMPTS + 1):
        await db.execute(
            "UPDATE notifications SET next_attempt_at = now() WHERE id=$1", nid)
        await box.drain(lambda c, s, b: _ok(False))
    row = await db.fetchrow("SELECT * FROM notifications WHERE id=$1", nid)
    assert row["status"] == "failed"
    assert row["attempts"] >= MAX_ATTEMPTS
    assert row["last_error"]


async def test_a_raising_sender_does_not_propagate(db):
    box = NotificationOutbox(db)
    await box.enqueue("telegram", "boom")

    async def explode(c, s, b):
        raise RuntimeError("telegram is down")

    sent, failed = await box.drain(explode)          # must not raise
    assert (sent, failed) == (0, 1)


async def test_loud_notifications_drain_first(db):
    box = NotificationOutbox(db)
    await box.enqueue("telegram", "routine", priority="normal")
    await box.enqueue("telegram", "DRAWDOWN HALT", priority="loud")
    order: list[str] = []

    async def capture(c, s, b):
        order.append(b)
        return True

    await box.drain(capture)
    assert order[0] == "DRAWDOWN HALT"


async def _ok(value: bool) -> bool:
    return value


# ══════════════════════════════════════════════════════════════════════════
# Arming — the invariant, now with a home in tests/
# ══════════════════════════════════════════════════════════════════════════

def _env(**kw) -> str:
    base = dict(account_id=1, symbols=["BTC/USD"], strategies=["ema_cross"],
                max_notional_usd=Decimal(250),
                max_daily_notional_usd=Decimal(1000),
                risk_config_id=1, code_git_sha="abc123")
    base.update(kw)
    return envelope_hash(**base)


async def _arm(repo, account, envelope=None) -> int:
    return await repo.create_human(
        account, armed_by="operator", confirmation_phrase="I ACCEPT THE RISK",
        envelope=envelope or _env(), symbols=["BTC/USD"],
        strategies=["ema_cross"], max_notional_usd=Decimal(250),
        max_daily_notional_usd=Decimal(1000), hours=168)


async def test_arming_requires_a_typed_confirmation(db, account):
    repo = ArmingRepository(db)
    with pytest.raises(ValueError, match="confirmation phrase"):
        await repo.create_human(
            account, armed_by="qd-engine", confirmation_phrase="   ",
            envelope=_env(), symbols=["BTC/USD"], strategies=["ema_cross"],
            max_notional_usd=Decimal(250),
            max_daily_notional_usd=Decimal(1000))
    assert not await repo.is_armed(account)


async def test_human_arming_then_indefinite_renewal(db, account):
    repo = ArmingRepository(db)
    root = await _arm(repo, account)
    assert await repo.is_armed(account)

    last = root
    for _ in range(5):
        new_id = await repo.renew(account, _env())
        assert new_id is not None
        last = new_id
    assert await repo.is_armed(account)
    assert await repo.is_human_rooted(last)
    chain = await repo.chain(last)
    assert chain[0]["origin"] == "human"
    assert len(chain) == 6


async def test_renewal_is_refused_when_the_envelope_changes(db, account):
    repo = ArmingRepository(db)
    await _arm(repo, account)
    assert await repo.renew(account, _env(max_notional_usd=Decimal(500))) is None
    assert await repo.renew(account, _env(symbols=["BTC/USD", "ETH/USD"])) is None
    assert await repo.renew(account, _env(code_git_sha="deadbeef")) is None
    # ...and the unchanged envelope still renews.
    assert await repo.renew(account, _env()) is not None


async def test_disarming_the_root_kills_the_whole_renewal_chain(db, account):
    """The bug found on 2026-07-27: a renewal is its own row with its own
    disarmed_at, so disarming the root alone left every child active and the
    system kept trading through a drawdown halt."""
    repo = ArmingRepository(db)
    root = await _arm(repo, account)
    leaf = root
    for _ in range(3):
        leaf = await repo.renew(account, _env())

    await db.execute(
        """UPDATE live_arming SET disarmed_at=now(), disarmed_by='risk',
               disarm_reason='max_drawdown' WHERE id=$1""", root)
    assert not await repo.is_human_rooted(leaf)
    assert not await repo.is_armed(account)


async def test_an_orphan_renewal_cannot_be_inserted(db, account):
    """Enforced by the schema, not by application code, so no refactor can
    produce an arming with no human ancestor."""
    with pytest.raises(IntegrityError, match="renewal_must_have_parent"):
        await db.execute(
            """INSERT INTO live_arming (account_id, expires_at, armed_by, origin,
                   envelope_hash, max_notional_usd, max_daily_notional_usd,
                   allowed_symbols, allowed_strategies)
               VALUES ($1, now() + interval '1 day', 'qd-engine', 'auto_renew',
                       'x', 1, 1, '{}', '{}')""", account)


async def test_expired_arming_is_not_active(db, account):
    repo = ArmingRepository(db)
    root = await _arm(repo, account)
    await db.execute(
        "UPDATE live_arming SET expires_at = now() - interval '1 hour' WHERE id=$1",
        root)
    assert not await repo.is_armed(account)


async def test_due_for_renewal_window(db, account):
    repo = ArmingRepository(db)
    root = await _arm(repo, account)
    assert not await repo.due_for_renewal(account, timedelta(hours=24))
    await db.execute(
        "UPDATE live_arming SET expires_at = now() + interval '2 hours' WHERE id=$1",
        root)
    assert await repo.due_for_renewal(account, timedelta(hours=24))


async def test_auto_rearm_allowed_only_after_a_transient_disarm(db, account):
    repo = ArmingRepository(db)
    await _arm(repo, account)
    await repo.disarm(account, reason="exchange_unreachable")
    assert await repo.may_auto_rearm(account) is True

    await _arm(repo, account)
    await repo.disarm(account, reason="max_drawdown")
    assert await repo.may_auto_rearm(account) is False


# ══════════════════════════════════════════════════════════════════════════
# Event sink
# ══════════════════════════════════════════════════════════════════════════

async def test_event_sink_append_never_blocks_and_drains_in_batches(db):
    sink = PostgresEventSink(db, batch_size=50)
    bus = EventBus(sink, source="test")
    for i in range(120):
        bus.publish(order_filled(f"c{i}", "BTC/USD", "buy", Decimal(1),
                                 Decimal(10), Decimal(0), is_maker=True))
    # Nothing has touched the database yet.
    assert await sink.max_sequence() == 0
    while await sink.drain_once():
        pass
    assert await sink.max_sequence() == 120
    assert (await sink.stats())["written"] == 120


async def test_event_sink_insert_is_idempotent(db):
    sink = PostgresEventSink(db)
    bus = EventBus(sink, source="test")
    bus.publish(order_filled("c1", "BTC/USD", "buy", Decimal(1), Decimal(10),
                             Decimal(0), is_maker=True))
    await sink.drain_once()
    # Replaying the same sequence must not duplicate — this is what makes
    # backfill safe to run at every boot.
    sink.append(next(iter(await sink.fetch(limit=1))))
    await sink.drain_once()
    assert await db.fetchval("SELECT count(*) FROM events") == 1


async def test_event_sink_queue_is_bounded_and_drops_oldest(db):
    sink = PostgresEventSink(db, queue_size=10)
    for i in range(30):
        sink.append(order_filled(f"c{i}", "BTC/USD", "buy", Decimal(1),
                                 Decimal(10), Decimal(0),
                                 is_maker=True).with_seq(i + 1))
    assert sink.dropped == 20
    assert len(sink._queue) == 10


async def test_tee_survives_one_log_failing(db, tmp_path):
    """The durable log must keep working when the database sink does not."""
    class Broken(InMemoryEventLog):
        def append(self, event):
            raise OSError("nope")

    jsonl = JsonlEventLog(tmp_path)
    tee = TeeEventLog(jsonl, Broken())
    bus = EventBus(tee, source="test")
    for _ in range(3):
        bus.publish(order_filled("c", "BTC/USD", "buy", Decimal(1), Decimal(10),
                                 Decimal(0), is_maker=True))
    tee.flush()
    assert len(list(jsonl.read())) == 3
    assert tee.failures[1] == 3


async def test_backfill_replays_what_postgres_missed(db, tmp_path):
    """The catch-up path after Postgres has been down."""
    jsonl = JsonlEventLog(tmp_path)
    sink = PostgresEventSink(db)
    bus = EventBus(TeeEventLog(jsonl, sink), source="test")
    for i in range(20):
        bus.publish(order_filled(f"c{i}", "BTC/USD", "buy", Decimal(1),
                                 Decimal(10), Decimal(0), is_maker=True))
    # Simulate the database having been unreachable: drop the queue entirely.
    sink._queue.clear()
    assert await sink.max_sequence() == 0

    written = await backfill_from_jsonl(sink, jsonl)
    assert written == 20
    assert await sink.max_sequence() == 20
    # Idempotent: running it again writes nothing new.
    assert await backfill_from_jsonl(sink, jsonl) == 0


async def test_trade_lifecycles_view_returns_maker_ratio_zero_not_null(db):
    """The NULL-propagation bug from Item 1, now pinned in tests/."""
    await db.execute(
        """INSERT INTO events (seq, ts, type, correlation_id, payload) VALUES
           (1, now(), 'execution.order_filled', 'all-taker',
            '{"qty":2,"fee":40,"is_maker":false}'::jsonb),
           (2, now(), 'execution.order_filled', 'mixed',
            '{"qty":1,"fee":10,"is_maker":true}'::jsonb),
           (3, now(), 'execution.order_filled', 'mixed',
            '{"qty":3,"fee":30,"is_maker":false}'::jsonb)""")
    rows = {r["correlation_id"]: r for r in
            await db.fetch("SELECT * FROM trade_lifecycles")}
    assert rows["all-taker"]["maker_ratio"] == Decimal(0)
    assert rows["mixed"]["maker_ratio"] == Decimal("0.25")


# ══════════════════════════════════════════════════════════════════════════
# Signals and risk events
# ══════════════════════════════════════════════════════════════════════════

async def test_rejected_signals_are_recorded_and_countable(db, account):
    repo = SignalRepository(db)
    for _ in range(3):
        await repo.record(account, TS, "ema_cross", "BTC/USD", 1,
                          accepted=False, reject_code="EDGE_BELOW_FEES",
                          reject_detail="30bps < 90bps")
    await repo.record(account, TS, "ema_cross", "BTC/USD", 1, accepted=True,
                      sized_qty=Decimal("0.1"))
    counts = await repo.rejection_counts(account, days=3650)
    assert counts == {"EDGE_BELOW_FEES": 3}


async def test_risk_events_can_be_raised_and_resolved(db, account):
    repo = RiskEventRepository(db)
    eid = await repo.record(account, "critical", "RECONCILE_MISMATCH",
                            "BTC/USD off by 0.3", {"symbol": "BTC/USD"})
    assert len(await repo.unresolved(account)) == 1
    await repo.resolve(eid)
    assert await repo.unresolved(account) == []


async def test_transaction_rolls_back_on_error(db, account):
    """The guarantee the outbox depends on."""
    with pytest.raises(RuntimeError):
        async with db.transaction() as conn:
            await conn.execute(
                """INSERT INTO risk_events (account_id, severity, code)
                   VALUES ($1,'info','X')""", account)
            raise RuntimeError("abort")
    assert await db.fetchval("SELECT count(*) FROM risk_events") == 0


async def test_schema_version_is_recorded_and_drift_is_detected(db, caplog):
    """Deliberately not Alembic — see `Database._warn_on_drift`. What has value
    today is knowing the running database no longer matches the code."""
    import logging
    row = await db.fetchrow(
        "SELECT * FROM schema_version ORDER BY applied_at DESC LIMIT 1")
    assert row is not None and len(row["sha256"]) == 64

    original = row["sha256"]
    # Pretend the file changed under a database built from an older version.
    await db.execute("UPDATE schema_version SET sha256 = 'deadbeef'")
    try:
        with caplog.at_level(logging.WARNING, logger="ct.store.db"):
            assert await db.bootstrap(SCHEMA) is False
        assert any("SCHEMA DRIFT" in r.message for r in caplog.records)
    finally:
        # schema_version deliberately survives TRUNCATE — it describes the
        # schema, not the data — so this test must restore it.
        await db.execute("UPDATE schema_version SET sha256 = $1", original)


async def test_bootstrap_registers_a_late_schema_version_marker(db):
    """An entrypoint-initialised database (schema.sql via the init mount) has
    the full schema but no `schema_version`. bootstrap() must register a
    clearly-labelled late baseline rather than leave drift detection silently
    dead and readers of the table broken (/system 500'd — 2026-07-30)."""
    await db.execute("DROP TABLE IF EXISTS schema_version")
    applied = await db.bootstrap(SCHEMA)
    assert applied is False, "schema exists; bootstrap must not re-apply it"
    row = await db.fetchrow(
        "SELECT sha256, source FROM schema_version ORDER BY applied_at DESC LIMIT 1")
    assert row is not None
    assert "late registration" in row["source"]
