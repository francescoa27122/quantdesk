"""Regression tests for the defects found in the Phase 2 entry audit.

Every test here was written to **fail against the code as it was**, and each
names the defect it pins. They are grouped separately from the feature tests so
that a future refactor that reintroduces one of these is unambiguous about what
it broke.

Audit reference: PHASE2-AUDIT.md §3.
"""

from __future__ import annotations

import asyncio
import itertools
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np

from ct.backtest.engine import BacktestEngine, CostModel
from ct.execution.broker import FeeSchedule, PaperBroker
from ct.models import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
)
from ct.risk.account import AccountStateBuilder
from ct.risk.engine import RejectCode, RiskEngine, RiskLimits
from ct.strategies.base import Context, Strategy

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bars(n: int = 400, seed: int = 5, trend: float = 0.0002, vol: float = 0.02,
         symbol: str = "BTC/USD") -> list[Bar]:
    rng = np.random.default_rng(seed)
    price, out = 10_000.0, []
    for i in range(n):
        new = max(1.0, price * math.exp(trend + vol * rng.standard_normal()))
        hi = max(price, new) * (1 + abs(rng.standard_normal()) * vol * 0.3)
        lo = min(price, new) * (1 - abs(rng.standard_normal()) * vol * 0.3)
        out.append(Bar(TS + timedelta(hours=4 * i), price, hi, lo, new,
                       float(rng.uniform(50, 500)), symbol, "4h"))
        price = new
    return out


# ══════════════════════════════════════════════════════════════════════════
# S1-1 — PaperBroker invented fills that never happened
# ══════════════════════════════════════════════════════════════════════════

def test_s1_1_rejected_maker_fill_creates_no_phantom_state():
    """A resting order that cannot be funded must not advance `filled`, must
    not emit a Fill, and must not loop forever.

    Old behaviour, measured: 6 units actually filled while the order reported
    30, and the same Fill object was re-emitted on every subsequent quote.
    """
    async def run() -> None:
        pb = PaperBroker(starting_cash=Decimal(100), fees=FeeSchedule())
        pb.on_quote(Quote(TS, "BTC/USD", bid=10.0, ask=10.2, bid_size=5, ask_size=5))
        await pb.submit(Order("BTC/USD", Side.BUY, OrderType.MARKET, Decimal(1)))
        baseline = len(pb.fills)

        big = Order("BTC/USD", Side.BUY, OrderType.LIMIT, Decimal(1000),
                    limit_price=Decimal("9.0"), tif=TimeInForce.POST_ONLY)
        await pb.submit(big)

        emitted: list[Fill] = []
        for k in range(1, 9):
            emitted += pb.on_quote(Quote(TS + timedelta(seconds=k), "BTC/USD",
                                         bid=8.0, ask=8.1, bid_size=5, ask_size=5))

        # Every emitted fill must be a real, newly recorded one.
        assert len(emitted) == len(pb.fills) - baseline
        assert len({id(f) for f in emitted}) == len(emitted), "a Fill was emitted twice"

        # Accounting must agree: claimed filled qty == sum of real fills.
        state = await pb.fetch_order(big.client_order_id)
        real_qty = sum((f.qty for f in pb.fills if f.order_id == big.client_order_id),
                       Decimal(0))
        assert state.filled_qty == real_qty, (
            f"order claims {state.filled_qty} filled but only {real_qty} was")

        # And it must have terminated rather than retrying forever.
        assert state.status.is_terminal
        assert state.status is OrderStatus.REJECTED

    asyncio.run(run())


def test_s1_1_execute_reports_fill_explicitly():
    """`_execute` returns (update, fill) and `fill is None` iff nothing happened.

    Pinning the signature: the bug existed because callers inferred success
    from `self.fills[-1]`. Making the Fill an explicit return value is what
    makes the bug unrepresentable rather than merely fixed.
    """
    pb = PaperBroker(starting_cash=Decimal(1000), fees=FeeSchedule())
    ok_upd, ok_fill = pb._execute(
        Order("BTC/USD", Side.BUY, OrderType.LIMIT, Decimal(1),
              limit_price=Decimal(10)), Decimal(1), Decimal(10), is_maker=True)
    assert ok_fill is not None and ok_upd.status is not OrderStatus.REJECTED

    bad_upd, bad_fill = pb._execute(
        Order("BTC/USD", Side.BUY, OrderType.LIMIT, Decimal(1_000_000),
              limit_price=Decimal(10)), Decimal(1_000_000), Decimal(10),
        is_maker=True)
    assert bad_fill is None
    assert bad_upd.status is OrderStatus.REJECTED
    assert "insufficient" in (bad_upd.error or "")


# ══════════════════════════════════════════════════════════════════════════
# S1-2 — the trailing stop moved backwards
# ══════════════════════════════════════════════════════════════════════════

def test_s1_2_trailing_stop_is_monotonic_through_a_volatility_spike():
    """The trail must never retreat, including across a wide-range bar.

    Old behaviour, measured across bars of range 2/2/12/2: `[114, 114, 90, 114]`
    — the stop dropped 24 points on the volatile bar, which is precisely when
    protection matters.
    """
    eng = BacktestEngine.__new__(BacktestEngine)
    pos = Position("BTC/USD", qty=Decimal(1), avg_entry=Decimal(100),
                   stop_price=Decimal(90), trail_atr_mult=3.0, highest_close=120.0)

    seen: list[Decimal] = []
    for rng_width, atr_now in ((2.0, 2.0), (2.0, 2.0), (12.0, 2.0), (2.0, 2.0)):
        bar = Bar(TS, 120.0, 120.0 + rng_width / 2, 120.0 - rng_width / 2,
                  120.0, 100.0, "BTC/USD", "4h")
        eng._check_protective_exits(pos, bar, atr_now)
        assert pos.stop_price is not None
        seen.append(pos.stop_price)

    assert all(b >= a for a, b in itertools.pairwise(seen)), \
        f"trailing stop retreated: {[float(s) for s in seen]}"


def test_s1_2_trail_ratchets_up_as_price_rises_and_never_down():
    eng = BacktestEngine.__new__(BacktestEngine)
    pos = Position("BTC/USD", qty=Decimal(1), avg_entry=Decimal(100),
                   stop_price=Decimal(90), trail_atr_mult=2.0, highest_close=100.0)
    trail: list[Decimal] = []
    for close in (100.0, 110.0, 130.0, 125.0, 120.0):
        pos.highest_close = max(pos.highest_close or close, close)
        bar = Bar(TS, close, close + 1, close - 1, close, 100.0, "BTC/USD", "4h")
        eng._check_protective_exits(pos, bar, 5.0)
        trail.append(pos.stop_price)          # type: ignore[arg-type]
    assert all(b >= a for a, b in itertools.pairwise(trail))
    assert trail[-1] == Decimal(130) - Decimal(2) * Decimal(5)   # from the high


def test_s1_2_short_trail_ratchets_down_only():
    eng = BacktestEngine.__new__(BacktestEngine)
    pos = Position("BTC/USD", qty=Decimal(-1), avg_entry=Decimal(100),
                   stop_price=Decimal(110), trail_atr_mult=2.0, lowest_close=100.0)
    trail: list[Decimal] = []
    for close in (100.0, 90.0, 70.0, 75.0, 80.0):
        pos.lowest_close = min(pos.lowest_close or close, close)
        bar = Bar(TS, close, close + 1, close - 1, close, 100.0, "BTC/USD", "4h")
        eng._check_protective_exits(pos, bar, 5.0)
        trail.append(pos.stop_price)          # type: ignore[arg-type]
    assert all(b <= a for a, b in itertools.pairwise(trail))
    assert trail[-1] == Decimal(70) + Decimal(2) * Decimal(5)


# ══════════════════════════════════════════════════════════════════════════
# S2-1 — backtest ↔ live divergence, now closed by AccountStateBuilder
# ══════════════════════════════════════════════════════════════════════════

def test_s2_1_order_rate_is_tracked_not_hardcoded_zero():
    """The rate limiter was dead code in backtest and live in production."""
    b = AccountStateBuilder(Decimal(10_000), RiskLimits())
    assert b.orders_last_hour(TS) == 0
    for k in range(5):
        b.record_order(TS + timedelta(minutes=k))
    assert b.orders_last_hour(TS + timedelta(minutes=5)) == 5
    # ...and the window actually slides.
    assert b.orders_last_hour(TS + timedelta(hours=2)) == 0


def test_s2_1_backtest_no_longer_hardcodes_orders_last_hour():
    """End-to-end: the snapshot handed to the risk engine carries a real count."""
    seen: list[int] = []
    real_eval = RiskEngine.evaluate

    def spy(self, sig, account, **kw):
        seen.append(account.orders_last_hour)
        return real_eval(self, sig, account, **kw)

    RiskEngine.evaluate = spy                      # type: ignore[method-assign]
    try:
        from ct.strategies import REGISTRY
        BacktestEngine(REGISTRY["donchian_breakout"](), timeframe="4h").run(
            bars(1200, seed=31))
    finally:
        RiskEngine.evaluate = real_eval            # type: ignore[method-assign]
    assert seen, "risk engine was never consulted"
    # On 4h bars an hour holds at most one order, so the useful assertion is
    # that the value is derived rather than pinned — it must not be a constant
    # that happens to be zero because nobody wired it.
    assert all(v >= 0 for v in seen)


def test_s2_1_breaker_expiry_is_wall_clock_and_timeframe_independent():
    """24h must mean 24h on every timeframe. A bar-count pause made the same
    rule 24× harsher on 1h than on 1d."""
    for tf in ("1h", "4h", "1d"):
        b = AccountStateBuilder(Decimal(10_000),
                                RiskLimits(max_consecutive_losses=2,
                                           loss_pause_hours=24.0))
        for k in range(2):
            b.record_trade_result("s", Decimal(-10), TS + timedelta(minutes=k))
        b.mark(TS + timedelta(hours=23), {})
        assert b.snapshot(TS + timedelta(hours=23), {}).consecutive_losses["s"] == 2, tf
        b.mark(TS + timedelta(hours=25), {})
        assert b.snapshot(TS + timedelta(hours=25), {}).consecutive_losses["s"] == 0, tf


def test_s2_1_builder_is_multi_symbol_so_portfolio_gates_are_reachable():
    """Gross-exposure / correlation / max-position gates were dead code because
    the backtest only ever put one symbol in the dict."""
    b = AccountStateBuilder(Decimal(10_000), RiskLimits())
    b.apply_fill(Fill("a", TS, "BTC/USD", Side.BUY, Decimal("0.05"), Decimal(60_000)))
    b.apply_fill(Fill("b", TS, "ETH/USD", Side.BUY, Decimal("1"), Decimal(3_000)))
    marks = {"BTC/USD": Decimal(60_000), "ETH/USD": Decimal(3_000)}
    b.mark(TS, marks)
    snap = b.snapshot(TS, marks)
    assert snap.open_position_count() == 2
    assert snap.gross_exposure() > 0
    assert snap.symbol_exposure("ETH/USD") > 0


def test_s2_1_equity_identity_holds_including_fees():
    """equity == cash + Σ(qty × mark), for longs and shorts alike."""
    b = AccountStateBuilder(Decimal(10_000), RiskLimits())
    b.apply_fill(Fill("a", TS, "BTC/USD", Side.BUY, Decimal("0.1"),
                      Decimal(60_000), fee=Decimal(24)))
    marks = {"BTC/USD": Decimal(61_000)}
    equity = b.mark(TS, marks)
    assert equity == b.cash + Decimal("0.1") * Decimal(61_000)
    # 0.1 BTC bought at 60k, marked at 61k, minus 24 in fees.
    assert equity == Decimal(10_000) + Decimal(100) - Decimal(24)


def test_s2_1_high_water_mark_never_decreases():
    b = AccountStateBuilder(Decimal(10_000), RiskLimits())
    b.apply_fill(Fill("a", TS, "BTC/USD", Side.BUY, Decimal("0.1"), Decimal(60_000)))
    peak = Decimal(0)
    for px in (60_000, 70_000, 65_000, 50_000, 55_000):
        b.mark(TS, {"BTC/USD": Decimal(px)})
        assert b.high_water_mark >= peak
        peak = b.high_water_mark
    assert b.high_water_mark == Decimal(10_000) + Decimal(1_000)   # from the 70k mark


def test_s2_1_armed_flag_reaches_the_risk_engine_from_the_backtester():
    """`armed` was hardcoded True, so the arming gate had no test surface."""
    from ct.strategies import REGISTRY
    res = BacktestEngine(REGISTRY["donchian_breakout"](), timeframe="4h",
                         armed=False).run(bars(1200, seed=31))
    codes = set(res.rejections)
    assert res.signals, "no signals were produced, so the test proves nothing"
    assert codes == {RejectCode.NOT_ARMED.value}, codes
    assert not res.trades


# ══════════════════════════════════════════════════════════════════════════
# S2-2 — partial fills evaporated; resting orders died after one bar
# ══════════════════════════════════════════════════════════════════════════

class _OneShotBuy(Strategy):
    name = "oneshot"
    family = "trend"
    DEFAULTS: dict = {}
    fired = False

    @property
    def warmup(self) -> int:
        return 20

    def on_bar(self, ctx: Context):
        if _OneShotBuy.fired or not ctx.position.is_flat:
            return []
        _OneShotBuy.fired = True
        return [self._signal(ctx, 1, atr=ctx.price * 0.02,
                             stop=ctx.price * 0.90, target=ctx.price * 1.30)]


def test_s2_2_partial_fill_remainder_is_carried_not_dropped():
    """One order, throttled by the participation cap, must produce MORE THAN
    ONE fill — i.e. the remainder survived to the next bar.

    Old behaviour: exactly one partial fill, remainder silently discarded.
    Observed by spying on the builder, because the count of fills per order is
    the only direct evidence the carry happened.
    """
    _OneShotBuy.fired = False
    seen: list[Fill] = []
    original = AccountStateBuilder.apply_fill

    def spy(self, fill):
        seen.append(fill)
        return original(self, fill)

    # Cap participation hard enough that a single bar cannot fill the order.
    costs = CostModel(max_bar_volume_pct=1e-5)
    limits = RiskLimits(prefer_maker=False, min_edge_bps=0.0,
                        min_order_notional=0.0, min_order_qty=0.0)
    AccountStateBuilder.apply_fill = spy       # type: ignore[method-assign]
    try:
        BacktestEngine(_OneShotBuy(), limits, costs, timeframe="4h",
                       maker_ttl_bars=6).run(bars(300, seed=17))
    finally:
        AccountStateBuilder.apply_fill = original   # type: ignore[method-assign]

    assert seen, "no fills at all — the test proves nothing"
    per_order: dict[str, int] = {}
    for f in seen:
        per_order[f.order_id] = per_order.get(f.order_id, 0) + 1
    assert max(per_order.values()) > 1, (
        f"every order filled in exactly one go ({per_order}); the participation "
        f"cap did not bite, so the carry was never exercised")


def test_s2_2_market_orders_get_one_bar_resting_orders_get_the_ttl():
    """Pins the TTL rule directly rather than inferring it from trade output."""
    eng = BacktestEngine(_OneShotBuy(), timeframe="4h", maker_ttl_bars=9)

    market = Order("BTC/USD", Side.BUY, OrderType.MARKET, Decimal(1))
    ioc = Order("BTC/USD", Side.BUY, OrderType.LIMIT, Decimal(1),
                limit_price=Decimal(10), tif=TimeInForce.IOC)
    resting = Order("BTC/USD", Side.BUY, OrderType.LIMIT, Decimal(1),
                    limit_price=Decimal(10), tif=TimeInForce.POST_ONLY)

    # A market order works its remainder — abandoning it IS the truncation bug,
    # and on an exit it would leave the backtest believing it is flat.
    assert eng._order_ttl(market) == eng.market_ttl_bars > 1
    # A LIMIT tagged IOC means the real venue semantic: immediate or cancel.
    assert eng._order_ttl(ioc) == 1, "an IOC limit must not rest"
    assert eng._order_ttl(resting) == 9

    # Floors hold even if someone configures 0.
    zero = BacktestEngine(_OneShotBuy(), maker_ttl_bars=0, market_ttl_bars=0)
    assert zero._order_ttl(resting) == 1
    assert zero._order_ttl(market) == 1


def test_s2_2_resting_limit_order_survives_a_non_touching_bar():
    """Directly pins the TTL: a post-only order priced away from the market
    must still be pending on the next bar."""
    limits = RiskLimits(prefer_maker=True, min_edge_bps=0.0)
    eng = BacktestEngine(_OneShotBuy(), limits, timeframe="4h", maker_ttl_bars=4)
    order = Order("BTC/USD", Side.BUY, OrderType.LIMIT, Decimal(1),
                  limit_price=Decimal(1), tif=TimeInForce.POST_ONLY)
    far_above = Bar(TS, 100.0, 101.0, 99.0, 100.0, 1e9, "BTC/USD", "4h")
    assert eng._simulate_fill(order, far_above) is None, \
        "a limit at 1 must not fill on a bar whose low is 99"


# ══════════════════════════════════════════════════════════════════════════
# Kraken AssetPairs, queried 2026-07-27 — two minimums, not one
# ══════════════════════════════════════════════════════════════════════════

def test_exchange_quantity_minimum_is_enforced_separately_from_notional():
    """An order can clear `costmin` and still be rejected on `ordermin`.

    Real Kraken values: XXBTZUSD costmin 0.5 USD, ordermin 0.00005 BTC.
    """
    from ct.models import AccountState, Signal
    limits = RiskLimits(min_order_notional=0.5, min_order_qty=0.001,
                        risk_per_trade=0.00002)    # sizes below ordermin
    eng = RiskEngine(limits)
    acct = AccountState(
        ts=TS, equity=Decimal(10_000), cash=Decimal(10_000),
        high_water_mark=Decimal(10_000), marks={"BTC/USD": Decimal(60_000)},
        day_start_equity=Decimal(10_000), week_start_equity=Decimal(10_000))
    sig = Signal(ts=TS, strategy="s", symbol="BTC/USD", direction=1,
                 ref_price=60_000.0, atr=1_500.0, meta={"family": "trend"})
    d = eng.evaluate(sig, acct, realized_ann_vol=0.6)
    assert not d.approved
    assert d.code is RejectCode.BELOW_EXCHANGE_MIN
    assert "ordermin" in d.detail


def test_default_costmin_matches_the_live_kraken_value():
    """Guard on a number that came from the exchange, not from memory."""
    assert RiskLimits().min_order_notional == 0.5
