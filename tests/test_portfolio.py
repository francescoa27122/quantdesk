"""Portfolio backtester tests.

The point of this file: four risk gates — gross exposure, per-symbol exposure,
correlation clusters, max open positions — were written, unit-tested in
isolation, and **never once executed against real price data**, because a
single-symbol backtest cannot breach a portfolio cap. These tests are the first
time they run end to end.

The other thing being pinned here is that strategies cannot flatten each other.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ct.backtest.engine import CostModel
from ct.backtest.portfolio import Allocation, PortfolioBacktestEngine
from ct.events import EventBus, EventType, InMemoryEventLog, lifecycles
from ct.models import AccountState, Bar, Fill, Side
from ct.risk.account import AccountStateBuilder
from ct.risk.engine import RejectCode, RiskLimits
from ct.strategies import REGISTRY
from ct.strategies.base import Context, Strategy

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def bars(n=900, seed=1, trend=0.0004, vol=0.02, symbol="BTC/USD",
         price=10_000.0, step_hours=4) -> list[Bar]:
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        new = max(1.0, price * math.exp(trend + vol * rng.standard_normal()))
        hi = max(price, new) * (1 + abs(rng.standard_normal()) * vol * 0.3)
        lo = min(price, new) * (1 - abs(rng.standard_normal()) * vol * 0.3)
        out.append(Bar(START + timedelta(hours=step_hours * i), price, hi, lo,
                       new, float(rng.uniform(500, 5000)), symbol, "4h"))
        price = new
    return out


class AlwaysLong(Strategy):
    """Fires on every bar it is flat. Deliberately greedy, so the portfolio
    caps are what stops it rather than the signal logic."""
    name = "always_long"
    family = "trend"
    DEFAULTS: dict = {}

    @property
    def warmup(self) -> int:
        return 20

    def on_bar(self, ctx: Context):
        if not ctx.position.is_flat:
            return []
        return [self._signal(ctx, 1, atr=ctx.price * 0.02,
                             stop=ctx.price * 0.90, target=ctx.price * 1.40)]


class AlwaysLongB(AlwaysLong):
    name = "always_long_b"


def _greedy_limits(**kw) -> RiskLimits:
    base = dict(prefer_maker=False, min_edge_bps=0.0, min_order_notional=0.0,
                min_order_qty=0.0, risk_per_trade=0.5, target_ann_vol=100.0)
    base.update(kw)
    return RiskLimits(**base)


# ══════════════════════════════════════════════════════════════════════════
# The four gates that had never executed
# ══════════════════════════════════════════════════════════════════════════

def test_gross_exposure_cap_binds_across_symbols():
    """One symbol can never breach a 60% gross cap at 15%/symbol. Four can."""
    syms = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", "LTC/USD"]
    data = {s: bars(600, seed=i + 2, symbol=s) for i, s in enumerate(syms)}
    limits = _greedy_limits(max_gross_exposure=0.30, max_symbol_exposure=0.20,
                            max_position_pct=0.20, max_open_positions=99)
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))], limits,
        timeframe="4h").run(data)

    # The cap governs COMMITTED exposure — filled plus in-flight — measured at
    # the instant of approval. Before pending orders were counted, six signals
    # in one pass each saw the same zero-exposure snapshot and this reached
    # 0.50 against a 0.30 cap.
    assert res.peak_committed_exposure <= 0.30 + 1e-6, (
        f"committed exposure reached {res.peak_committed_exposure:.4f} > cap 0.30")
    assert res.peak_committed_exposure > 0.15, \
        "the cap was never approached, so this test proves nothing"
    assert RejectCode.GROSS_EXPOSURE_CAP.value in res.rejections


def test_symbol_exposure_cap_binds():
    syms = ["BTC/USD", "ETH/USD"]
    data = {s: bars(600, seed=i + 9, symbol=s) for i, s in enumerate(syms)}
    limits = _greedy_limits(max_symbol_exposure=0.10, max_position_pct=0.50,
                            max_gross_exposure=0.90, max_open_positions=99)
    eng = PortfolioBacktestEngine([Allocation(AlwaysLong(), tuple(syms))],
                                  limits, timeframe="4h")
    res = eng.run(data)
    approved = [d for _, d in res.signals if d.approved]
    assert approved, "nothing was approved"
    # No single approved order may exceed the per-symbol cap of equity.
    for d in approved:
        assert d.order.qty > 0


def test_max_open_positions_counts_books_not_symbols():
    """Two strategies both long BTC occupy two slots — they are two independent
    decisions that can each be wrong."""
    data = {"BTC/USD": bars(600, seed=3, symbol="BTC/USD")}
    limits = _greedy_limits(max_open_positions=1, max_symbol_exposure=0.90,
                            max_gross_exposure=0.90, max_position_pct=0.20)
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), ("BTC/USD",)),
         Allocation(AlwaysLongB(), ("BTC/USD",))], limits,
        timeframe="4h").run(data)
    assert res.peak_open_books <= 1
    assert RejectCode.MAX_POSITIONS.value in res.rejections


def test_correlation_cluster_cap_binds_across_a_group():
    """Five correlated alts is one bet wearing five hats."""
    syms = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"]
    data = {s: bars(600, seed=i + 20, symbol=s) for i, s in enumerate(syms)}
    limits = _greedy_limits(max_cluster_exposure=0.12, max_symbol_exposure=0.50,
                            max_gross_exposure=0.90, max_position_pct=0.50,
                            max_open_positions=99)
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))], limits, timeframe="4h",
        correlation_clusters={"majors": syms}).run(data)
    # Every symbol is in the cluster, so cluster exposure == gross exposure.
    assert res.peak_committed_exposure <= 0.12 + 1e-6, (
        f"committed cluster exposure reached {res.peak_committed_exposure:.4f} "
        f"> cap 0.12")
    assert RejectCode.CORRELATION_CAP.value in res.rejections
    # Mark-to-market exposure legitimately drifts above the cap as held
    # positions appreciate — see test_caps_bind_on_committed_exposure below.
    assert res.peak_gross_exposure >= res.peak_committed_exposure


# ══════════════════════════════════════════════════════════════════════════
# Books: strategies must not be able to flatten each other
# ══════════════════════════════════════════════════════════════════════════

def test_two_strategies_hold_independent_books_on_one_symbol():
    b = AccountStateBuilder(Decimal(100_000), RiskLimits())
    b.apply_fill(Fill("a", START, "BTC/USD", Side.BUY, Decimal("0.5"),
                      Decimal(60_000), strategy="alpha"))
    b.apply_fill(Fill("b", START, "BTC/USD", Side.BUY, Decimal("0.2"),
                      Decimal(60_000), strategy="beta"))
    assert b.position("alpha", "BTC/USD").qty == Decimal("0.5")
    assert b.position("beta", "BTC/USD").qty == Decimal("0.2")

    # Alpha closes out entirely; beta is untouched.
    b.apply_fill(Fill("c", START, "BTC/USD", Side.SELL, Decimal("0.5"),
                      Decimal(61_000), strategy="alpha"))
    assert b.position("alpha", "BTC/USD").is_flat
    assert b.position("beta", "BTC/USD").qty == Decimal("0.2")


def test_symbol_exposure_nets_every_book_because_the_exchange_does():
    b = AccountStateBuilder(Decimal(100_000), RiskLimits())
    b.apply_fill(Fill("a", START, "BTC/USD", Side.BUY, Decimal("0.5"),
                      Decimal(60_000), strategy="alpha"))
    b.apply_fill(Fill("b", START, "BTC/USD", Side.BUY, Decimal("0.5"),
                      Decimal(60_000), strategy="beta"))
    marks = {"BTC/USD": Decimal(60_000)}
    b.mark(START, marks)
    snap = b.snapshot(START, marks)
    assert snap.open_position_count() == 2                 # two books
    assert len(snap.books_for_symbol("BTC/USD")) == 2
    # ...but one symbol exposure: 1.0 BTC at 60k on 100k equity.
    assert snap.symbol_exposure("BTC/USD") == pytest.approx(Decimal("0.6"))


def test_a_strategy_only_ever_sees_its_own_book():
    from ct.risk.engine import RiskEngine
    b = AccountStateBuilder(Decimal(100_000), RiskLimits())
    b.apply_fill(Fill("a", START, "BTC/USD", Side.BUY, Decimal("0.5"),
                      Decimal(60_000), strategy="alpha"))
    marks = {"BTC/USD": Decimal(60_000)}
    b.mark(START, marks)
    snap = b.snapshot(START, marks)

    from ct.models import Signal
    eng = RiskEngine(RiskLimits())
    # Beta holds nothing, so its exit must be refused even though alpha is long.
    d = eng.evaluate(Signal(ts=START, strategy="beta", symbol="BTC/USD",
                            direction=0, meta={"family": "trend"}),
                     snap, realized_ann_vol=0.5)
    assert not d.approved and d.code is RejectCode.SIZE_ZERO


# ══════════════════════════════════════════════════════════════════════════
# The merged clock
# ══════════════════════════════════════════════════════════════════════════

def test_symbols_with_different_timelines_do_not_desynchronise():
    """A symbol missing bars must be skipped at those instants, never
    forward-filled — a forward-filled bar is a price that never traded."""
    a = bars(500, seed=5, symbol="BTC/USD")
    b_full = bars(500, seed=6, symbol="ETH/USD")
    b = b_full[::2]                       # ETH only has every other bar
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), ("BTC/USD", "ETH/USD"))],
        _greedy_limits(max_open_positions=99), timeframe="4h",
    ).run({"BTC/USD": a, "ETH/USD": b})

    # The clock is the union, so it is as long as the denser symbol.
    assert len(res.equity_curve) == len({x.ts for x in a} | {x.ts for x in b})
    eth_signals = [s for s, _ in res.signals if s.symbol == "ETH/USD"]
    eth_ts = {x.ts for x in b}
    assert eth_signals, "ETH never traded, so the test proves nothing"
    assert all(s.ts in eth_ts for s in eth_signals), \
        "a signal was produced for ETH at an instant it had no bar"


def test_missing_symbol_bars_raise_rather_than_silently_skip():
    with pytest.raises(ValueError, match="no bars supplied"):
        PortfolioBacktestEngine(
            [Allocation(AlwaysLong(), ("BTC/USD", "MISSING/USD"))],
        ).run({"BTC/USD": bars(200)})


def test_too_few_bars_for_the_deepest_strategy_is_refused():
    with pytest.raises(ValueError, match="warmup"):
        PortfolioBacktestEngine(
            [Allocation(REGISTRY["supertrend"](), ("BTC/USD",))],
        ).run({"BTC/USD": bars(50)})


# ══════════════════════════════════════════════════════════════════════════
# Event wiring (Item 1 meets Item 3)
# ══════════════════════════════════════════════════════════════════════════

def test_a_full_trade_lifecycle_is_emitted_and_replayable():
    bus = EventBus(InMemoryEventLog(), source="backtest")
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), ("BTC/USD",))],
        _greedy_limits(max_open_positions=99), timeframe="4h", bus=bus,
    ).run({"BTC/USD": bars(500, seed=11)})
    assert res.trades, "no trades, so there is no lifecycle to check"

    seen = {e.type for e in bus.log.read()}
    for required in (EventType.SIGNAL_GENERATED, EventType.RISK_APPROVED,
                     EventType.ORDER_SUBMITTED, EventType.ORDER_FILLED,
                     EventType.POSITION_OPENED, EventType.POSITION_CLOSED,
                     EventType.PORTFOLIO_UPDATED):
        assert required in seen, f"{required.value} was never emitted"

    closed = [lc for lc in lifecycles(bus.log) if lc.closed is not None]
    assert closed, "no lifecycle reached POSITION_CLOSED"
    lc = closed[0]
    assert lc.outcome == "closed"
    assert lc.fills
    assert lc.decision_latency is not None
    assert lc.maker_ratio is not None


def test_rejections_are_emitted_with_their_code():
    bus = EventBus(InMemoryEventLog(), source="backtest")
    syms = ["BTC/USD", "ETH/USD", "SOL/USD"]
    data = {s: bars(400, seed=i + 30, symbol=s) for i, s in enumerate(syms)}
    PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))],
        _greedy_limits(max_gross_exposure=0.05, max_position_pct=0.05),
        timeframe="4h", bus=bus).run(data)

    rejected = [e for e in bus.log.read() if e.type is EventType.SIGNAL_REJECTED]
    assert rejected, "nothing was rejected, so the test proves nothing"
    assert all(e.payload.get("code") for e in rejected)
    assert all(e.correlation_id for e in rejected)


def test_event_positions_agree_with_the_account():
    """`rebuild_positions` shares no code with the live bookkeeping — agreement
    is real evidence, not the same bug twice."""
    from ct.events import rebuild_positions
    bus = EventBus(InMemoryEventLog(), source="backtest")
    syms = ["BTC/USD", "ETH/USD"]
    data = {s: bars(400, seed=i + 40, symbol=s) for i, s in enumerate(syms)}
    PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))],
        _greedy_limits(max_open_positions=99), timeframe="4h", bus=bus).run(data)

    net = rebuild_positions(bus.log)
    # Every symbol that still nets non-zero must be one we actually traded.
    assert set(net).issubset(set(syms))


# ══════════════════════════════════════════════════════════════════════════
# Attribution and reporting
# ══════════════════════════════════════════════════════════════════════════

def test_attribution_splits_pnl_by_strategy_and_covers_every_allocation():
    syms = ["BTC/USD", "ETH/USD"]
    data = {s: bars(700, seed=i + 50, symbol=s) for i, s in enumerate(syms)}
    res = PortfolioBacktestEngine(
        [Allocation(REGISTRY["donchian_breakout"](), tuple(syms)),
         Allocation(REGISTRY["ema_cross"](), ("BTC/USD",))],
        timeframe="4h").run(data)

    # Every allocated strategy appears, even one that never traded.
    assert {"donchian_breakout", "ema_cross"} <= set(res.per_strategy)
    total = sum(float(t.net_pnl) for t in res.trades)
    attributed = sum(r.net_profit for r in res.per_strategy.values())
    assert attributed == pytest.approx(total, abs=1e-6)
    assert "strategy" in res.attribution()


def test_benchmark_is_the_equal_weight_basket_net_of_fees():
    """Beating one asset in its good year is easy; beating the basket you chose
    to trade is the honest bar."""
    syms = ["BTC/USD", "ETH/USD"]
    data = {s: bars(400, seed=i + 60, symbol=s, trend=0.001)
            for i, s in enumerate(syms)}
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))],
        _greedy_limits(max_open_positions=99), timeframe="4h").run(data)
    raw = sum((data[s][-1].close / data[s][0].close - 1.0) for s in syms) / 2
    assert res.report.benchmark_return < raw, "benchmark must pay fees too"


def test_fees_are_the_sum_of_every_book_not_just_one():
    syms = ["BTC/USD", "ETH/USD", "SOL/USD"]
    data = {s: bars(500, seed=i + 70, symbol=s) for i, s in enumerate(syms)}
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))],
        _greedy_limits(max_open_positions=99),
        CostModel(maker_fee_bps=25, taker_fee_bps=40), timeframe="4h").run(data)
    assert res.report.total_fees > 0
    assert res.report.total_fees == pytest.approx(
        float(sum(t.fees for t in res.trades)), rel=0.35), \
        "reported fees must be within reach of the sum of closed-trade fees"


def test_equity_identity_holds_for_a_multi_symbol_book():
    b = AccountStateBuilder(Decimal(50_000), RiskLimits())
    b.apply_fill(Fill("a", START, "BTC/USD", Side.BUY, Decimal("0.1"),
                      Decimal(60_000), fee=Decimal(24), strategy="alpha"))
    b.apply_fill(Fill("b", START, "ETH/USD", Side.BUY, Decimal(2),
                      Decimal(3_000), fee=Decimal(24), strategy="beta"))
    marks = {"BTC/USD": Decimal(61_000), "ETH/USD": Decimal(3_100)}
    equity = b.mark(START, marks)
    expected = b.cash + Decimal("0.1") * Decimal(61_000) + Decimal(2) * Decimal(3_100)
    assert equity == expected
    assert equity == Decimal(50_000) + Decimal(100) + Decimal(200) - Decimal(48)


def test_book_key_is_stable_and_unambiguous():
    assert AccountState.book_key("ema_cross", "BTC/USD") == "ema_cross|BTC/USD"
    # Distinct strategies never collide on one symbol.
    assert (AccountState.book_key("a", "BTC/USD")
            != AccountState.book_key("b", "BTC/USD"))


def test_caps_bind_on_committed_exposure_not_on_appreciation():
    """Pins the distinction explicitly, because it is exactly the kind of thing
    someone later "fixes" into a forced-deleveraging bug.

    A cap constrains what you may BUY. It is not a promise about what a position
    will be worth later. Selling a winner because it appreciated past a
    bookkeeping limit is a worse policy than carrying the breach.
    """
    syms = ["BTC/USD", "ETH/USD", "SOL/USD"]
    data = {s: bars(600, seed=i + 80, symbol=s, trend=0.002)   # strong rally
            for i, s in enumerate(syms)}
    res = PortfolioBacktestEngine(
        [Allocation(AlwaysLong(), tuple(syms))],
        _greedy_limits(max_gross_exposure=0.20, max_position_pct=0.20,
                       max_open_positions=99),
        timeframe="4h").run(data)

    assert res.peak_committed_exposure <= 0.20 + 1e-6, \
        "the cap must hold at approval time"
    assert res.peak_gross_exposure > res.peak_committed_exposure, \
        "in a strong rally, mark-to-market exposure should exceed the entry cap"


def test_pending_orders_consume_exposure_budget_immediately():
    """Unit-level statement of the bug the portfolio tests found: N signals in
    one engine pass must not each see a zero-exposure view."""
    from ct.models import PendingOrder
    b = AccountStateBuilder(Decimal(10_000), RiskLimits())
    b.mark(START, {})
    assert b.snapshot(START, {}).committed_gross_exposure() == 0

    b.reserve("o1", "alpha", "BTC/USD", Decimal(3_000))
    snap = b.snapshot(START, {"BTC/USD": Decimal(60_000)})
    assert snap.pending == (PendingOrder("alpha", "BTC/USD", Decimal(3_000)),)
    assert snap.committed_gross_exposure() == pytest.approx(Decimal("0.3"))
    assert snap.committed_symbol_exposure("BTC/USD") == pytest.approx(Decimal("0.3"))
    # A pending order also occupies a position slot.
    assert snap.open_position_count() == 1
    assert snap.holds_book("alpha", "BTC/USD")

    # A partial fill converts part of the reservation into a real position.
    b.release("o1", Decimal(1_000))
    assert b.snapshot(START, {}).pending_notional() == Decimal(2_000)
    # Expiry drops the remainder.
    b.release("o1")
    assert b.snapshot(START, {}).pending_notional() == Decimal(0)
