"""Risk-engine tests.

The risk engine is the component where a bug costs money, so it gets the
heaviest testing in the project. The property tests at the bottom are the
important ones: they assert invariants that must hold for ANY input, which is
the only way to have confidence in a 16-gate decision tree.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ct.models import AccountState, Position, Regime, Side, Signal
from ct.risk.engine import RejectCode, RiskEngine, RiskLimits
from ct.risk.sizing import (
    fractional_kelly,
    kelly_fraction,
    periods_per_year,
    round_to_increment,
    vol_target_size,
)

TS = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def account(**kw) -> AccountState:
    base = dict(
        ts=TS, equity=Decimal(10_000), cash=Decimal(10_000),
        high_water_mark=Decimal(10_000), positions={},
        marks={"BTC/USD": Decimal(60_000)},
        day_start_equity=Decimal(10_000), week_start_equity=Decimal(10_000),
    )
    base.update(kw)
    return AccountState(**base)


def signal(**kw) -> Signal:
    base = dict(ts=TS, strategy="ema_cross", symbol="BTC/USD", direction=1,
                ref_price=60_000.0, atr=1_500.0, meta={"family": "trend"})
    base.update(kw)
    return Signal(**base)


def book(qty="0.1", strategy="ema_cross", symbol="BTC/USD",
         entry=58_000) -> dict[str, Position]:
    """A positions dict keyed the way the account addresses books.

    `positions` is keyed by "<strategy>|<symbol>", not by symbol, so that two
    strategies trading one symbol cannot flatten each other. Constructing the
    dict by hand in a test is how that contract silently drifts — use this.
    """
    return {AccountState.book_key(strategy, symbol):
            Position(symbol=symbol, strategy=strategy, qty=Decimal(str(qty)),
                     avg_entry=Decimal(entry))}


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(RiskLimits())


# ── happy path ───────────────────────────────────────────────────────────

def test_approves_a_clean_signal(engine):
    d = engine.evaluate(signal(), account(), realized_ann_vol=0.6)
    assert d.approved, d.detail
    assert d.order is not None
    assert d.order.side is Side.BUY
    assert d.order.qty > 0
    assert d.order.attached_stop is not None


def test_risk_budget_sizing_loses_exactly_risk_per_trade_at_the_stop():
    """The defining property of the primary sizer."""
    limits = RiskLimits(risk_per_trade=0.005, default_stop_atr_mult=2.0,
                        max_position_pct=1.0, max_symbol_exposure=1.0,
                        max_gross_exposure=1.0, target_ann_vol=10.0)
    e = RiskEngine(limits)
    equity = Decimal(10_000)
    d = e.evaluate(signal(atr=1_500.0), account(equity=equity, cash=equity),
                   realized_ann_vol=0.6)
    assert d.approved
    stop_distance = Decimal(str(2.0 * 1500.0))
    loss_at_stop = d.order.qty * stop_distance
    expected = equity * Decimal("0.005")
    assert abs(loss_at_stop - expected) / expected < Decimal("0.001")


def test_size_is_the_minimum_of_the_three_sizers(engine):
    d = engine.evaluate(signal(), account(), realized_ann_vol=0.6)
    assert d.approved
    diag = d.diagnostics
    chosen = Decimal(diag[diag["binding_sizer"]])
    for key in ("risk_budget", "vol_target", "notional_cap"):
        assert chosen <= Decimal(diag[key]) + Decimal("1e-12")


# ── the gates, one test each ─────────────────────────────────────────────

def test_kill_switch_blocks_everything(engine):
    d = engine.evaluate(signal(), account(), realized_ann_vol=0.6, kill_switch=True)
    assert not d.approved and d.code is RejectCode.KILL_SWITCH


def test_kill_switch_blocks_exits_too(engine):
    """Deliberate: the kill switch means 'the system is not trusted right now',
    and an exit is still an order sent to an exchange."""
    d = engine.evaluate(signal(direction=0), account(positions=book()),
                        realized_ann_vol=0.6, kill_switch=True)
    assert not d.approved and d.code is RejectCode.KILL_SWITCH


def test_not_armed_blocks_entries_but_never_exits(engine):
    """The asymmetry that keeps a small loss from becoming a large one."""
    entry = engine.evaluate(signal(), account(), realized_ann_vol=0.6, armed=False)
    assert not entry.approved and entry.code is RejectCode.NOT_ARMED

    exit_ = engine.evaluate(signal(direction=0), account(positions=book()),
                            realized_ann_vol=0.6, armed=False)
    assert exit_.approved and exit_.order.reduce_only


def test_daily_drawdown_halts_entries_not_exits(engine):
    acct = account(equity=Decimal(9_600), day_start_equity=Decimal(10_000))
    d = engine.evaluate(signal(), acct, realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.DAILY_DRAWDOWN_HALT

    acct2 = account(equity=Decimal(9_600), day_start_equity=Decimal(10_000),
                    positions=book())
    assert engine.evaluate(signal(direction=0), acct2, realized_ann_vol=0.6).approved


def test_total_drawdown_halt(engine):
    acct = account(equity=Decimal(7_500), high_water_mark=Decimal(10_000),
                   day_start_equity=Decimal(7_500), week_start_equity=Decimal(7_500))
    d = engine.evaluate(signal(), acct, realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.MAX_DRAWDOWN_HALT


def test_consecutive_losses_pause_only_the_offending_strategy(engine):
    acct = account(consecutive_losses={"ema_cross": 5})
    assert engine.evaluate(signal(), acct, realized_ann_vol=0.6).code \
        is RejectCode.CONSECUTIVE_LOSSES
    assert engine.evaluate(signal(strategy="donchian_breakout"), acct,
                           realized_ann_vol=0.6).approved


def test_stale_data_rejected(engine):
    acct = account(data_age_bars={"BTC/USD": 5})
    assert engine.evaluate(signal(), acct, realized_ann_vol=0.6).code \
        is RejectCode.STALE_DATA


def test_shorts_rejected_on_a_spot_account(engine):
    assert engine.evaluate(signal(direction=-1), account(), realized_ann_vol=0.6).code \
        is RejectCode.SHORT_NOT_ALLOWED


def test_no_stop_and_no_atr_cannot_be_sized(engine):
    d = engine.evaluate(signal(atr=None, stop_price=None), account(),
                        realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.NO_STOP


def test_stop_too_tight_rejected(engine):
    """A stop inside 1×ATR is noise, not a risk limit."""
    d = engine.evaluate(signal(stop_price=59_900.0, atr=1_500.0), account(),
                        realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.BAD_STOP


def test_fee_gate_rejects_a_thin_edge(engine):
    """§0.1 in code: a 30 bps target cannot survive a ~90 bps round trip."""
    d = engine.evaluate(signal(meta={"family": "trend", "expected_edge_bps": 30}),
                        account(), realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.EDGE_BELOW_FEES


def test_fee_gate_passes_a_fat_edge(engine):
    d = engine.evaluate(signal(meta={"family": "trend", "expected_edge_bps": 400}),
                        account(), realized_ann_vol=0.6)
    assert d.approved


def test_regime_gate_blocks_reversion_in_a_trend(engine):
    d = engine.evaluate(signal(strategy="bollinger_reversion",
                               meta={"family": "reversion"},
                               regime=Regime.TRENDING),
                        account(), realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.REGIME_BLOCKED


def test_crisis_blocks_every_family(engine):
    for family in ("trend", "reversion", "breakout", "momentum"):
        d = engine.evaluate(signal(meta={"family": family}, regime=Regime.CRISIS),
                            account(), realized_ann_vol=0.6)
        assert not d.approved and d.code is RejectCode.REGIME_BLOCKED


def test_ml_confidence_can_only_veto(engine):
    e = RiskEngine(RiskLimits(min_ml_confidence=0.55))
    low = e.evaluate(signal(meta={"family": "trend", "ml_confidence": 0.4}),
                     account(), realized_ann_vol=0.6)
    assert not low.approved and low.code is RejectCode.ML_CONFIDENCE_LOW

    high = e.evaluate(signal(meta={"family": "trend", "ml_confidence": 0.9}),
                      account(), realized_ann_vol=0.6)
    baseline = engine.evaluate(signal(), account(), realized_ann_vol=0.6)
    assert high.approved
    # High confidence must not produce a LARGER position than no model at all.
    assert high.order.qty <= baseline.order.qty


def test_max_positions_blocks_a_new_symbol_not_an_existing_one():
    e = RiskEngine(RiskLimits(max_open_positions=2))
    held = {
        **book(1, symbol="ETH/USD", entry=3_000),
        **book(10, symbol="SOL/USD", entry=150),
    }
    acct = account(positions=held,
                   marks={"BTC/USD": Decimal(60_000), "ETH/USD": Decimal(3_000),
                          "SOL/USD": Decimal(150)})
    assert e.evaluate(signal(), acct, realized_ann_vol=0.6).code is RejectCode.MAX_POSITIONS


def test_order_rate_limiter(engine):
    assert engine.evaluate(signal(), account(orders_last_hour=30),
                           realized_ann_vol=0.6).code is RejectCode.ORDER_RATE_LIMIT


def test_gross_exposure_clamps_rather_than_rejects():
    e = RiskEngine(RiskLimits(max_gross_exposure=0.30, max_symbol_exposure=0.30,
                              max_position_pct=0.30, risk_per_trade=0.5))
    held = book(1, symbol="ETH/USD", entry=2_500)
    acct = account(positions=held,
                   marks={"BTC/USD": Decimal(60_000), "ETH/USD": Decimal(2_500)})
    d = e.evaluate(signal(), acct, realized_ann_vol=0.6)
    assert d.approved
    # 25% already used of a 30% budget → the new position must fit in 5%.
    assert d.order.qty * Decimal(60_000) <= Decimal("0.0501") * acct.equity


def test_correlation_cluster_cap():
    e = RiskEngine(RiskLimits(max_cluster_exposure=0.20, max_gross_exposure=0.9,
                              max_symbol_exposure=0.9, max_position_pct=0.9,
                              risk_per_trade=0.5))
    held = book(1, symbol="ETH/USD", entry=1_800)
    acct = account(positions=held,
                   marks={"BTC/USD": Decimal(60_000), "ETH/USD": Decimal(1_800)},
                   correlation_clusters={"majors": ["BTC/USD", "ETH/USD"]})
    d = e.evaluate(signal(), acct, realized_ann_vol=0.6)
    assert d.approved
    assert d.order.qty * Decimal(60_000) <= Decimal("0.0201") * acct.equity


def test_below_exchange_minimum_rejected():
    e = RiskEngine(RiskLimits(risk_per_trade=0.00001, min_order_notional=10.0))
    d = e.evaluate(signal(), account(), realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.BELOW_EXCHANGE_MIN


def test_exit_with_no_position_is_rejected(engine):
    d = engine.evaluate(signal(direction=0), account(), realized_ann_vol=0.6)
    assert not d.approved and d.code is RejectCode.SIZE_ZERO


def test_exit_closes_the_whole_position(engine):
    d = engine.evaluate(signal(direction=0), account(positions=book("0.25")),
                        realized_ann_vol=0.6)
    assert d.approved
    assert d.order.qty == Decimal("0.25")
    assert d.order.side is Side.SELL and d.order.reduce_only


# ── sizing unit tests ────────────────────────────────────────────────────

def test_annualization_uses_365_not_252():
    """Guard on the constant that silently mis-sizes every position."""
    assert periods_per_year("1d") == 365.0
    assert periods_per_year("4h") == 365.0 * 6
    with pytest.raises(ValueError):
        periods_per_year("3h")   # must fail loudly, not guess


def test_kelly_is_off_by_default_and_capped_when_on():
    assert fractional_kelly(0.6, 2.0, fraction=0.0) == 1.0
    # Even an absurd Kelly input can only ever return <= 1.0 (a REDUCTION).
    assert fractional_kelly(0.99, 100.0, fraction=1.0) <= 1.0
    # No edge → size to zero.
    assert fractional_kelly(0.2, 0.5, fraction=0.25) == 0.0
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.6 - 0.4 / 2.0)


def test_vol_target_clamped_when_realized_vol_approaches_zero():
    """Without the clamp, a quiet market produces an enormous position right
    before the volatility that ended the quiet period."""
    big = vol_target_size(Decimal(10_000), 60_000.0, realized_ann_vol=1e-9,
                          target_ann_vol=0.2, max_leverage_multiple=3.0)
    assert big <= Decimal(10_000) * 3 / Decimal(60_000) + Decimal("1e-9")


def test_round_to_increment_always_rounds_down():
    assert round_to_increment(Decimal("1.999"), Decimal("0.1")) == Decimal("1.9")
    assert round_to_increment(Decimal("0.05"), Decimal("0.1")) == Decimal("0")


# ── property tests: must hold for ANY input ──────────────────────────────

@pytest.mark.parametrize("equity", [100, 1_000, 10_000, 250_000])
@pytest.mark.parametrize("atr_val", [10.0, 500.0, 1_500.0, 9_000.0])
@pytest.mark.parametrize("vol", [0.1, 0.5, 1.5, 4.0])
def test_property_never_exceeds_notional_cap(equity, atr_val, vol):
    limits = RiskLimits()
    e = RiskEngine(limits)
    acct = account(equity=Decimal(equity), cash=Decimal(equity),
                   high_water_mark=Decimal(equity),
                   day_start_equity=Decimal(equity),
                   week_start_equity=Decimal(equity))
    d = e.evaluate(signal(atr=atr_val), acct, realized_ann_vol=vol)
    if d.approved:
        notional = d.order.qty * Decimal(60_000)
        cap = Decimal(str(limits.max_position_pct)) * Decimal(equity)
        assert notional <= cap * Decimal("1.0001")


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_property_strength_is_monotonic(strength):
    """More conviction must never mean a smaller position."""
    e = RiskEngine(RiskLimits())
    d = e.evaluate(signal(strength=strength), account(), realized_ann_vol=0.6)
    full = e.evaluate(signal(strength=1.0), account(), realized_ann_vol=0.6)
    if d.approved and full.approved:
        assert d.order.qty <= full.order.qty * Decimal("1.0001")


@pytest.mark.parametrize("code_field", [
    "kill_switch", "not_armed", "halted", "stale", "dd", "losses", "rate",
])
def test_property_every_rejection_has_a_machine_readable_code(code_field):
    """A rejection with no code is invisible in the `signals` table, which
    means an unexplained silent strategy."""
    e = RiskEngine(RiskLimits())
    cases = {
        "kill_switch": (signal(), account(), {"kill_switch": True}),
        "not_armed": (signal(), account(), {"armed": False}),
        "halted": (signal(), account(halted=True, halt_reason="x"), {}),
        "stale": (signal(), account(data_age_bars={"BTC/USD": 99}), {}),
        "dd": (signal(), account(equity=Decimal(5_000)), {}),
        "losses": (signal(), account(consecutive_losses={"ema_cross": 9}), {}),
        "rate": (signal(), account(orders_last_hour=999), {}),
    }
    sig, acct, kw = cases[code_field]
    d = e.evaluate(sig, acct, realized_ann_vol=0.6, **kw)
    assert not d.approved
    assert isinstance(d.code, RejectCode)
    assert d.code.value == d.code.value.upper()


def test_property_position_replay_reconstructs_state():
    """The invariant the reconciliation job depends on (ARCHITECTURE.md §6.3):
    replaying fills must reproduce the position exactly."""
    from ct.models import Fill

    pos = Position("BTC/USD")
    fills = [
        Fill("a", TS, "BTC/USD", Side.BUY, Decimal("0.5"), Decimal(60_000)),
        Fill("b", TS, "BTC/USD", Side.BUY, Decimal("0.5"), Decimal(62_000)),
        Fill("c", TS, "BTC/USD", Side.SELL, Decimal("0.4"), Decimal(65_000)),
    ]
    for f in fills:
        pos.apply_fill(f)
    assert pos.qty == Decimal("0.6")
    assert pos.avg_entry == Decimal(61_000)      # (0.5*60k + 0.5*62k)/1.0

    replay = Position("BTC/USD")
    for f in fills:
        replay.apply_fill(f)
    assert replay.qty == pos.qty and replay.avg_entry == pos.avg_entry
    assert replay.realized_pnl == pos.realized_pnl


def test_position_flip_resets_the_entry_price():
    """The case naive implementations corrupt: a fill larger than the open
    position, reversing it."""
    from ct.models import Fill

    pos = Position("BTC/USD")
    pos.apply_fill(Fill("a", TS, "BTC/USD", Side.BUY, Decimal(1), Decimal(60_000)))
    realized = pos.apply_fill(
        Fill("b", TS, "BTC/USD", Side.SELL, Decimal(3), Decimal(65_000)))
    assert realized == Decimal(5_000)            # closed 1 unit for +5k
    assert pos.qty == Decimal(-2)                # residual is a new short
    assert pos.avg_entry == Decimal(65_000)      # at the fill price, not 60k
