"""Strategy lifecycle FSM and allocator — Phase 2 item 7.

Two defects motivate this file, and the tests that matter are the ones pinning
them shut:

1. **A restart resurrected a quarantined strategy.** State lived in
   `RuntimeState.paused_strategies`, a `set` rebuilt on every boot, and nothing
   wrote the `strategies` table. With `restart: unless-stopped`, a strategy
   stopped for crashing came back on the next restart.
2. **Scarce capacity went to whichever strategy was listed first.** Caps bind on
   committed exposure and signals were considered in allocation order, so the
   backtest result depended on how the allocation list was typed.

The FSM's asymmetry is the design and is tested directly: **transitions that
reduce risk are automatic; transitions that increase it need a person.**
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ct.risk.allocator import Allocator, Weights, returns_by_strategy
from ct.strategies.lifecycle import (
    Actor,
    IllegalTransition,
    StrategyLifecycle,
    StrategyState,
)

NOW = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)


def _lc(state=StrategyState.PAPER, name="ema_cross") -> StrategyLifecycle:
    return StrategyLifecycle(name=name, state=state)


# ── the asymmetry ────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", [StrategyState.PAPER, StrategyState.LIVE])
def test_the_engine_may_always_stop_a_strategy(state):
    """Reductions never wait for permission — the same rule as disarming."""
    assert _lc(state).quarantine("raised on BTC/USD", now=NOW).state \
        is StrategyState.QUARANTINED
    assert _lc(state).pause("5 consecutive losses", now=NOW).state \
        is StrategyState.PAUSED
    assert _lc(state).to(StrategyState.RETIRED, actor=Actor.ENGINE,
                         reason="gate regressed", now=NOW).state \
        is StrategyState.RETIRED


def test_the_engine_may_never_start_one():
    """Every risk-increasing transition is a human's."""
    for start, target in [
        (StrategyState.PROPOSED, StrategyState.PAPER),
        (StrategyState.PAPER, StrategyState.LIVE),
        (StrategyState.QUARANTINED, StrategyState.PAPER),
        (StrategyState.QUARANTINED, StrategyState.LIVE),
        (StrategyState.RETIRED, StrategyState.PAPER),
    ]:
        with pytest.raises(IllegalTransition, match=r"may move|not a legal"):
            _lc(start).to(target, actor=Actor.ENGINE, now=NOW)
        # ...and the same move is fine when a person makes it.
        assert _lc(start).to(target, actor=Actor.HUMAN, now=NOW).state is target


def test_a_crash_needs_a_human_to_clear():
    """The point of a quarantine is that somebody reads the traceback."""
    lc = _lc().quarantine("boom", now=NOW)
    assert not lc.may_run
    with pytest.raises(IllegalTransition):
        lc.to(StrategyState.PAPER, actor=Actor.ENGINE, now=NOW)
    lc.to(StrategyState.PAPER, actor=Actor.HUMAN, reason="fixed", now=NOW)
    assert lc.may_run


def test_an_illegal_transition_raises_rather_than_being_ignored():
    """Silently ignoring it is how a quarantined strategy keeps trading."""
    with pytest.raises(IllegalTransition, match="not a legal transition"):
        _lc(StrategyState.PROPOSED).to(StrategyState.LIVE, actor=Actor.HUMAN)


def test_a_transition_to_the_current_state_is_not_an_error():
    lc = _lc(StrategyState.PAPER)
    assert lc.to(StrategyState.PAPER, actor=Actor.ENGINE).state \
        is StrategyState.PAPER


# ── pauses are time-boxed and heal themselves ────────────────────────────


def test_a_pause_expires_back_to_what_it_was_doing():
    """`live` must come back as `live`, not as `paper` — guessing would demote
    a promoted strategy every time its loss breaker tripped."""
    lc = _lc(StrategyState.LIVE).pause("loss breaker", hours=24, now=NOW)
    assert lc.state is StrategyState.PAUSED
    assert not lc.resume_due(NOW + timedelta(hours=23))
    lc.resume_if_due(NOW + timedelta(hours=23))
    assert lc.state is StrategyState.PAUSED

    lc.resume_if_due(NOW + timedelta(hours=25))
    assert lc.state is StrategyState.LIVE


def test_a_pause_does_not_expire_a_quarantine():
    lc = _lc().quarantine("boom", now=NOW)
    lc.resume_if_due(NOW + timedelta(days=365))
    assert lc.state is StrategyState.QUARANTINED


def test_only_running_states_run():
    assert _lc(StrategyState.PAPER).may_run
    assert _lc(StrategyState.LIVE).may_run
    for state in (StrategyState.PROPOSED, StrategyState.PAUSED,
                  StrategyState.QUARANTINED, StrategyState.RETIRED):
        assert not _lc(state).may_run


def test_live_state_is_not_permission_to_trade_real_money():
    """`live` is a statement about the promotion gate. Actually trading needs
    QD_MODE=live and a human-rooted arming row; this FSM is not on that path."""
    assert _lc(StrategyState.LIVE).may_trade_live
    assert not _lc(StrategyState.PAPER).may_trade_live


def test_demotion_is_always_available_to_the_engine():
    lc = _lc(StrategyState.LIVE).demote("promotion gate regressed", now=NOW)
    assert lc.state is StrategyState.PAPER


def test_history_records_who_did_what():
    lc = _lc().pause("losses", now=NOW)
    lc.resume_if_due(NOW + timedelta(hours=25))
    lc.quarantine("boom", now=NOW + timedelta(hours=26))
    actors = [h[3] for h in lc.history]
    assert actors == ["engine", "engine", "engine"]
    assert lc.history[-1][2] is StrategyState.QUARANTINED


# ── durability: the defect that started this ─────────────────────────────


class _FakeDB:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        return self.rows

    async def fetchrow(self, sql, *args):
        return self.rows[0] if self.rows else None

    def wrote(self, needle: str):
        return [c for c in self.executed if needle in " ".join(c[0].split())]


def test_a_quarantine_is_written_to_the_database():
    """Without this row the stop lives only in memory, and a restart undoes
    it. `restart: unless-stopped` makes that routine, not exceptional."""
    from ct.store.strategies import StrategyRepository

    db = _FakeDB()
    lc = _lc().quarantine("raised on BTC/USD", now=NOW)
    asyncio.run(StrategyRepository(db).save(lc, actor=Actor.ENGINE))

    update = db.wrote("UPDATE strategies")
    assert len(update) == 1
    assert update[0][1][1] == "quarantined"
    assert update[0][1][4] is False, "a quarantined strategy stayed enabled"
    assert db.wrote("INSERT INTO audit_log"), "the stop was not audited"


def test_registering_a_strategy_does_not_reset_its_state():
    """A redeploy re-registers everything. If that cleared `status`, it would
    silently un-quarantine the one strategy somebody stopped on purpose."""
    from ct.store.strategies import StrategyRepository

    db = _FakeDB()
    asyncio.run(StrategyRepository(db).upsert(
        "ema_cross", class_path="ct.strategies.trend", params={"fast": 20}))
    sql = " ".join(db.wrote("INSERT INTO strategies")[0][0].split())
    conflict = sql.split("DO UPDATE SET", 1)[1]
    for protected in ("status", "enabled", "status_reason"):
        assert protected not in conflict, \
            f"registration overwrites {protected}"


def test_an_unknown_status_is_treated_as_needing_a_human():
    """The schema's original vocabulary was active/quarantined/paused. Anything
    unrecognised must not start trading."""
    from ct.store.strategies import _from_row

    lc = _from_row({"name": "x", "status": "active", "status_reason": None,
                    "paused_until": None, "updated_at": NOW})
    assert lc.state is StrategyState.QUARANTINED
    assert not lc.may_run


def test_params_hash_is_order_independent():
    from ct.store.strategies import params_hash

    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


# ── the engine honours durable state ─────────────────────────────────────


def _engine_with_lifecycles(rows):
    """A TradingEngine whose strategy repository returns `rows`."""
    from ct.engine import Allocation, TradingEngine
    from ct.events import NullEventBus
    from ct.execution.broker import PaperBroker
    from ct.strategies import REGISTRY

    class _Repo:
        def __init__(self):
            self.saved: list[StrategyLifecycle] = []
            self.registered: list[str] = []

        async def load_all(self):
            return dict(rows)

        async def upsert(self, name, **kw):
            self.registered.append(name)

        async def save(self, lc, *, actor):
            self.saved.append(lc)

    engine = TradingEngine(
        [Allocation(REGISTRY["ema_cross"](), ("BTC/USD",))], PaperBroker(),
        db=object(), initial_equity=Decimal(10_000),
        bus=NullEventBus(source="test"), clock=lambda: NOW)
    repo = _Repo()
    engine.strategies_repo = repo
    return engine, repo


def test_a_quarantined_strategy_stays_stopped_across_a_restart():
    """The regression this whole item exists for."""
    engine, _ = _engine_with_lifecycles(
        {"ema_cross": _lc(StrategyState.QUARANTINED)})
    asyncio.run(engine._load_lifecycles())
    assert "ema_cross" in engine.state.paused_strategies


def test_a_strategy_the_table_has_never_seen_does_not_run():
    """Appearing in QD_STRATEGIES is not permission to trade. It is registered
    as `proposed` and waits for a person."""
    engine, repo = _engine_with_lifecycles({})
    asyncio.run(engine._load_lifecycles())
    assert "ema_cross" in engine.state.paused_strategies
    assert repo.registered == ["ema_cross"]


def test_a_paper_strategy_runs():
    engine, _ = _engine_with_lifecycles({"ema_cross": _lc(StrategyState.PAPER)})
    asyncio.run(engine._load_lifecycles())
    assert "ema_cross" not in engine.state.paused_strategies


def test_an_expired_pause_is_resumed_and_persisted_on_boot():
    lc = _lc(StrategyState.PAPER).pause("losses", hours=1,
                                        now=NOW - timedelta(hours=5))
    engine, repo = _engine_with_lifecycles({"ema_cross": lc})
    asyncio.run(engine._load_lifecycles())
    assert lc.state is StrategyState.PAPER
    assert "ema_cross" not in engine.state.paused_strategies
    assert repo.saved, "the resume was not written back"


def test_a_lifecycle_load_failure_stops_everything():
    """Fail closed. Not knowing whether a strategy is quarantined is not a
    reason to run it."""
    engine, repo = _engine_with_lifecycles({})

    async def _boom():
        raise ConnectionError("database is gone")

    repo.load_all = _boom
    asyncio.run(engine._load_lifecycles())
    assert "ema_cross" in engine.state.paused_strategies


# ── allocator: weights ───────────────────────────────────────────────────


def test_weights_fall_back_to_equal_and_say_so():
    """Three trades is not a variance estimate. The label matters as much as
    the number: a report must never imply a risk model that did not run."""
    w = Allocator().weights({"a": [0.01, -0.02, 0.03], "b": [0.01, 0.01, 0.0]})
    assert w.method == "equal"
    assert w.weights == {"a": 0.5, "b": 0.5}
    assert "3 observations" in w.reasons["a"]


def test_the_quieter_strategy_gets_the_bigger_weight():
    calm = [0.01, -0.01] * 15
    wild = [0.10, -0.10] * 15
    w = Allocator().weights({"calm": calm, "wild": wild})
    assert w.method == "inverse_vol"
    assert w.of("calm") > w.of("wild")
    assert abs(sum(w.weights.values()) - 1.0) < 1e-9


def test_no_strategy_takes_more_than_the_cap():
    """An inverse-vol weight is a reciprocal, so the flattest sample — the
    estimator's least reliable case — would otherwise take nearly everything.
    Measured without the cap: 0.996 of a two-strategy book."""
    almost_flat = [0.0001, -0.0001] * 15
    normal = [0.05, -0.05] * 15
    alloc = Allocator()
    w = alloc.weights({"flat": almost_flat, "normal": normal})
    assert w.of("flat") <= alloc._cap_for(2) + 1e-9
    assert abs(sum(w.weights.values()) - 1.0) < 1e-9


def test_capping_does_not_collapse_the_book_to_equal_weight():
    """A flat 0.5 cap would force exact equality on two strategies, erasing
    the very ordering the weighting exists to express."""
    w = Allocator().weights({"calm": [0.01, -0.01] * 15,
                             "wild": [0.10, -0.10] * 15})
    assert w.of("calm") != pytest.approx(w.of("wild"))


def test_the_cap_tightens_as_the_book_grows():
    alloc = Allocator()
    assert alloc._cap_for(2) > alloc._cap_for(10)
    assert alloc._cap_for(10) == pytest.approx(0.3)


def test_a_strategy_with_no_history_is_not_given_zero():
    """Absence of an estimate is not evidence a strategy deserves nothing."""
    w = Allocator().weights({"measured": [0.02, -0.02] * 15, "new": [0.01]})
    assert w.method == "mixed"
    assert w.of("new") > 0.0


def test_weights_of_nothing_is_not_a_crash():
    assert Allocator().weights({}).weights == {}


def test_returns_by_strategy_reads_closed_trades():
    class _T:
        def __init__(self, strategy, r):
            self.strategy = strategy
            self.return_pct = r

    out = returns_by_strategy([_T("a", 0.1), _T("b", -0.2), _T("a", 0.3)])
    assert out == {"a": [0.1, 0.3], "b": [-0.2]}


# ── allocator: arbitration ───────────────────────────────────────────────


class _Sig:
    def __init__(self, strategy, symbol, edge=None, strength=1.0):
        self.strategy = strategy
        self.symbol = symbol
        self.strength = strength
        self.meta = {} if edge is None else {"expected_edge_bps": edge}


def test_the_strongest_signal_is_considered_first():
    signals = [_Sig("weak", "BTC/USD", edge=20),
               _Sig("strong", "ETH/USD", edge=300),
               _Sig("middling", "SOL/USD", edge=90)]
    order = [s.strategy for s in Allocator().rank(signals)]
    assert order == ["strong", "middling", "weak"]


def test_ranking_does_not_depend_on_input_order():
    """The artefact this fixes: capacity used to go to whoever was listed
    first, so reordering the allocation list changed the backtest."""
    signals = [_Sig("a", "BTC/USD", edge=50), _Sig("b", "ETH/USD", edge=150),
               _Sig("c", "SOL/USD", edge=100)]
    forward = [s.strategy for s in Allocator().rank(signals)]
    backward = [s.strategy for s in Allocator().rank(list(reversed(signals)))]
    assert forward == backward == ["b", "c", "a"]


def test_ties_break_deterministically_not_by_input_order():
    signals = [_Sig("zeta", "BTC/USD", edge=100),
               _Sig("alpha", "ETH/USD", edge=100)]
    assert [s.strategy for s in Allocator().rank(signals)] == ["alpha", "zeta"]
    assert [s.strategy for s in Allocator().rank(list(reversed(signals)))] \
        == ["alpha", "zeta"]


def test_a_signal_with_no_edge_estimate_sorts_last_but_is_not_dropped():
    """Keeping it visible: an un-gated strategy should be starved of priority,
    not silently removed from the pass."""
    signals = [_Sig("silent", "BTC/USD"), _Sig("loud", "ETH/USD", edge=10)]
    order = Allocator().rank(signals)
    assert [s.strategy for s in order] == ["loud", "silent"]
    assert len(order) == 2


def test_weights_scale_the_ranking():
    signals = [_Sig("small", "BTC/USD", edge=100),
               _Sig("big", "ETH/USD", edge=60)]
    w = Weights({"small": 0.1, "big": 0.9}, "inverse_vol")
    assert [s.strategy for s in Allocator().rank(signals, weights=w)] \
        == ["big", "small"]


def test_an_unweighted_strategy_ranks_on_its_own_edge():
    """Absent from the map means no opinion, not worthless — otherwise a
    strategy the weighting run had not seen would be silently starved."""
    signals = [_Sig("known", "BTC/USD", edge=10),
               _Sig("unknown", "ETH/USD", edge=500)]
    w = Weights({"known": 1.0}, "inverse_vol")
    ranked = Allocator().rank(signals, weights=w)
    assert ranked[0].strategy == "unknown"


def test_an_explicit_zero_weight_does_sort_last():
    signals = [_Sig("off", "BTC/USD", edge=500),
               _Sig("on", "ETH/USD", edge=10)]
    w = Weights({"off": 0.0, "on": 1.0}, "inverse_vol")
    assert [s.strategy for s in Allocator().rank(signals, weights=w)] \
        == ["on", "off"]


def test_rank_accepts_a_key_for_wrapped_items():
    """The portfolio engine ranks `(signal, alloc, series, index)` tuples."""
    items = [( _Sig("a", "BTC/USD", edge=10), "ctx-a"),
             ( _Sig("b", "ETH/USD", edge=99), "ctx-b")]
    order = Allocator().rank(items, key=lambda item: item[0])
    assert [i[0].strategy for i in order] == ["b", "a"]


# ── the artefact, at the level it actually mattered ──────────────────────


def _competing(name: str, edge_bps: float):
    """A strategy that signals on every flat bar with a fixed expected edge.

    Real strategies signal rarely and by design (§0.1), so a realistic scenario
    almost never puts two of them in the same pass — the first version of this
    test produced contention on 2 passes out of 118 and passed with arbitration
    disabled, proving nothing. Forcing contention every bar is what makes the
    order-dependence observable at all.
    """
    from ct.models import Signal
    from ct.strategies.base import Strategy

    class _Competing(Strategy):
        DEFAULTS: dict = {}

        @property
        def warmup(self) -> int:
            return 5

        def on_bar(self, ctx) -> list[Signal]:
            if ctx.index < self.warmup or not ctx.position.is_flat:
                return []
            px = ctx.price
            return [self._signal(ctx, 1, atr=px * 0.02, stop=px * 0.95,
                                 target=px * 1.15,
                                 expected_edge_bps=edge_bps)]

    _Competing.name = name
    _Competing.family = "trend"
    return _Competing()


def _contended_run(order: list[tuple[str, float]]):
    """One position slot, three strategies signalling every bar."""
    import math

    from ct.backtest.portfolio import Allocation, PortfolioBacktestEngine
    from ct.models import Bar
    from ct.risk.engine import RiskLimits

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    px, x, rows = 30_000.0, 7, []
    for i in range(300):
        x = (1103515245 * x + 12345) % (2**31)
        u = x / 2**31
        px *= math.exp(0.0006 + 0.021 * (u - 0.5))
        hi, lo = px * (1 + 0.011 * u), px * (1 - 0.011 * (1 - u))
        op = px * (1 + 0.003 * (u - 0.5))
        rows.append(Bar(ts=t0 + timedelta(hours=4 * i), open=op,
                        high=max(hi, op, px), low=min(lo, op, px), close=px,
                        volume=200.0, symbol="BTC/USD", timeframe="4h"))

    allocations = [Allocation(_competing(n, e), ("BTC/USD",), weight=1 / 3)
                   for n, e in order]
    # One slot. Whoever is considered first takes it, so the outcome is a pure
    # readout of the arbitration order.
    limits = RiskLimits(max_open_positions=1)
    return PortfolioBacktestEngine(allocations, limits,
                                   initial_equity=50_000.0).run({"BTC/USD": rows})


# Order matters to the test itself: the FIRST entry must be a signal that
# would actually be accepted, or the fee gate rejects it and the scarce slot
# falls to the next one regardless of arbitration — which is how the first
# version of this test passed with arbitration disabled. `weak` at 40 bps is
# below `min_edge_bps` (60) and is there to be refused; `middling` leads
# because it is viable and inferior, which is the adversarial case.
CONTENDERS = [("middling", 120.0), ("strong", 400.0), ("weak", 40.0)]


def test_capacity_goes_to_the_best_signal_not_the_first_listed():
    """The artefact this item exists to remove."""
    result = _contended_run(CONTENDERS)
    traded = {t.strategy for t in result.trades}
    assert traded == {"strong"}, (
        f"the scarce slot went to {traded}, not to the strongest signal")


def test_a_portfolio_backtest_does_not_depend_on_allocation_order():
    """Retyping the allocation list in another order must not change the run.

    Verified to fail with arbitration disabled — see the session log; a test
    that passes either way would prove nothing.
    """
    forward = _contended_run(CONTENDERS)
    backward = _contended_run(list(reversed(CONTENDERS)))

    assert forward.report.n_trades == backward.report.n_trades
    assert forward.equity_curve == backward.equity_curve, (
        "the backtest still depends on how the allocation list was typed")
    assert forward.rejections == backward.rejections
    assert ({t.strategy for t in forward.trades}
            == {t.strategy for t in backward.trades})
