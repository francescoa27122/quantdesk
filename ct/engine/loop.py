"""`TradingEngine` — the loop.

The single writer of orders, fills and positions (ARCHITECTURE.md §1). Every
piece it needs already existed and was tested in isolation; this assembles them
and owns the ordering between them, which is where the remaining risk lives.

## Boot order is not arbitrary

    connect → load books → re-query open orders → reconcile → THEN maybe trade

**The engine boots halted and stays halted until reconciliation passes clean.**
Disagreeing with the exchange about what you own invalidates every risk gate at
once, because all of them are computed from a position number. Trading first and
checking later is not a shortcut, it is trading blind.

## Each cycle, in this order and for these reasons

    1. kill switch      — cheapest, most absolute; nothing proceeds past it
    2. control commands — a halt queued a second ago must land before any order
    3. market data      — new closed bars, and how stale the feed is
    4. protective exits — stops before entries, always: a stop that fires this
                          bar must not be delayed behind a new position
    5. strategies       — propose
    6. risk             — size or veto
    7. execution        — submit; persist first
    8. fills            — record atomically with their notification
    9. mark to market   — equity, drawdown, high-water mark
   10. housekeeping     — arming renewal, outbox drain, heartbeat

## What it does NOT do

No live broker. `Broker` is injected and `PaperBroker` is what runs today;
`CcxtBroker` is a separate deliverable behind the arming ceremony. Nothing in
this file can place a real order, and that is by design rather than by omission.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ct.data.integrity import timeframe_seconds
from ct.engine.control import ControlPlane, KillSwitch, RuntimeState
from ct.engine.execution import OrderExecutor
from ct.events import (
    EventBus,
    NullEventBus,
    heartbeat,
    new_correlation_id,
    portfolio_updated,
    risk_approved,
    risk_limit_exceeded,
    signal_generated,
    signal_rejected,
)
from ct.events.types import Event, EventType, Severity
from ct.execution.broker import Broker
from ct.models import (
    AccountState,
    Bar,
    ExitReason,
    Order,
    OrderType,
    Regime,
    Side,
    Signal,
    TimeInForce,
)
from ct.risk.account import AccountStateBuilder
from ct.risk.engine import RiskEngine, RiskLimits
from ct.strategies.base import Context, Strategy
from ct.strategies.lifecycle import Actor as _ActorEnum

if TYPE_CHECKING:
    # These are imported lazily at their use sites to keep the module's import
    # graph acyclic; the annotations need them at check time only, and
    # `from __future__ import annotations` keeps them as strings at runtime.
    from ct.data.repository import CandleRepository
    from ct.store import (
        ArmingRepository,
        Database,
        NotificationOutbox,
        OrderRepository,
        PositionRepository,
        ReconciliationService,
        StrategyRepository,
    )
    from ct.strategies.features import FeatureSet

log = logging.getLogger(__name__)


_ENGINE = _ActorEnum.ENGINE


class Allocation:
    """A strategy's mandate. Mirrors the portfolio backtester's `Allocation`
    so a configuration that was validated there means the same thing here."""

    __slots__ = ("strategy", "symbols", "timeframe", "weight")

    def __init__(self, strategy: Strategy, symbols: tuple[str, ...],
                 timeframe: str = "4h", weight: float = 1.0) -> None:
        if not symbols:
            raise ValueError(f"{strategy.name}: no symbols allocated")
        timeframe_seconds(timeframe)
        self.strategy = strategy
        self.symbols = symbols
        self.timeframe = timeframe
        self.weight = weight


class TradingEngine:
    def __init__(
        self,
        allocations: list[Allocation],
        broker: Broker,
        *,
        db: Database | None = None,
        account_id: int = 1,
        limits: RiskLimits | None = None,
        bus: EventBus | None = None,
        kill_switch: KillSwitch | None = None,
        initial_equity: Decimal = Decimal(10_000),
        mode: str = "paper",
        correlation_clusters: dict[str, list[str]] | None = None,
        cycle_seconds: float = 20.0,
        history_bars: int = 400,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.allocations = allocations
        self.broker = broker
        self.db = db
        self.account_id = account_id
        self.limits = limits or RiskLimits()
        self.bus = bus if bus is not None else NullEventBus(source="engine")
        self.kill = kill_switch or KillSwitch()
        self.mode = mode
        self.correlation_clusters = correlation_clusters or {}
        self.cycle_seconds = cycle_seconds
        # How much history each strategy sees. Must exceed the deepest warmup or
        # nothing ever fires; +50 gives indicators room to stabilise.
        self.history_bars = max(history_bars,
                                max((a.strategy.warmup for a in allocations),
                                    default=0) + 50)
        self._now = clock or (lambda: datetime.now(timezone.utc))

        self.state = RuntimeState(mode=mode)
        self.acct = AccountStateBuilder(initial_equity, self.limits)
        self.risk = RiskEngine(self.limits)

        # Wired in `boot()` when a database is present. The engine runs without
        # one for tests, but then nothing survives a restart.
        self.candles: CandleRepository | None = None
        self.orders: OrderRepository | None = None
        self.positions_repo: PositionRepository | None = None
        self.executor: OrderExecutor | None = None
        self.control: ControlPlane | None = None
        self.arming: ArmingRepository | None = None
        self.outbox: NotificationOutbox | None = None
        self.reconciler: ReconciliationService | None = None
        self.strategies_repo: StrategyRepository | None = None
        # Durable lifecycle state, loaded on boot. Empty means "no database",
        # in which case the in-memory set is all there is — honest, but it does
        # not survive a restart.
        self.lifecycles: dict[str, Any] = {}

        self._last_bar: dict[tuple[str, str], datetime] = {}
        self._features: dict[tuple[Any, ...], FeatureSet] = {}
        # Realized annualised vol per symbol, refreshed on every new closed
        # bar. Mirrors BacktestEngine._realized_ann_vol exactly — same window,
        # same EWMA, same annualisation — so live sizing and backtested sizing
        # are the same number. Until 2026-07-30 this was a constant 0.5 and
        # the vol-target sizer was not actually vol-targeting.
        self._vols: dict[str, float] = {}
        self.vol_lookback = 30
        # UTC date of the last promotion evaluation, so housekeeping runs it
        # once per day rather than every 20 seconds.
        self._last_promotion_eval: date | None = None
        self._correlations: dict[str, str] = {}
        self._legs: dict[str, str] = {}
        # Which names are paused BECAUSE of lifecycle state, as opposed to a
        # transient `pause_strategy` control command. The lifecycle refresh may
        # only unblock names it blocked — otherwise an external enable would
        # silently cancel an operator's pause.
        self._lifecycle_blocked: set[str] = set()
        # Books a flatten order has already been submitted for, so a slow fill
        # does not attract a duplicate close every cycle.
        self._flatten_submitted: set[str] = set()
        self._stopping = asyncio.Event()
        self.cycles = 0
        self.signals_seen = 0
        self.orders_placed = 0
        # Degradation accounting. A cycle that completed with a failing step
        # did less than it looks like it did, and that must be visible rather
        # than inferred from the logs.
        self.step_failures = 0
        self.degraded_cycles = 0
        self._degraded_this_cycle = False
        # ~5 cycles is 100 s at the default cadence: long enough to ride out a
        # container restart of qd-db, short enough that nothing opens a new
        # position while blind.
        self.max_degraded_cycles = 5
        # Set by run_forever while APScheduler is driving. None means nobody is
        # scheduling this engine — the test harness and anything embedding it
        # call `cycle()` directly — which `_housekeeping` uses to decide whether
        # it still has to run the daily promotion evaluation itself.
        self._scheduler: Any | None = None
        # Identifies this engine to the module-level job functions. Per account,
        # because that is the only thing that could ever legitimately give one
        # process two engines.
        self.scheduler_key = f"account:{self.account_id}"

    # ── boot ─────────────────────────────────────────────────────────────

    async def boot(self, *, exchange_balances: dict[str, Decimal] | None = None,
                   ) -> bool:
        """Prepare and decide whether trading may start. Returns `reconciled`."""
        self.bus.publish(Event(
            EventType.ENGINE_STARTED,
            {"component": "engine", "mode": self.mode,
             "strategies": [a.strategy.name for a in self.allocations],
             "can_lose_money": self.broker.can_lose_money},
            source="engine", severity=Severity.INFO))
        # Logged at every boot so "I thought it was on paper" is not something
        # anyone can say afterwards.
        log.info("engine booting in %s mode; broker can_lose_money=%s",
                 self.mode, self.broker.can_lose_money)

        if self.db is not None:
            await self._wire_repositories()
            # `_wire_repositories` sets all of these unconditionally; the
            # assert states that coupling rather than leaving each use site to
            # trip over `None`. It fails loudly here — on the boot path, before
            # anything can trade — instead of as an AttributeError mid-cycle.
            assert self.positions_repo is not None
            assert self.executor is not None
            assert self.reconciler is not None
            books = await self.positions_repo.load(self.account_id)
            self.acct.positions.update(books)
            log.info("loaded %d position books", len(books))

            # Lifecycle BEFORE anything can trade. A quarantine that a restart
            # forgets is not a quarantine, and `restart: unless-stopped` makes
            # restarts routine.
            await self._load_lifecycles()

            # Re-query anything in flight BEFORE placing anything new.
            outcome = await self.executor.reconcile_open_orders()
            if outcome:
                log.info("re-queried %d open orders: %s", len(outcome), outcome)

            result = await self.reconciler.run(self.account_id, exchange_balances)
            self.state.reconciled = result.ok
            if not result.ok:
                self.state.halt("reconciliation mismatch")
                await self._risk_event("RECONCILE_MISMATCH", result.summary(),
                                       discrepancies=[str(d) for d
                                                      in result.discrepancies])
                for d in result.discrepancies:
                    log.error("  %s", d)
                return False
        else:
            # No database: nothing to disagree with. Honest, but it also means
            # nothing survives a restart.
            self.state.reconciled = True

        self.state.resume()
        await self._refresh_arming()
        return True

    async def _wire_repositories(self) -> None:
        from ct.data.repository import CandleRepository
        from ct.store import (
            ArmingRepository,
            NotificationOutbox,
            OrderRepository,
            PositionRepository,
            ReconciliationService,
        )
        # Bound once: every caller reaches this behind `if self.db is not None`,
        # and repeating that check eight times would say less than one assert.
        db = self.db
        assert db is not None, "_wire_repositories called without a database"
        self.candles = CandleRepository(db)
        self.orders = OrderRepository(db)
        self.positions_repo = PositionRepository(db)
        self.control = ControlPlane(db, self.account_id)
        self.arming = ArmingRepository(db)
        self.outbox = NotificationOutbox(db)
        self.reconciler = ReconciliationService(db)
        from ct.store import StrategyRepository
        self.strategies_repo = StrategyRepository(db)
        self.executor = OrderExecutor(self.broker, self.orders,
                                      self.positions_repo, self.bus,
                                      account_id=self.account_id)

    # ── the loop ─────────────────────────────────────────────────────────

    async def cycle_guarded(self) -> None:
        """`cycle()` with the failure contract that used to live in the loop.

        Extracted unchanged so the scheduler drives exactly what the `while`
        loop drove. One bad cycle must not kill the engine — but it must be
        loud, and it must not be swallowed by APScheduler's executor, which
        would log it under its own logger and publish nothing.
        """
        try:
            await self.cycle()
        except Exception:
            log.exception("cycle failed")
            self.bus.publish(Event(
                EventType.RISK_LIMIT_EXCEEDED,
                {"code": "CYCLE_ERROR", "detail": "see logs"},
                severity=Severity.CRITICAL, source="engine"))

    async def evaluate_promotion(self, now: datetime) -> None:
        """Daily promotion evaluation, moved out of `_housekeeping`.

        The date guard stays: the scheduler coalesces missed fires, and a
        restart can re-fire the day's job, so this must be idempotent rather
        than merely infrequent.
        """
        if self.db is None or self._last_promotion_eval == now.date():
            return
        self._last_promotion_eval = now.date()
        try:
            from ct.engine import promotion
            await promotion.evaluate(self.db, now=now)
        except Exception:
            log.exception("promotion evaluation failed; next attempt tomorrow "
                          "(the gate reads stale rows until then)")

    async def run_forever(self) -> None:
        """Boot, then let APScheduler drive the cycle (ARCHITECTURE.md Merge 5).

        The engine still owns its own shutdown: `stop()` sets `_stopping`, and
        this returns as soon as it is set. The scheduler is torn down here
        rather than in a signal handler so that a cycle already in flight is
        allowed to finish — killing one mid-way could leave an order placed and
        unrecorded.
        """
        from ct.config import settings
        from ct.engine import scheduler as sched

        if not await self.boot():
            log.error("engine did not reconcile; staying halted")

        cfg = settings()
        scheduler, jobstore = sched.build_scheduler(dsn=cfg.db_dsn or None)
        sched.register_engine(self.scheduler_key, self)
        # start() BEFORE add_job(): additions made to a stopped scheduler sit in
        # _pending_jobs, where replace_existing does not deduplicate them — so a
        # second registration would give the engine two trading cycles, and with
        # a persistent jobstore the duplicate outlives the process.
        #
        # start_scheduler, not scheduler.start(): a SQLAlchemy job store does not
        # connect until started, so an unreachable database surfaces here and
        # must degrade to an in-memory schedule rather than stop the engine.
        scheduler, jobstore = sched.start_scheduler(scheduler, jobstore)
        sched.schedule_engine_jobs(
            scheduler, self.scheduler_key,
            cycle_seconds=self.cycle_seconds,
            misfire_grace_seconds=cfg.job_misfire_grace_seconds,
            with_promotion=self.db is not None)
        self._scheduler = scheduler
        log.info("scheduler started", extra={
            "jobstore": jobstore,
            "cycle_seconds": self.cycle_seconds,
            "jobs": sched.describe_jobs(scheduler)})

        # The old loop ran a cycle immediately and then slept; an interval
        # trigger waits one interval first. Keep the original behaviour — the
        # first cycle after boot is the one that reconciles against reality.
        await self.cycle_guarded()

        # Evaluate promotion once at boot as well as on the daily schedule.
        # `_housekeeping` used to do this on the FIRST cycle, so a fresh deploy
        # had a promotion record within seconds. Moving it to a cron job alone
        # regressed that to "whenever 00:05 UTC next comes round", leaving the
        # arm-live gate reading "never evaluated" for up to a day after every
        # deploy. Verified against a real database: promotion_status stayed
        # empty after boot until this was added. The date guard inside makes it
        # idempotent, so this costs nothing when the daily job has already run.
        if self.db is not None:
            await self.evaluate_promotion(self._now())

        try:
            await self._stopping.wait()
        finally:
            # wait=True: let an in-flight cycle finish. shutdown() below then
            # cancels resting orders with a consistent view.
            with contextlib.suppress(Exception):
                scheduler.shutdown(wait=True)
            sched.unregister_engine(self.scheduler_key)
            self._scheduler = None
            log.info("scheduler stopped")

    async def _step(self, name: str, coro: Awaitable[Any],
                    default: Any = None) -> Any:
        """Run one cycle step; a failure degrades the cycle instead of ending it.

        **Found by chaos testing, and it mattered.** The cycle used to be a
        straight sequence of awaits, so a database blip in step 2 (control
        commands) raised, `run_forever` caught it, and the cycle ended — having
        never reached **step 4, protective exits**. One outage of ten minutes
        is ~30 consecutive skipped stop-loss checks, precisely when you least
        want them skipped.

        Isolating the steps costs one thing and it must be paid explicitly: a
        cycle can now complete while blind. `degraded_cycles` counts that, and
        `_cycle_end` halts the engine once it has been blind long enough.
        """
        try:
            result = await coro
        except Exception:
            self.step_failures += 1
            self._degraded_this_cycle = True
            # Once per step per cycle, not per occurrence: a dead database
            # would otherwise write a log line every 20 seconds forever.
            log.exception("cycle step %r failed; continuing with the rest", name)
            return default
        return result

    async def cycle(self) -> None:
        """One pass. Order of operations is the design — see the module docstring.

        Every step that touches the database is isolated, so the steps that
        protect capital do not depend on the steps that record it.
        """
        self.cycles += 1
        self._degraded_this_cycle = False
        now = self._now()

        # 1. kill switch — cheapest, most absolute, and touches nothing but the
        #    filesystem, so it cannot be taken out by a database.
        self.state.kill_engaged = self.kill.engaged
        if self.state.kill_engaged and self.cycles % 30 == 1:
            log.warning("kill switch ENGAGED (%s): no orders will be sent",
                        self.kill.reason())

        # 2. control commands — a halt queued a second ago must land first
        await self._step("control", self._apply_control())

        # 3. market data
        marks, ages, fresh = await self._step(
            "market_data", self._load_market_state(now), ({}, {}, {}))
        if marks:
            self.acct.mark(now, marks)

        # 4. protective exits BEFORE entries — a stop firing this bar must not
        #    queue behind a new position, and must not queue behind a database.
        await self._step("protective_exits",
                         self._check_protective_exits(marks, now))

        # 4b. flatten requests. Runs even while halted — flattening is a close
        #     path, and refusing to close because the account is in trouble is
        #     how a small loss becomes a large one. Only the kill switch wins.
        await self._step("flatten", self._apply_flatten())

        # 5–7. strategies → risk → execution, only on symbols with a new bar
        if fresh and self.state.may_open:
            await self._step("strategies",
                             self._run_strategies(fresh, marks, ages, now))

        # 8. fills
        if self.executor is not None:
            await self._step("fills", self.executor.poll_fills(
                correlation_ids=self._correlations))

        # 9. mark to market
        if marks:
            snap = self.acct.snapshot(now, marks,
                                      correlation_clusters=self.correlation_clusters)
            self.bus.publish(portfolio_updated(
                self.acct.equity, self.acct.cash,
                float(snap.gross_exposure()), float(snap.drawdown()),
                snap.open_position_count()))
            await self._step("drawdown", self._enforce_drawdown(snap))
            if self.candles is not None:
                await self._step("equity", self._record_equity(now, snap))

        # 10. housekeeping
        await self._step("housekeeping", self._housekeeping(now))
        self._cycle_end()

    async def _record_equity(self, now: datetime, snap: AccountState) -> None:
        from ct.store import EquityRepository
        assert self.db is not None      # gated on `self.candles is not None`
        await EquityRepository(self.db).record(
            self.account_id, now, equity=self.acct.equity, cash=self.acct.cash,
            positions_value=self.acct.equity - self.acct.cash,
            gross_exposure=snap.gross_exposure(), drawdown=snap.drawdown())

    def _cycle_end(self) -> None:
        """Stop opening once the engine has been blind for long enough.

        An engine that cannot read `control_commands` cannot be halted from the
        dashboard or the CLI, and an engine that cannot read candles is sizing
        against stale marks. Continuing to *open* positions in that state is
        trading blind; continuing to *close* them is exactly what should still
        happen, which is why this halts rather than kills.
        """
        if self._degraded_this_cycle:
            self.degraded_cycles += 1
            if (self.degraded_cycles >= self.max_degraded_cycles
                    and not self.state.halted):
                self.state.halt(
                    f"{self.degraded_cycles} consecutive degraded cycles — "
                    f"the engine cannot see its own state")
                self.bus.publish(risk_limit_exceeded(
                    "ENGINE_DEGRADED",
                    f"halted after {self.degraded_cycles} cycles with failing "
                    f"steps; exits still run"))
        elif self.degraded_cycles:
            log.info("cycle healthy again after %d degraded cycles",
                     self.degraded_cycles)
            self.degraded_cycles = 0

    # ── the pieces ───────────────────────────────────────────────────────

    async def _load_market_state(
        self, now: datetime,
    ) -> tuple[dict[str, Decimal], dict[str, int], dict[tuple[str, str], list[Bar]]]:
        """Latest marks, data age, and history for symbols with a NEW closed bar.

        Data age is derived here from the candles table rather than asked of the
        collector: the two containers share no channel, and one number computed
        one way cannot disagree with itself.
        """
        marks: dict[str, Decimal] = {}
        ages: dict[str, int] = {}
        fresh: dict[tuple[str, str], list[Bar]] = {}
        if self.candles is None:
            return marks, ages, fresh

        for alloc in self.allocations:
            tf = alloc.timeframe
            step = timeframe_seconds(tf)
            for symbol in alloc.symbols:
                key = (symbol, tf)
                latest = await self.candles.latest_ts(symbol, tf)
                if latest is None:
                    ages[symbol] = max(ages.get(symbol, 0), 10**6)
                    continue
                age = max(0, int((now - latest).total_seconds() // step) - 1)
                ages[symbol] = max(ages.get(symbol, 0), age)
                if self._last_bar.get(key) == latest:
                    continue                       # nothing new since last pass
                bars = await self.candles.load(symbol, tf,
                                               limit=self.history_bars)
                if len(bars) < alloc.strategy.warmup + 1:
                    continue
                self._last_bar[key] = latest
                fresh[key] = bars
                marks[symbol] = Decimal(str(bars[-1].close))
                self._vols[symbol] = self._compute_vol(
                    [b.close for b in bars], tf)

        # Symbols with no new bar still need a mark for exposure accounting.
        for alloc in self.allocations:
            for symbol in alloc.symbols:
                if symbol in marks:
                    continue
                bars = await self.candles.load(symbol, alloc.timeframe, limit=1)
                if bars:
                    marks[symbol] = Decimal(str(bars[-1].close))
        return marks, ages, fresh

    async def _run_strategies(self, fresh: dict[tuple[str, str], list[Bar]],
                              marks: dict[str, Decimal], ages: dict[str, int],
                              now: datetime) -> None:
        for alloc in self.allocations:
            strat = alloc.strategy
            if strat.name in self.state.paused_strategies:
                continue
            for symbol in alloc.symbols:
                bars = fresh.get((symbol, alloc.timeframe))
                if not bars:
                    continue
                ctx = self._context(strat, symbol, alloc.timeframe, bars)
                try:
                    proposals = strat.on_bar(ctx)
                except Exception:
                    # Quarantine the strategy, not the engine. A strategy bug
                    # must not stop stops from firing.
                    log.exception("%s raised on %s; pausing it", strat.name, symbol)
                    self.state.paused_strategies.add(strat.name)
                    await self._quarantine(strat.name,
                                           f"raised on {symbol}")
                    self.bus.publish(Event(
                        EventType.STRATEGY_QUARANTINED,
                        {"strategy": strat.name, "symbol": symbol},
                        severity=Severity.WARNING, source="engine"))
                    await self._risk_event(
                        "STRATEGY_QUARANTINED",
                        f"{strat.name} raised on {symbol}; strategy paused",
                        severity="warn", publish=False,
                        strategy=strat.name, symbol=symbol)
                    continue
                for sig in proposals:
                    await self._consider(sig, alloc, marks, ages, now)

    def _context(self, strat: Strategy, symbol: str, timeframe: str,
                 bars: list[Bar]) -> Context:
        import numpy as np

        from ct.strategies.features import FeatureSet

        i = len(bars) - 1
        book = self.acct.position(strat.name, symbol)
        o = np.array([b.open for b in bars])
        h = np.array([b.high for b in bars])
        low_ = np.array([b.low for b in bars])
        c = np.array([b.close for b in bars])
        v = np.array([b.volume for b in bars])
        # Cached per (symbol, timeframe, window) rather than per strategy: on a
        # multi-strategy allocation every strategy on this symbol shares one
        # computation. Rebuilt when the window moves, which is once a bar.
        key = (symbol, timeframe, bars[-1].ts, len(bars))
        cached = self._features.get(key)
        if cached is None:
            cached = FeatureSet(o, h, low_, c, v)
            self._features = {key: cached}      # one window at a time
        return Context(
            symbol=symbol, timeframe=timeframe, index=i, ts=bars[-1].ts,
            open=o, high=h, low=low_, close=c, volume=v,
            position=book, regime=Regime.UNKNOWN, params=strat.params,
            f=cached.view(i))

    async def _consider(self, sig: Signal, alloc: Allocation,
                        marks: dict[str, Decimal], ages: dict[str, int],
                        now: datetime) -> None:
        self.signals_seen += 1
        book_key = f"{sig.strategy}|{sig.symbol}"
        cor = self._legs.get(book_key) or new_correlation_id()
        sig_ev = self.bus.publish(signal_generated(
            sig.strategy, sig.symbol, sig.direction, sig.strength, cor,
            ref_price=sig.ref_price))

        snapshot = self.acct.snapshot(
            now, marks, data_age_bars=ages,
            correlation_clusters=self.correlation_clusters)
        decision = self.risk.evaluate(
            sig, snapshot, realized_ann_vol=self._realized_vol(sig.symbol),
            strategy_weight=alloc.weight, armed=self.state.armed,
            kill_switch=self.state.kill_engaged,
            strategy_paused=sig.strategy in self.state.paused_strategies)

        if self.db is not None:
            from ct.store import SignalRepository
            await SignalRepository(self.db).record(
                self.account_id, now, sig.strategy, sig.symbol, sig.direction,
                accepted=decision.approved, strength=sig.strength,
                ref_price=Decimal(str(sig.ref_price)) if sig.ref_price else None,
                reject_code=decision.code.value if decision.code else None,
                reject_detail=decision.detail,
                sized_qty=decision.order.qty if decision.order else None)

        if not decision.approved:
            self.bus.publish(signal_rejected(
                sig.strategy, sig.symbol,
                decision.code.value if decision.code else "UNKNOWN",
                decision.detail, cor, causation_id=sig_ev.seq))
            return

        order = decision.order
        assert order is not None
        approved = self.bus.publish(risk_approved(
            sig.strategy, sig.symbol, order.qty,
            decision.diagnostics.get("binding_sizer", "?"), cor,
            causation_id=sig_ev.seq))

        self._correlations[order.client_order_id] = cor
        self._legs[book_key] = cor
        if self.executor is not None:
            await self.executor.submit(order, correlation_id=cor,
                                       causation_id=approved.seq)
            self.acct.reserve(order.client_order_id, order.strategy,
                              order.symbol, order.notional(
                                  marks.get(order.symbol, Decimal(0))))
        self.acct.record_order(now)
        self.orders_placed += 1

    async def _check_protective_exits(self, marks: dict[str, Decimal],
                                      now: datetime) -> None:
        """Stops and targets, enforced engine-side every cycle.

        Never relies solely on a resting exchange stop: those can be cancelled
        by venue maintenance, rejected on a thin book, and are visible to the
        venue. Runs even while halted — `may_close` is blocked only by the kill
        switch.
        """
        if not self.state.may_close or self.executor is None:
            return
        for key, book in list(self.acct.positions.items()):
            if book.is_flat:
                continue
            mark = marks.get(book.symbol)
            if mark is None:
                continue
            long = book.qty > 0
            hit: ExitReason | None = None
            if book.stop_price is not None:
                if (long and mark <= book.stop_price) or (
                        not long and mark >= book.stop_price):
                    hit = ExitReason.TRAIL if book.trail_atr_mult else ExitReason.STOP
            if hit is None and book.target_price is not None:
                if (long and mark >= book.target_price) or (
                        not long and mark <= book.target_price):
                    hit = ExitReason.TARGET
            if hit is None:
                continue

            order = Order(
                symbol=book.symbol, side=Side.SELL if long else Side.BUY,
                type=OrderType.MARKET, qty=abs(book.qty), tif=TimeInForce.IOC,
                strategy=book.strategy, reduce_only=True,
                meta={"reason_enum": hit.value})
            cor = self._legs.get(key) or new_correlation_id()
            self._correlations[order.client_order_id] = cor
            log.warning("protective exit %s on %s at %s", hit.value,
                        book.symbol, mark)
            await self.executor.submit(order, correlation_id=cor)

    async def _apply_flatten(self) -> None:
        """Close positions a `flatten_all`/`flatten_symbol` request points at.

        Until 2026-07-29 the flags were set and **nothing flattened** — the CLI
        and dashboard both had to carry a warning saying so. Now: one
        reduce-only market order per open book, deduplicated so a slow fill
        does not attract a second close every cycle, and the flag clears only
        when the book is actually flat — completion is observed, not assumed.

        Runs while halted (a halt is exactly when you want to get out); only
        the kill switch blocks it, because the kill switch means "I do not
        trust this system to send ANY order".
        """
        st = self.state
        if not (st.flatten_all or st.flatten_requested):
            self._flatten_submitted.clear()
            return
        if not st.may_close:
            return                                   # kill switch wins

        open_books = {k: b for k, b in self.acct.positions.items()
                      if not b.is_flat}
        # Books that filled since the last pass may be re-flattened later.
        self._flatten_submitted &= set(open_books)

        # A symbol request is complete when nothing is open on that symbol.
        for sym in sorted(st.flatten_requested):
            if all(b.symbol != sym for b in open_books.values()):
                st.flatten_requested.discard(sym)
                log.warning("flatten %s: complete, no open position remains", sym)
                await self._risk_event(
                    "FLATTEN_COMPLETE", f"{sym} flat after flatten request",
                    severity="info", publish=False, symbol=sym)
        if st.flatten_all and not open_books:
            st.flatten_all = False
            log.warning("flatten_all: complete, book is flat")
            await self._risk_event(
                "FLATTEN_COMPLETE", "all positions closed by flatten request",
                severity="info", publish=False)
        if not (st.flatten_all or st.flatten_requested):
            self._flatten_submitted.clear()
            return

        targets = {k: b for k, b in open_books.items()
                   if st.flatten_all or b.symbol in st.flatten_requested}
        if targets and self.executor is None:
            log.error("flatten requested but no executor is wired; "
                      "cannot submit closing orders")
            return
        # Bound once, outside the loop. Flatten is a risk-reducing path: if the
        # executor is missing that is a wiring bug and must raise, never be
        # quietly skipped, so this asserts rather than returning early.
        executor = self.executor
        assert executor is not None, "flatten requires a wired executor"
        for key, book in targets.items():
            if key in self._flatten_submitted:
                continue                             # close already in flight
            long = book.qty > 0
            order = Order(
                symbol=book.symbol, side=Side.SELL if long else Side.BUY,
                type=OrderType.MARKET, qty=abs(book.qty), tif=TimeInForce.IOC,
                strategy=book.strategy, reduce_only=True,
                meta={"reason_enum": ExitReason.FLATTEN.value})
            cor = self._legs.get(key) or new_correlation_id()
            self._correlations[order.client_order_id] = cor
            log.warning("flatten: closing %s (qty %s) at market", key, book.qty)
            await executor.submit(order, correlation_id=cor)
            self._flatten_submitted.add(key)

    async def _load_lifecycles(self) -> None:
        """Restore durable strategy state, and register anything new.

        A strategy the engine is configured to run but the table has never seen
        is registered as `proposed` — which means it does **not** run. Starting
        an unknown strategy because it appeared in an env var is the wrong
        default for something that can put on positions; a person enables it.
        """
        if self.strategies_repo is None:
            return

        try:
            self.lifecycles = await self.strategies_repo.load_all()
        except Exception:
            # Fail closed: unknown lifecycle state means nothing runs.
            log.exception("could not load strategy lifecycles; pausing all")
            for alloc in self.allocations:
                self.state.paused_strategies.add(alloc.strategy.name)
                self._lifecycle_blocked.add(alloc.strategy.name)
            return

        for alloc in self.allocations:
            name = alloc.strategy.name
            lc = self.lifecycles.get(name)
            if lc is None:
                with contextlib.suppress(Exception):
                    await self.strategies_repo.upsert(
                        name, class_path=type(alloc.strategy).__module__,
                        params=alloc.strategy.params,
                        symbols=list(alloc.symbols),
                        timeframe=alloc.timeframe, weight=alloc.weight)
                log.warning("%s is not enabled (registered as proposed); it "
                            "will not run until a human enables it "
                            "(`qd enable-strategy %s`)", name, name)
                self.state.paused_strategies.add(name)
                self._lifecycle_blocked.add(name)
                continue
            # Compare before and after: `resume_if_due` has already changed
            # the state by the time we look, so a condition testing for PAUSED
            # afterwards can never be true and the resume is never written
            # back — the strategy would resume on every boot and forget again.
            was = lc.state
            lc.resume_if_due(self._now())
            if lc.state is not was:
                with contextlib.suppress(Exception):
                    await self.strategies_repo.save(lc, actor=_ENGINE)
            if not lc.may_run:
                self.state.paused_strategies.add(name)
                self._lifecycle_blocked.add(name)
                log.warning("%s is %s: %s", name, lc.state.value,
                            lc.reason or "no reason recorded")

    async def _quarantine(self, name: str, reason: str) -> None:
        """Stop a strategy in a way a restart cannot undo."""
        if self.strategies_repo is None:
            return
        from ct.strategies.lifecycle import StrategyLifecycle, StrategyState

        lc = self.lifecycles.get(name) or StrategyLifecycle(
            name=name, state=StrategyState.PAPER)
        self._lifecycle_blocked.add(name)
        try:
            lc.quarantine(reason, now=self._now())
            self.lifecycles[name] = lc
            await self.strategies_repo.save(lc, actor=_ENGINE)
        except Exception:
            # An unpersisted quarantine still holds for this process, and the
            # in-memory set already has it. Losing the row is worse than not
            # stopping? No — it is strictly better than crashing the loop.
            log.exception("could not persist quarantine of %s", name)

    async def _risk_event(self, code: str, detail: str, *,
                          severity: str = "critical", publish: bool = True,
                          **payload: Any) -> None:
        """Publish AND persist. Both, always, and in that order.

        The event log is the observability spine, but it is not queryable the
        way an operator needs at 3 a.m. — `risk_events` is the table the
        dashboard and `qd status` read to answer "is anything unresolved?".
        Until this existed, `RiskEventRepository` had no callers anywhere in the
        codebase: every breach was published as an event and **no row was ever
        written**, so both surfaces reported "0 unresolved risk events" whether
        or not anything had breached. A green light that cannot turn red is
        worse than no light.

        `publish=False` for a site that has already emitted its own, more
        specific event — one condition should not produce two events just to
        get a row written.
        """
        if publish:
            self.bus.publish(risk_limit_exceeded(code, detail))
        if self.db is None:
            return
        try:
            from ct.store import RiskEventRepository
            await RiskEventRepository(self.db).record(
                self.account_id, severity, code, detail, payload=payload or {})
        except Exception:
            # Never let bookkeeping take down the loop that enforces stops.
            log.exception("could not persist risk event %s", code)

    async def _enforce_drawdown(self, snapshot: AccountState) -> None:
        dd = snapshot.drawdown()
        if dd >= Decimal(str(self.limits.max_total_drawdown)) and not self.state.halted:
            self.state.halt(f"total drawdown {dd:.2%}")
            await self._risk_event(
                "MAX_DRAWDOWN_HALT", f"drawdown {dd:.2%} — flattening and disarming",
                drawdown=float(dd), limit=self.limits.max_total_drawdown)
            self.state.flatten_all = True
            if self.arming is not None:
                # Disarm needs no permission. A drawdown breach is a HARD
                # reason, so no auto re-arm follows it.
                await self.arming.disarm(self.account_id, reason="max_drawdown")
                self.state.armed = False

    async def _apply_control(self) -> None:
        if self.control is None:
            return
        for cmd in await self.control.pending():
            result = "applied"
            if cmd.command == "halt":
                self.state.halt(f"requested by {cmd.requested_by}")
            elif cmd.command == "resume":
                result = "applied" if self.state.resume() else "refused: unreconciled"
            elif cmd.command == "flatten_all":
                self.state.flatten_all = True
            elif cmd.command == "flatten_symbol":
                self.state.flatten_requested.add(str(cmd.args.get("symbol", "")))
            elif cmd.command == "pause_strategy":
                self.state.paused_strategies.add(str(cmd.args.get("strategy", "")))
            elif cmd.command == "resume_strategy":
                self.state.paused_strategies.discard(str(cmd.args.get("strategy", "")))
            elif cmd.command == "resolve_risk_event":
                # The dashboard queues this rather than writing risk_events
                # itself: the engine stays the single writer of its own tables.
                from ct.store import RiskEventRepository
                try:
                    raw_id = cmd.args.get("id")
                    if raw_id is None:
                        raise TypeError("no id supplied")
                    eid = int(raw_id)
                except (TypeError, ValueError):
                    result = f"refused: bad event id {cmd.args.get('id')!r}"
                else:
                    assert self.db is not None
                    await RiskEventRepository(self.db).resolve(eid)
                    result = f"resolved risk_event {eid}"
            elif cmd.command == "reload_config":
                await self._reload_limits()
            else:
                result = f"unknown command {cmd.command!r}"
            await self.control.mark_applied(cmd.id, result)
            log.info("control: %s by %s -> %s", cmd.command, cmd.requested_by, result)

    async def _reload_limits(self) -> None:
        # Only reachable from `_apply_control`, which runs behind a wired
        # control plane — and a control plane implies a database.
        assert self.db is not None
        row = await self.db.fetchrow(
            """SELECT limits FROM risk_config WHERE account_id=$1
               ORDER BY effective_from DESC LIMIT 1""", self.account_id)
        if row is None:
            return
        import json
        raw = row["limits"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        known = {f.name for f in RiskLimits.__dataclass_fields__.values()}
        self.limits = RiskLimits(**{k: v for k, v in data.items() if k in known})
        self.risk = RiskEngine(self.limits)
        self.acct.limits = self.limits
        log.info("risk config reloaded")

    async def _refresh_arming(self) -> None:
        """Arming gates LIVE trading only.

        The two switches are `QD_MODE=live` **and** an unexpired `live_arming`
        row (ARCHITECTURE.md §8.4). If the first is off, the second is
        irrelevant — checking it anyway made paper mode require a human tap,
        which is exactly backwards: the 90-day paper track record the promotion
        gate demands cannot be accumulated if paper refuses to trade until
        somebody arms it.

        This failed silently. With `db=None` the paper branch armed correctly
        and the unit tests passed; with a real database, paper mode sat there
        doing nothing and looking healthy. Found only by running the whole
        system end to end.
        """
        if self.mode != "live":
            self.state.armed = True
            return
        if self.arming is None:
            # Live with no database has nowhere to record a human tap, so
            # there is no way to be armed. Refuse rather than assume.
            self.state.armed = False
            log.error("live mode requires a database for the arming record")
            return
        self.state.armed = await self.arming.is_armed(self.account_id)
        if not self.state.armed:
            log.warning("QD_MODE=live but no valid arming row: no orders will "
                        "be sent. Run `qd arm-live`.")

    async def _refresh_lifecycles(self, now: datetime) -> None:
        """Pick up lifecycle changes made while the engine is running.

        Boot-only loading meant `qd enable-strategy` (or any external change —
        dashboard, psql) did nothing until the next container restart: the
        human gate existed but pulling it had no visible effect, which is how
        an operator ends up restarting containers "to make it work". Polled
        from housekeeping, so an external change lands within one cycle.

        Failure keeps the current states: an engine that cannot read lifecycle
        rows keeps every existing block and unblocks nothing — the same
        fail-closed direction as boot.
        """
        if self.strategies_repo is None:
            return
        try:
            fresh = await self.strategies_repo.load_all()
        except Exception:
            log.exception("could not refresh strategy lifecycles; "
                          "keeping current states")
            return
        self.lifecycles = fresh
        for alloc in self.allocations:
            name = alloc.strategy.name
            lc = fresh.get(name)
            if lc is None:
                continue                 # boot registers; nothing to do yet
            # A time-boxed pause now expires while running, not only at boot.
            was = lc.state
            lc.resume_if_due(now)
            if lc.state is not was:
                with contextlib.suppress(Exception):
                    await self.strategies_repo.save(lc, actor=_ENGINE)
            if lc.may_run and name in self._lifecycle_blocked:
                # Only unblock names *lifecycle* blocked — an operator's
                # transient `pause_strategy` control command is not ours to undo.
                self._lifecycle_blocked.discard(name)
                self.state.paused_strategies.discard(name)
                log.warning("%s is now %s — enabled externally; it trades on "
                            "its next bar", name, lc.state.value)
                self.bus.publish(Event(
                    EventType.STRATEGY_RESUMED,
                    {"strategy": name, "state": lc.state.value,
                     "reason": lc.reason},
                    source="engine", severity=Severity.INFO))
            elif not lc.may_run and name not in self._lifecycle_blocked:
                self._lifecycle_blocked.add(name)
                if name not in self.state.paused_strategies:
                    log.warning("%s is now %s (%s) — stopped externally",
                                name, lc.state.value,
                                lc.reason or "no reason recorded")
                self.state.paused_strategies.add(name)

    async def _housekeeping(self, now: datetime) -> None:
        await self._refresh_lifecycles(now)
        # Promotion evaluation used to live here, gated by a date check — a cron
        # job hand-rolled inside the hot path. It is now a real daily job (see
        # ct/engine/scheduler.py). `evaluate_promotion` keeps the date guard, so
        # nothing about its idempotence changed.
        #
        # Belt and braces: if the engine is running WITHOUT a scheduler — the
        # test harness drives `cycle()` directly, and so does anything embedding
        # the engine — the daily job does not exist and promotion would never
        # run, leaving the arm-live gate reading "never evaluated" forever. The
        # date guard makes running it here harmless when the job also exists.
        if self._scheduler is None:
            await self.evaluate_promotion(now)
        if self.arming is not None and self.state.armed:
            # Auto-renew a human-originated arming while the envelope holds.
            # He arms once; the engine keeps it alive. ARCHITECTURE.md §8.4.
            with contextlib.suppress(Exception):
                if await self.arming.due_for_renewal(self.account_id):
                    current = await self.arming.active(self.account_id)
                    if current is not None:
                        renewed = await self.arming.renew(
                            self.account_id, current["envelope_hash"])
                        if renewed is not None:
                            self.bus.publish(Event(
                                EventType.ARMING_RENEWED,
                                {"arming_id": renewed}, source="engine"))
        if self.outbox is not None:
            with contextlib.suppress(Exception):
                await self.outbox.drain(self._send_notification)
        if self.cycles % 15 == 1:
            self.bus.publish(heartbeat("engine", **self.status()))

    async def _send_notification(self, channel: str, subject: str | None,
                                 body: str) -> bool:
        """Real delivery via `ct.notify.TelegramSender` (2026-07-30).

        Unconfigured degrades to log-and-drain — same behaviour the old stub
        had, but now it says so. Failure returns False and the outbox's
        backoff owns the retry."""
        if getattr(self, "_notifier", None) is None:
            from ct.notify import TelegramSender
            self._notifier = TelegramSender.from_env()
            if self._notifier.configured:
                log.info("telegram notifications are LIVE (chat id configured)")
        return await self._notifier.send(channel, subject, body)

    def _compute_vol(self, closes: list[float], timeframe: str) -> float:
        """Identical math to `BacktestEngine._realized_ann_vol` — that identity
        is the point: a sizer fed different vol live than in the backtest is
        two different systems wearing one name."""
        import math

        from ct.risk.sizing import annualize_vol, ewma_volatility

        window = closes[-(self.vol_lookback + 1):]
        if len(window) < 3:
            return 0.0
        rets = [math.log(max(b, 1e-12) / max(a, 1e-12))
                for a, b in itertools.pairwise(window)]
        return annualize_vol(ewma_volatility(rets, halflife=self.vol_lookback),
                             timeframe)

    def _realized_vol(self, symbol: str) -> float:
        # 0.5 is the pre-first-bar fallback only; a fresh bar replaces it on
        # the first cycle that loads history for the symbol.
        return self._vols.get(symbol, 0.5)

    # ── lifecycle ────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stopping.set()

    async def shutdown(self) -> None:
        """SIGTERM path: cancel resting orders, persist, exit.

        Positions are NOT flattened — a container restart is not a reason to
        market-sell the book, and doing so would turn every deploy into a
        taxable event. Resting stops remain the protection.
        """
        self._stopping.set()
        if self.executor is not None:
            cancelled = await self.executor.cancel_all_open()
            log.info("cancelled %d resting orders on shutdown", cancelled)
            for key, book in self.acct.positions.items():
                if not book.is_flat and self.positions_repo is not None:
                    await self.positions_repo.save_protective_levels(
                        self.account_id, book)
                _ = key
        self.bus.publish(Event(EventType.ENGINE_STOPPING,
                               {"component": "engine"}, source="engine"))
        self.bus.flush()

    def status(self) -> dict[str, object]:
        return {
            **self.state.summary(),
            "cycles": self.cycles,
            "degraded_cycles": self.degraded_cycles,
            "step_failures": self.step_failures,
            "signals": self.signals_seen,
            "orders": self.orders_placed,
            "equity": float(self.acct.equity),
            "open_books": sum(1 for p in self.acct.positions.values()
                              if not p.is_flat),
            # Live realized vols, so the sizer's actual input is observable
            # (a constant 0.5 hid here for a month).
            "realized_vols": {s: round(v, 4) for s, v in self._vols.items()},
        }


__all__ = ["Allocation", "TradingEngine"]
