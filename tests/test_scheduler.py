"""APScheduler wiring for the engine (ARCHITECTURE.md Merge 5).

This replaced the timing of the loop that queues stop-loss checks, so the tests
that matter here are not "does a job fire" but the safety properties:

  * **no overlap** — two cycles at once is not slowness, it is double-sizing
    against a stale account snapshot;
  * **a backlog collapses** — after a stall, run once, not fifteen times;
  * **a stale fire is dropped** — a cycle whose time passed while the process
    was down would act on prices that have since moved;
  * **a failing cycle does not stop the schedule**, and is still loud;
  * **jobs are module-level** — the trap documented in ct/engine/scheduler.py,
    where a bound method serialises to its class and comes back unbound.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone

import pytest

from ct.engine import scheduler as sched


class FakeEngine:
    """Just enough engine for the scheduler to drive."""

    def __init__(self, *, work: float = 0.0, boom: bool = False):
        self.work = work
        self.boom = boom
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self.promotions: list[datetime] = []

    async def cycle_guarded(self) -> None:
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.boom:
                raise RuntimeError("cycle exploded")
            if self.work:
                await asyncio.sleep(self.work)
        finally:
            self.concurrent -= 1

    async def evaluate_promotion(self, now: datetime) -> None:
        self.promotions.append(now)


@pytest.fixture
def key():
    k = "test:engine"
    yield k
    sched.unregister_engine(k)


def _started(dsn=None):
    """A started scheduler, matching run_forever's order: start, then add jobs.

    Adding to a stopped scheduler puts jobs in _pending_jobs, where
    replace_existing does not deduplicate — the ordering is part of the
    contract, so the tests exercise the same one production uses.
    """
    scheduler, described = sched.build_scheduler(dsn=dsn)
    scheduler.start()
    return scheduler, described


async def _run_for(scheduler, seconds: float):
    try:
        await asyncio.sleep(seconds)
    finally:
        scheduler.shutdown(wait=False)


# ── the property that protects the book ──────────────────────────────────


async def test_cycles_never_overlap(key):
    """max_instances=1. A cycle that outruns its interval must not be joined by
    the next one — `cycle()` mutates the account and places orders."""
    engine = FakeEngine(work=0.30)
    sched.register_engine(key, engine)
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=0.05,
                               misfire_grace_seconds=30, with_promotion=False)
    await _run_for(scheduler, 1.0)
    assert engine.max_concurrent == 1, \
        f"{engine.max_concurrent} cycles ran at once"
    assert engine.calls >= 1


async def test_a_backlog_collapses_instead_of_bursting(key):
    """coalesce=True. After a stall the engine should run once, not replay every
    fire it missed — fifteen cycles in a burst is fifteen chances to size."""
    engine = FakeEngine(work=0.45)
    sched.register_engine(key, engine)
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=0.05,
                               misfire_grace_seconds=30, with_promotion=False)
    await _run_for(scheduler, 1.0)
    # ~20 fire opportunities in the window; a coalescing, non-overlapping
    # schedule can only have run a couple of times.
    assert engine.calls <= 4, f"burst of {engine.calls} cycles"


async def test_job_defaults_are_set_on_the_scheduler_itself(key):
    """Belt and braces: the two safety flags are on job_defaults as well as on
    each add_job, so a future job added without them still inherits them."""
    scheduler, _ = sched.build_scheduler(dsn=None)
    assert scheduler._job_defaults["max_instances"] == 1
    assert scheduler._job_defaults["coalesce"] is True


# ── failure isolation ────────────────────────────────────────────────────


async def test_a_failing_cycle_does_not_stop_the_schedule(key):
    engine = FakeEngine(boom=True)
    sched.register_engine(key, engine)
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=0.05,
                               misfire_grace_seconds=30, with_promotion=False)
    await _run_for(scheduler, 0.5)
    assert engine.calls >= 3, "the schedule stopped after a failure"


async def test_a_job_for_an_unregistered_engine_is_a_logged_noop(caplog):
    """A jobstore can outlive its process and hold a job for an account this
    process is not running. Crashing the scheduler thread is the wrong answer."""
    with caplog.at_level(logging.WARNING, logger="ct.engine.scheduler"):
        await sched.run_cycle("nobody:home")
        await sched.run_promotion("nobody:home")
    assert sum("not running" in r.message for r in caplog.records) == 2


# ── the bound-method trap ────────────────────────────────────────────────


def test_job_targets_are_module_level_functions():
    """Regression guard for the defect this module exists to avoid.

    `obj_to_ref(engine.cycle)` SUCCEEDS and returns the class's function with
    the instance discarded, so a persistent jobstore restores something that
    raises TypeError on every fire. The failure only appears after a restart.
    """
    from apscheduler.util import obj_to_ref, ref_to_obj

    for fn in (sched.run_cycle, sched.run_promotion):
        ref = obj_to_ref(fn)
        assert ref.startswith("ct.engine.scheduler:"), ref
        assert ref_to_obj(ref) is fn, "job function does not round-trip"


def test_a_bound_method_would_not_survive_the_round_trip():
    """Documents *why* the functions above are module-level. If this ever stops
    being true, the indirection can go."""
    from apscheduler.util import obj_to_ref, ref_to_obj

    from ct.engine.loop import TradingEngine

    engine = object.__new__(TradingEngine)
    restored = ref_to_obj(obj_to_ref(engine.cycle))
    assert not hasattr(restored, "__self__"), \
        "bound methods now round-trip; the registry indirection is unnecessary"
    with pytest.raises(TypeError):
        restored()


# ── job registration ─────────────────────────────────────────────────────


async def test_both_jobs_are_registered_with_the_right_shape(key):
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=20,
                               misfire_grace_seconds=30)
    jobs = {j.id: j for j in scheduler.get_jobs()}
    assert set(jobs) == {sched.CYCLE_JOB_ID, sched.PROMOTION_JOB_ID}

    cycle = jobs[sched.CYCLE_JOB_ID]
    assert cycle.max_instances == 1
    assert cycle.coalesce is True
    assert cycle.misfire_grace_time == 30
    assert tuple(cycle.args) == (key,)   # APScheduler normalises args to a tuple
    assert "interval" in str(cycle.trigger)

    promo = jobs[sched.PROMOTION_JOB_ID]
    assert "cron" in str(promo.trigger)
    # Deliberately generous: promotion is idempotent, and a stale-by-an-hour
    # record beats the arm-live gate having no record at all.
    assert promo.misfire_grace_time == 3600
    scheduler.shutdown(wait=False)


async def test_promotion_fires_in_the_schedulers_timezone_not_the_machines(key):
    """Regression: a pre-built CronTrigger keeps the timezone it was CONSTRUCTED
    with — the machine's local zone — regardless of the scheduler's.

    APScheduler only applies the scheduler timezone when it builds the trigger
    itself from trigger="cron" plus kwargs. So a UTC scheduler was firing the
    daily job at 00:05 *local*: 07:05 UTC on the dev machine, 06:05 UTC on a NAS
    set to America/Chicago, while the README said 00:05 UTC. Found by reading
    next_run_time out of a real Postgres jobstore.

    It is not only a documentation problem: `evaluate_promotion` guards on the
    UTC date, so a local trigger makes which UTC day gets evaluated depend on
    where the box happens to be.
    """
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=20,
                               misfire_grace_seconds=30)
    promo = scheduler.get_job(sched.PROMOTION_JOB_ID)
    assert str(promo.trigger.timezone) == str(scheduler.timezone) == "UTC", (
        f"trigger tz {promo.trigger.timezone!s} != scheduler tz "
        f"{scheduler.timezone!s}")
    nxt = promo.next_run_time.astimezone(timezone.utc)
    assert (nxt.hour, nxt.minute) == (0, 5), \
        f"daily job fires at {nxt:%H:%M} UTC, not the documented 00:05"
    scheduler.shutdown(wait=False)


async def test_scheduling_is_idempotent(key):
    """replace_existing=True — a restart re-adds the same jobs, not duplicates."""
    scheduler, _ = _started()
    for _ in range(3):
        sched.schedule_engine_jobs(scheduler, key, cycle_seconds=20,
                                   misfire_grace_seconds=30)
    assert len(scheduler.get_jobs()) == 2
    scheduler.shutdown(wait=False)


async def test_promotion_job_is_skipped_without_a_database(key):
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=20,
                               misfire_grace_seconds=30, with_promotion=False)
    assert [j.id for j in scheduler.get_jobs()] == [sched.CYCLE_JOB_ID]
    scheduler.shutdown(wait=False)


async def test_describe_jobs_is_loggable(key):
    scheduler, _ = _started()
    sched.schedule_engine_jobs(scheduler, key, cycle_seconds=20,
                               misfire_grace_seconds=30)
    described = sched.describe_jobs(scheduler)
    assert {d["id"] for d in described} == {sched.CYCLE_JOB_ID,
                                            sched.PROMOTION_JOB_ID}
    for d in described:
        assert d["trigger"] and "name" in d
    scheduler.shutdown(wait=False)


# ── jobstore selection ───────────────────────────────────────────────────


def test_no_dsn_means_memory_and_says_so():
    _, described = sched.build_scheduler(dsn=None)
    assert "memory" in described


async def test_an_unreachable_database_degrades_to_memory_at_start(caplog):
    """A scheduler that cannot persist is degraded, not a reason to refuse to
    run the book. But it must not be silent — an operator who believes the
    schedule survives restarts, and is wrong, finds out at the worst moment.

    The failure lands at start(), not at construction: SQLAlchemy is lazy, so
    building a jobstore against an unreachable host succeeds. The first version
    of this test asserted a fallback at build time and passed only because
    psycopg was not installed, so the bad DSN raised ImportError instead of
    reaching the real path. Installing psycopg to test against a real database
    is what exposed it.
    """
    built, described = sched.build_scheduler(
        dsn="postgresql://nobody:pw@127.0.0.1:1/definitely-not-there")
    assert described == "postgres", "construction should NOT have failed yet"

    with caplog.at_level(logging.WARNING, logger="ct.engine.scheduler"):
        scheduler, final = sched.start_scheduler(built, described)
    try:
        assert scheduler.running
        assert "memory" in final, final
        assert any("will NOT survive a restart" in r.message
                   for r in caplog.records), caplog.text
        # ...and it is still usable.
        sched.schedule_engine_jobs(scheduler, "k", cycle_seconds=20,
                                   misfire_grace_seconds=30)
        assert len(scheduler.get_jobs()) == 2
    finally:
        scheduler.shutdown(wait=False)


def test_an_unopenable_jobstore_falls_back_at_build_time(caplog):
    """The other failure point: a store that cannot even be constructed."""
    import apscheduler.jobstores.sqlalchemy as js

    def boom(*a, **k):
        raise ImportError("no driver")

    orig = js.SQLAlchemyJobStore
    js.SQLAlchemyJobStore = boom
    try:
        with caplog.at_level(logging.WARNING, logger="ct.engine.scheduler"):
            _, described = sched.build_scheduler(dsn="postgresql://x@y/z")
        assert "memory" in described
        assert any("will NOT survive a restart" in r.message
                   for r in caplog.records)
    finally:
        js.SQLAlchemyJobStore = orig


def test_dsn_is_translated_for_sqlalchemy(monkeypatch):
    """Same asyncpg -> psycopg translation the migrations do; without it
    SQLAlchemy reaches for psycopg2, which this project does not ship."""
    seen = {}

    class FakeStore:
        def __init__(self, url, tablename, engine_options=None):
            seen["url"] = url
            seen["table"] = tablename
            seen["engine_options"] = engine_options or {}

    import apscheduler.jobstores.sqlalchemy as js
    monkeypatch.setattr(js, "SQLAlchemyJobStore", FakeStore)
    # _build_jobstore, not build_scheduler: the latter would hand this fake to
    # AsyncIOScheduler, which rightly insists on a real BaseJobStore.
    store, described = sched._build_jobstore(
        "postgresql+asyncpg://qd:pw@qd-db:5432/quantdesk")
    assert isinstance(store, FakeStore)
    assert described == "postgres"
    assert seen["url"] == "postgresql+psycopg://qd:pw@qd-db:5432/quantdesk"
    assert seen["table"] == "apscheduler_jobs"
    # Bounded, or an unreachable database blocks boot for over two minutes
    # before failing — measured at 130 s against a dead host.
    assert seen["engine_options"]["connect_args"]["connect_timeout"] == 5
    assert seen["engine_options"]["pool_pre_ping"] is True


# ── promotion moved out of the hot path ──────────────────────────────────


async def test_promotion_job_calls_the_engine(key):
    engine = FakeEngine()
    sched.register_engine(key, engine)
    await sched.run_promotion(key)
    assert len(engine.promotions) == 1
    assert engine.promotions[0].tzinfo is timezone.utc


async def test_engine_promotion_is_idempotent_within_a_day():
    """The scheduler coalesces missed fires and a restart can re-fire the day's
    job, so this has to be idempotent rather than merely infrequent."""
    from ct.engine.loop import TradingEngine

    engine = object.__new__(TradingEngine)
    engine.db = object()
    engine._last_promotion_eval = None
    calls = []

    class FakePromotion:
        @staticmethod
        async def evaluate(db, *, now):
            calls.append(now)

    import ct.engine as pkg
    original = getattr(pkg, "promotion", None)
    pkg.promotion = FakePromotion
    try:
        now = datetime(2026, 8, 4, 0, 5, tzinfo=timezone.utc)
        await TradingEngine.evaluate_promotion(engine, now)
        await TradingEngine.evaluate_promotion(engine, now)
        await TradingEngine.evaluate_promotion(
            engine, now.replace(hour=23, minute=59))
        assert len(calls) == 1, "promotion ran more than once in a day"
        assert engine._last_promotion_eval == date(2026, 8, 4)

        # ...and runs again the next day.
        await TradingEngine.evaluate_promotion(
            engine, datetime(2026, 8, 5, 0, 5, tzinfo=timezone.utc))
        assert len(calls) == 2
    finally:
        if original is not None:
            pkg.promotion = original


async def test_promotion_failure_is_swallowed_and_retried_tomorrow(caplog):
    from ct.engine.loop import TradingEngine

    engine = object.__new__(TradingEngine)
    engine.db = object()
    engine._last_promotion_eval = None

    class Boom:
        @staticmethod
        async def evaluate(db, *, now):
            raise RuntimeError("promotion exploded")

    import ct.engine as pkg
    original = getattr(pkg, "promotion", None)
    pkg.promotion = Boom
    try:
        with caplog.at_level(logging.ERROR, logger="ct.engine.loop"):
            await TradingEngine.evaluate_promotion(
                engine, datetime(2026, 8, 4, tzinfo=timezone.utc))
        assert any("promotion evaluation failed" in r.message
                   for r in caplog.records)
    finally:
        if original is not None:
            pkg.promotion = original


# ── run_forever, end to end ──────────────────────────────────────────────
#
# The scheduler module can be perfect and the engine still not use it. These
# drive the real TradingEngine.run_forever, which is the code path that
# replaced the while loop.


def _real_engine(tmp_path, **kw):
    from decimal import Decimal

    from ct.engine import Allocation, KillSwitch, TradingEngine
    from ct.events import EventBus, InMemoryEventLog
    from ct.execution.broker import FeeSchedule, PaperBroker
    from ct.risk.engine import RiskLimits
    from ct.strategies import REGISTRY

    return TradingEngine(
        [Allocation(REGISTRY["donchian_breakout"](), ("BTC/USD",), "4h")],
        PaperBroker(starting_cash=Decimal(10_000), fees=FeeSchedule()),
        db=None,
        limits=RiskLimits(prefer_maker=False, min_edge_bps=0.0,
                          min_order_notional=0.0),
        bus=EventBus(InMemoryEventLog(), source="test"),
        kill_switch=KillSwitch(str(tmp_path / "KILL")),
        **kw)


async def test_run_forever_actually_cycles_then_stops_promptly(tmp_path):
    """The whole point. Also guards the `next_run_time=None` trap: passing an
    explicit None adds the job PAUSED, which produced a perfectly configured
    trading cycle that never ran once."""
    engine = _real_engine(tmp_path, cycle_seconds=0.05)
    task = asyncio.create_task(engine.run_forever())
    await asyncio.sleep(0.6)
    assert engine.cycles >= 3, f"only {engine.cycles} cycles ran"
    assert engine._scheduler is not None, "run_forever did not attach a scheduler"

    engine.stop()
    await asyncio.wait_for(task, timeout=5)
    assert engine._scheduler is None, "scheduler was not torn down"
    assert sched.registered_engine(engine.scheduler_key) is None, \
        "engine stayed in the registry after shutdown"


async def test_run_forever_evaluates_promotion_at_boot(tmp_path):
    """Regression, found against a real database: moving promotion to a cron job
    alone left `promotion_status` empty until the next 00:05 UTC, so the
    arm-live gate read "never evaluated" for up to a day after every deploy.
    Previously `_housekeeping` populated it on the first cycle."""
    engine = _real_engine(tmp_path, cycle_seconds=30)
    # A sentinel db so the `self.db is not None` guard is taken, with boot() and
    # the cycle stubbed so neither tries to query it — run_forever calls boot()
    # first, and that is what reached repositories.py when only db was set.
    engine.db = object()
    seen = []

    async def fake_eval(now):
        seen.append(now)

    async def no_boot():
        return True

    async def no_cycle():
        return None

    engine.evaluate_promotion = fake_eval
    engine.boot = no_boot
    engine.cycle_guarded = no_cycle
    task = asyncio.create_task(engine.run_forever())
    await asyncio.sleep(0.4)
    engine.stop()
    await asyncio.wait_for(task, timeout=5)
    assert len(seen) == 1, f"promotion evaluated {len(seen)} times at boot"


async def test_run_forever_runs_one_cycle_immediately(tmp_path):
    """The old loop cycled first and slept second; an interval trigger waits a
    full interval. The first cycle after boot is the one that reconciles against
    reality, so it must not be delayed by the cadence."""
    engine = _real_engine(tmp_path, cycle_seconds=30)   # far longer than the test
    task = asyncio.create_task(engine.run_forever())
    await asyncio.sleep(0.4)
    assert engine.cycles >= 1, "no cycle ran before the first interval elapsed"
    engine.stop()
    await asyncio.wait_for(task, timeout=5)


async def test_stop_is_honoured_even_while_a_cycle_is_in_flight(tmp_path):
    engine = _real_engine(tmp_path, cycle_seconds=0.05)
    task = asyncio.create_task(engine.run_forever())
    await asyncio.sleep(0.2)
    engine.stop()
    # shutdown(wait=True) lets an in-flight cycle finish; killing one mid-way
    # could leave an order placed and unrecorded. It must still be prompt.
    await asyncio.wait_for(task, timeout=5)


async def test_engine_without_a_scheduler_still_evaluates_promotion(tmp_path):
    """The test harness and anything embedding the engine call cycle() directly,
    so the daily job does not exist. `_housekeeping` must still cover it or the
    arm-live gate reads 'never evaluated' forever."""
    engine = _real_engine(tmp_path)
    assert engine._scheduler is None
    seen = []

    async def fake_eval(now):
        seen.append(now)

    engine.evaluate_promotion = fake_eval
    await engine._housekeeping(datetime(2026, 8, 4, tzinfo=timezone.utc))
    assert len(seen) == 1, "promotion was skipped with no scheduler present"


async def test_scheduler_driven_engine_does_not_double_evaluate(tmp_path):
    """...and when the daily job DOES exist, housekeeping must not also run it."""
    engine = _real_engine(tmp_path)
    engine._scheduler = object()          # pretend the scheduler is attached
    seen = []

    async def fake_eval(now):
        seen.append(now)

    engine.evaluate_promotion = fake_eval
    await engine._housekeeping(datetime(2026, 8, 4, tzinfo=timezone.utc))
    assert seen == [], "housekeeping duplicated the scheduled promotion job"


async def test_promotion_is_a_noop_without_a_database():
    from ct.engine.loop import TradingEngine

    engine = object.__new__(TradingEngine)
    engine.db = None
    engine._last_promotion_eval = None
    await TradingEngine.evaluate_promotion(
        engine, datetime(2026, 8, 4, tzinfo=timezone.utc))
    assert engine._last_promotion_eval is None
