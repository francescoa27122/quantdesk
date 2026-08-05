"""In-process job scheduling for `qd-engine` (ARCHITECTURE.md Merge 5).

    "A dedicated scheduler container that triggers work in another container
     needs an RPC path. In-process AsyncIOScheduler with a Postgres jobstore
     survives restarts and has no network surface."

The engine ran a bare `while` loop instead. That worked, and this file has to
justify replacing it — a timing change in the loop that queues stop-loss checks
is not a refactor to make casually. Three things it buys:

**Cadence instead of drift.** The old loop slept `cycle_seconds` *after* each
cycle, so the real period was `work + sleep` and every slow cycle pushed all
later ones later. An interval trigger fires on a fixed schedule.

**One place that knows about time.** Daily promotion evaluation was an
`if self._last_promotion_eval != now.date()` check buried in `_housekeeping`,
i.e. a cron job hand-rolled inside the hot path.

**Restart survival**, which is the part that needed care — see below.

## Why these are module-level functions and not `engine.cycle`

This is the whole reason this file exists rather than three lines in `loop.py`.

A persistent jobstore has to store the callable somehow. APScheduler tries
`obj_to_ref` first, and for a *bound method* that call **succeeds and returns
the wrong thing**: `engine.cycle` serialises to `ct.engine.loop:TradingEngine.
cycle` — the class's function, with the instance thrown away. Verified:

    >>> ref = obj_to_ref(engine.cycle)      # 'ct.engine.loop:TradingEngine.cycle'
    >>> ref_to_obj(ref)()
    TypeError: TradingEngine.cycle() missing 1 required positional argument: 'self'

So `add_job(engine.cycle, ...)` works perfectly in the process that added it —
APScheduler holds the live callable in memory — and then, after the first
restart, every fire raises TypeError. The trading cycle stops running and the
only symptom is a log line. That is the exact failure a persistent jobstore was
supposed to prevent.

Jobs therefore take a string key and look the engine up here. A string
round-trips through any jobstore, and a key with no registered engine is a
logged no-op rather than a crash loop.

## Overlap is the safety property that matters

`cycle()` mutates `self.acct`, polls fills and places orders. Two of them
running at once is not "slower", it is double-sizing against a stale account
snapshot. `max_instances=1` is not tuning; removing it is a defect.

`coalesce=True` pairs with it: if the process was blocked for five minutes, run
the cycle **once** now, not fifteen times in a burst. And `misfire_grace_time`
bounds staleness — a cycle whose fire time passed while the process was down is
a cycle that would act on prices that have since moved, so past the grace it is
dropped rather than replayed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

CYCLE_JOB_ID = "qd:cycle"
PROMOTION_JOB_ID = "qd:promotion"

# Engines this process is running, by key. Almost always exactly one; the dict
# exists so the job functions can be module-level (see the docstring) rather
# than to support a fleet.
_ENGINES: dict[str, Any] = {}


def register_engine(key: str, engine: Any) -> None:
    _ENGINES[key] = engine


def unregister_engine(key: str) -> None:
    _ENGINES.pop(key, None)


def registered_engine(key: str) -> Any | None:
    return _ENGINES.get(key)


# ── job functions (module-level so a persistent jobstore can reference them) ──


async def run_cycle(key: str) -> None:
    """One trading cycle.

    A missing engine is a no-op, not an error: a jobstore that outlived its
    process can hold a job for an account this process is not running, and the
    correct response is to say so and do nothing rather than to crash the
    scheduler thread.
    """
    engine = _ENGINES.get(key)
    if engine is None:
        log.warning("scheduled cycle for an engine that is not running",
                    extra={"engine_key": key})
        return
    await engine.cycle_guarded()


async def run_promotion(key: str) -> None:
    """Daily promotion evaluation — the record `arm-live` reads.

    Was an `if self._last_promotion_eval != now.date()` check inside
    `_housekeeping`. The date guard is kept inside the engine, so a coalesced
    double-fire is idempotent rather than a double evaluation.
    """
    engine = _ENGINES.get(key)
    if engine is None:
        log.warning("scheduled promotion for an engine that is not running",
                    extra={"engine_key": key})
        return
    await engine.evaluate_promotion(datetime.now(timezone.utc))


# ── construction ─────────────────────────────────────────────────────────


def _build_jobstore(dsn: str | None) -> tuple[Any, str]:
    """Postgres when we can, memory when we cannot, and always say which.

    Persistence only matters for the daily job — the interval job is re-added
    identically at every boot, so storing it buys nothing. What it does buy is
    that a restart at 23:59 neither re-runs nor skips the day's promotion
    evaluation.

    Falling back silently would be the wrong failure: an operator who believes
    the schedule survives restarts, and is wrong, finds out at the worst moment.
    """
    if not dsn:
        return None, "memory (no QD_DB_DSN)"
    try:
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

        # SQLAlchemy needs an explicit driver or it reaches for psycopg2, which
        # this project does not ship. Same translation migrations/env.py does.
        url = dsn
        for prefix in ("postgresql+asyncpg://", "postgres://", "postgresql://"):
            if url.startswith(prefix):
                url = "postgresql+psycopg://" + url[len(prefix):]
                break
        store = SQLAlchemyJobStore(
            url=url, tablename="apscheduler_jobs",
            engine_options={
                # Bound how long boot can block on a job store. With SQLAlchemy's
                # default (none), an unreachable database made start() hang for
                # over two minutes before failing — measured at 130 s — and the
                # engine cannot trade until that returns. A schedule is not worth
                # two minutes of not being able to place a stop.
                "connect_args": {"connect_timeout": 5},
                # The store is idle between the daily job's fires, which is long
                # enough for a NAT or a database restart to have quietly dropped
                # the connection underneath it.
                "pool_pre_ping": True,
            })
        return store, "postgres"
    except Exception as exc:
        # ImportError (no psycopg installed) or a bad URL. The engine must still
        # start and trade — a scheduler that cannot persist is a degraded
        # scheduler, not a reason to refuse to run the book.
        log.warning("could not open the Postgres jobstore; scheduling falls "
                    "back to memory and will NOT survive a restart",
                    extra={"error": repr(exc)})
        return None, f"memory (fallback: {type(exc).__name__})"


def _new_scheduler(store: Any, timezone_name: str) -> Any:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    return AsyncIOScheduler(
        jobstores={"default": store} if store is not None else {},
        timezone=timezone_name,
        job_defaults={
            # See the module docstring: overlap is not slowness, it is
            # double-sizing against a stale account snapshot.
            "max_instances": 1,
            "coalesce": True,
        },
    )


def build_scheduler(
    *,
    dsn: str | None = None,
    timezone_name: str = "UTC",
) -> tuple[Any, str]:
    """An `AsyncIOScheduler` wired to this process's event loop.

    Returns `(scheduler, jobstore_description)` so the caller can log what it
    actually got rather than what it asked for.
    """
    store, described = _build_jobstore(dsn)
    return _new_scheduler(store, timezone_name), described


def start_scheduler(scheduler: Any, described: str,
                    *, timezone_name: str = "UTC") -> tuple[Any, str]:
    """Start it, falling back to an in-memory schedule if the store will not open.

    Constructing a `SQLAlchemyJobStore` does **not** connect — SQLAlchemy is
    lazy — so an unreachable database gets past `_build_jobstore` and only fails
    here, where `start()` runs `create_all`. The first version treated
    construction as the only failure point, and its test passed for the wrong
    reason: psycopg simply was not installed, so the bad-DSN case raised
    ImportError instead of reaching the real path.

    Trading must not stop because a job store is unavailable. The engine already
    holds an open database connection by this point, so this is a narrow window
    — but a scheduler is not worth refusing to run the book over.
    """
    try:
        scheduler.start()
        return scheduler, described
    except Exception as exc:
        log.warning("job store failed to start; falling back to an in-memory "
                    "schedule that will NOT survive a restart",
                    extra={"error": repr(exc), "jobstore": described})
        fallback = _new_scheduler(None, timezone_name)
        fallback.start()
        return fallback, f"memory (fallback: {type(exc).__name__})"


def schedule_engine_jobs(
    scheduler: Any,
    key: str,
    *,
    cycle_seconds: float,
    misfire_grace_seconds: int,
    promotion_hour: int = 0,
    promotion_minute: int = 5,
    with_promotion: bool = True,
) -> None:
    """Register the engine's jobs. Idempotent — `replace_existing=True`.

    **Call this AFTER `scheduler.start()`.** Before start, APScheduler holds
    additions in `_pending_jobs`, and `replace_existing` does not deduplicate
    against that list — so scheduling twice produces two trading cycles rather
    than replacing one. With a persistent jobstore that survives the process, so
    the duplicate is permanent.

    `promotion_minute=5` rather than 0: a job at exactly midnight competes with
    every other midnight cron on the box, and nothing here needs the first
    minute of the day.
    """
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler.add_job(
        run_cycle,
        trigger=IntervalTrigger(seconds=cycle_seconds),
        args=[key],
        id=CYCLE_JOB_ID,
        name="trading cycle",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=misfire_grace_seconds,
        # NOTE: do not pass `next_run_time=None` here. In APScheduler an
        # *explicit* None means "add this job PAUSED" — it is not the same as
        # omitting the argument, which means "compute the first fire from the
        # trigger". Passing None silently produced a scheduler with a
        # correctly-configured trading cycle that never ran once. Caught by
        # test_cycles_never_overlap seeing zero calls.
    )

    if with_promotion:
        scheduler.add_job(
            run_promotion,
            # timezone= is required, not cosmetic. APScheduler only applies the
            # scheduler's timezone when it builds the trigger itself from
            # trigger="cron" plus kwargs; a PRE-BUILT CronTrigger keeps whatever
            # timezone it was constructed with, which defaults to the machine's
            # local zone. So a scheduler configured for UTC was firing this at
            # 00:05 *local* — 07:05 UTC on the development machine, 06:05 UTC on
            # a NAS set to America/Chicago — while the README said 00:05 UTC.
            # Caught only by running it against a real database and reading the
            # stored next_run_time.
            #
            # It matters beyond the documentation: `evaluate_promotion` guards on
            # the UTC date, so a local-time trigger makes which UTC day gets
            # evaluated depend on the deployment's timezone.
            trigger=CronTrigger(hour=promotion_hour, minute=promotion_minute,
                                timezone=scheduler.timezone),
            args=[key],
            id=PROMOTION_JOB_ID,
            name="daily promotion evaluation",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            # Generous, unlike the cycle's: promotion is idempotent and reading
            # a stale-by-an-hour record is strictly better than the arm-live
            # gate having no record at all.
            misfire_grace_time=3600,
        )


def describe_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """For the boot log and `status()`. A schedule nobody can see is a schedule
    nobody notices has stopped."""
    out = []
    for job in scheduler.get_jobs():
        nxt = getattr(job, "next_run_time", None)
        out.append({
            "id": job.id,
            "name": job.name,
            "trigger": str(job.trigger),
            "next_run": nxt.isoformat() if nxt else None,
        })
    return out


__all__ = [
    "CYCLE_JOB_ID",
    "PROMOTION_JOB_ID",
    "build_scheduler",
    "describe_jobs",
    "register_engine",
    "registered_engine",
    "run_cycle",
    "run_promotion",
    "schedule_engine_jobs",
    "unregister_engine",
]
