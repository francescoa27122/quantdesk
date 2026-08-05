"""`python -m ct.engine` — the qd-engine container entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from decimal import Decimal

log = logging.getLogger("ct.engine")


async def _healthcheck() -> int:
    """Compose healthcheck. Proves the process imports and the database
    answers — deliberately not more, because a health check that claims more
    than it verifies is itself a source of false confidence."""
    dsn = os.environ.get("QD_DB_DSN")
    if not dsn:
        print("QD_DB_DSN not set")
        return 1
    from ct.store import Database
    db = Database(dsn, max_size=1)
    try:
        await db.connect(retries=1)
        ok = await db.healthy()
    except Exception as exc:
        print(f"unhealthy: {exc!r}")
        return 1
    finally:
        await db.close()
    print("ok" if ok else "database unreachable")
    return 0 if ok else 1


async def main(argv: list[str] | None = None) -> int:
    from ct.logging import configure as _configure_logging

    _configure_logging()
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--healthcheck" in argv:
        return await _healthcheck()

    from ct.config import settings, strategy_config
    from ct.engine import Allocation, KillSwitch, TradingEngine
    from ct.events import EventBus, JsonlEventLog
    from ct.execution.broker import PaperBroker
    from ct.store import Database, PostgresEventSink, TeeEventLog
    from ct.strategies import REGISTRY

    mode = os.environ.get("QD_MODE", "paper")
    if mode == "live":
        # The live broker is a separate deliverable behind the arming
        # ceremony. Refusing loudly is the only correct behaviour: silently
        # falling back to paper while the operator believes it is live is
        # strictly worse than not starting.
        log.error("QD_MODE=live but no live broker is implemented in this "
                  "build. Refusing to start rather than silently paper-trading "
                  "while you believe otherwise.")
        return 2

    dsn = os.environ["QD_DB_DSN"]
    symbols = tuple(s.strip() for s in
                    os.environ.get("QD_SYMBOLS", "BTC/USD,ETH/USD").split(",")
                    if s.strip())
    timeframe = os.environ.get("QD_TIMEFRAME", "4h")
    names = [s.strip() for s in
             os.environ.get("QD_STRATEGIES", "donchian_breakout").split(",")
             if s.strip()]

    db = Database(dsn)
    await db.connect()
    sink = PostgresEventSink(db)
    await sink.start()
    bus = EventBus(TeeEventLog(JsonlEventLog("/app/logs/events"), sink),
                   source="engine")

    # Strategy parameters come from `config/strategies.yml`, not from code
    # defaults. This line was `REGISTRY[n]()` — no parameters at all — so the
    # engine traded whatever DEFAULTS happened to say, and nothing recorded
    # which numbers those were. ARCHITECTURE.md §2 has specified versioned YAML
    # since the design was written; this is that.
    #
    # A strategy absent from the file still constructs on its own DEFAULTS.
    # That is deliberate: an unlisted strategy should run, not crash the
    # engine at boot. It is logged at WARNING because running unattributable
    # parameters is worth noticing.
    cfg = strategy_config()
    log.info("strategy config loaded", extra={"config": cfg.describe(),
                                              "config_hash": cfg.config_hash()})
    allocations = []
    for n in names:
        params = cfg.params_for(n)
        if not params:
            log.warning("no config entry; running on code defaults",
                        extra={"strategy": n})
        allocations.append(Allocation(
            REGISTRY[n](**params),
            cfg.symbols_for(n, symbols),
            cfg.timeframe_for(n, timeframe),
            weight=1.0 / max(1, len(names))))
    engine = TradingEngine(
        allocations, PaperBroker(), db=db,
        # Was never wired: QD_CYCLE_SECONDS was documented and ignored, so the
        # engine always ran at the constructor's 20 s whatever the deployment
        # said.
        cycle_seconds=settings().cycle_seconds,
        account_id=int(os.environ.get("QD_ACCOUNT_ID", "1")),
        bus=bus, kill_switch=KillSwitch(
            os.environ.get("QD_KILL_SWITCH_FILE", "/app/state/KILL")),
        initial_equity=Decimal(os.environ.get("QD_INITIAL_EQUITY", "10000")),
        mode=mode)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, engine.stop)
    try:
        await engine.run_forever()
    finally:
        await engine.shutdown()
        await sink.stop()
        bus.close()
        await db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
