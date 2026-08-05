"""Structured logging. One `configure()` call per entrypoint.

ARCHITECTURE.md lists `structlog` as the logging stack; the code used three
separate `logging.basicConfig()` calls with three different format strings and
two different default levels. That is not merely untidy — it means the thing you
grep at 3 a.m. has a different shape depending on which container produced it.

## What this does and does not change

**Call sites do not change.** Every module keeps `log = logging.getLogger(
__name__)` and keeps calling `log.info("...", arg)`. structlog is installed as a
*formatter* underneath stdlib logging via `ProcessorFormatter`, not as a
replacement API. That matters for a codebase with ~40 existing call sites: a
migration that required touching all of them would be a large diff through the
trading path to change how text is rendered, which is a bad trade.

Modules that want structure can opt in per call:

    log.info("order filled", extra={"symbol": sym, "qty": qty})

and the keys land as first-class fields in JSON output rather than being
interpolated into a sentence a machine then has to re-parse.

## Two renderers

`json` in containers, because the NAS ships logs to `json-file` with a size cap
and a JSON line survives being truncated, grepped, and read by something that is
not a person. `console` when a human is at a terminal — the CLI is a research
tool and a wall of JSON would make it unusable.

`auto` picks: a TTY means console, anything else means JSON. That is the right
default because it is correct in both places without configuration, and being
wrong is cosmetic rather than dangerous.

## Levels

The default is `INFO`, matching the engine and collector. The CLI asks for
`WARNING` explicitly rather than by reading the same environment variable with a
different default — a second default for the same variable is invisible drift,
and `tests/test_config.py::test_no_env_default_drifted_from_source` now fails if
anybody reintroduces one.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

_configured = False
# The handler this module installed. Tracked so a re-configure replaces its own
# work and leaves every foreign handler alone.
_handler: logging.Handler | None = None

# Noisy third parties. These are libraries whose INFO is our DEBUG: apscheduler
# narrates every job it adds, and httpx logs a line per request — which on the
# collector is one line per candle fetch.
_NOISY = {
    "apscheduler": logging.WARNING,
    "apscheduler.scheduler": logging.WARNING,
    "apscheduler.executors.default": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "asyncio": logging.WARNING,
    "ccxt": logging.WARNING,
}


# Everything the stdlib puts on a LogRecord itself. Anything NOT in here was
# put there by a caller passing extra={...}, and is what we want to promote.
_STDLIB_RECORD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
})


def _promote_extras(_logger: Any, _name: str,
                    event_dict: dict[str, Any]) -> dict[str, Any]:
    """Lift `log.info("msg", extra={"symbol": "BTC/USD"})` into real fields.

    Without this, `extra=` is attached to the LogRecord and then silently
    dropped by ProcessorFormatter, so structured call sites would render
    identically to unstructured ones — the log would *look* structured while
    carrying none of the structure. Existing fields win, so a caller cannot
    accidentally overwrite `level` or `timestamp`.
    """
    record = event_dict.get("_record")
    if record is None:
        return event_dict
    for key, value in vars(record).items():
        if key.startswith("_") or key in _STDLIB_RECORD_ATTRS:
            continue
        event_dict.setdefault(key, value)
    return event_dict


def _shared_processors() -> list[Any]:
    return [
        _promote_extras,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # UTC, always. A trading log in local time is a log you cannot line up
        # against an exchange's fills without knowing which side of a DST
        # boundary each line fell.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def configure(
    level: str | int | None = None,
    fmt: Literal["json", "console", "auto"] | None = None,
    *,
    force: bool = False,
) -> None:
    """Install structlog under stdlib logging. Idempotent.

    Idempotence is not politeness: `python -m ct.engine --healthcheck` imports
    the same module the long-running process does, and the compose healthcheck
    runs it every 30 seconds. Re-adding handlers each time would multiply every
    log line by the number of health checks since boot.
    """
    global _configured
    if _configured and not force:
        return

    from ct.config import settings

    s = settings()
    level = level if level is not None else s.log_level
    fmt = fmt or s.log_format
    if fmt == "auto":
        fmt = "console" if sys.stderr.isatty() else "json"

    # Tracebacks. Without an explicit exc_info processor the JSON renderer emits
    # the raw (type, value, traceback-object) tuple's repr — which contains the
    # exception class and a memory address, and no frames at all. The engine's
    # `log.exception("cycle failed")` is precisely the line you read after an
    # incident, so this is the difference between a diagnosable failure and a
    # useless one.
    #   json    -> dict_tracebacks: frames as structured data, greppable by field
    #   console -> format_exc_info: the familiar rendered traceback
    renderer: Any
    exc_processor: Any
    if fmt == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        exc_processor = structlog.processors.format_exc_info
    else:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
        exc_processor = structlog.processors.dict_tracebacks

    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Runs only for records that came through plain stdlib logging, i.e.
        # every existing `log.info("...")` call site in this repository.
        foreign_pre_chain=_shared_processors(),
        processors=[
            exc_processor,
            # Must run AFTER exc_processor: the exception formatters read
            # `_record` to recover exc_info, and this is what deletes it.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # stderr, not stdout. `qd backtest --json` writes a machine-readable payload
    # to stdout and a log line in the middle of it corrupts the document — a
    # test caught exactly that once already.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Remove only the handler WE installed on a previous call — never a foreign
    # one. The first version cleared root.handlers wholesale, which is fine in a
    # container and wrong everywhere else: under pytest it deleted caplog's
    # handler, so a later test in a different file asserting on log records
    # silently captured nothing. `test_web`'s malformed-hash test passed alone
    # and failed in the full run, which is the worst shape a test failure can
    # have. The same trap would catch anyone attaching a file or syslog handler
    # before calling this.
    #
    # Nothing calls logging.basicConfig() in this codebase any more (that was
    # the point), so there is no default handler left to displace.
    global _handler
    root = logging.getLogger()
    if _handler is not None and _handler in root.handlers:
        root.removeHandler(_handler)
    root.addHandler(handler)
    _handler = handler
    root.setLevel(level)

    for name, lvl in _NOISY.items():
        logging.getLogger(name).setLevel(lvl)

    _configured = True


def reset_for_tests() -> None:
    global _configured
    _configured = False


def get_logger(name: str | None = None) -> Any:
    """A structlog logger, for new code that wants to bind context.

    Existing modules keep using `logging.getLogger(__name__)`; both render
    identically because they share the same handler and processor chain.
    """
    return structlog.stdlib.get_logger(name)


__all__ = ["configure", "get_logger", "reset_for_tests"]
