"""structlog wiring.

The point of these tests is that the log stays *diagnosable*. Two things broke
during implementation and both would have shipped silently, because a log that
is missing information still looks like a log:

  * `extra={...}` was dropped, so structured call sites rendered identically to
    unstructured ones — structured-looking output carrying no structure;
  * `exc_info` rendered as the raw tuple repr, i.e. an exception class and a
    memory address with zero frames, on the exact line (`log.exception("cycle
    failed")`) you read after an incident.
"""

from __future__ import annotations

import json
import logging

import pytest

from ct.logging import configure, reset_for_tests


@pytest.fixture(autouse=True)
def _fresh_logging():
    """Save and restore the root handlers around every test in this file.

    `configure()` deliberately removes existing root handlers — replacing
    basicConfig's is its job. In a test session that also removes pytest's
    `caplog` handler, and because the removal outlives the test, a later test in
    a *different* file that asserts on caplog silently records nothing. That is
    exactly what happened: test_web's malformed-hash test passed alone and
    failed in the full run.
    """
    import structlog

    root = logging.getLogger()
    saved_level = root.level
    reset_for_tests()
    try:
        yield
    finally:
        reset_for_tests()
        # Detach only what this module installed. Restoring a snapshot of
        # root.handlers would itself be hostile — it would resurrect handlers
        # another fixture had deliberately removed.
        for h in root.handlers[:]:
            if isinstance(getattr(h, "formatter", None),
                          structlog.stdlib.ProcessorFormatter):
                root.removeHandler(h)
        root.setLevel(saved_level)


def _emit(capsys, fmt, fn):
    configure(level="INFO", fmt=fmt, force=True)
    fn(logging.getLogger("ct.test"))
    return capsys.readouterr().err


def _json_lines(err):
    return [json.loads(x) for x in err.strip().splitlines() if x.strip()]


def test_json_line_has_the_core_fields(capsys):
    err = _emit(capsys, "json", lambda log: log.info("hello"))
    (rec,) = _json_lines(err)
    assert rec["event"] == "hello"
    assert rec["level"] == "info"
    assert rec["logger"] == "ct.test"
    assert rec["timestamp"].endswith("Z"), "timestamps must be UTC"


def test_extra_is_promoted_to_real_fields(capsys):
    """Regression: `extra` was attached to the LogRecord then silently dropped."""
    err = _emit(capsys, "json", lambda log: log.warning(
        "order rejected", extra={"symbol": "BTC/USD", "qty": 0.25}))
    (rec,) = _json_lines(err)
    assert rec["symbol"] == "BTC/USD"
    assert rec["qty"] == 0.25


def test_extra_cannot_clobber_core_fields(capsys):
    err = _emit(capsys, "json", lambda log: log.info(
        "x", extra={"level": "nonsense", "logger": "fake"}))
    (rec,) = _json_lines(err)
    assert rec["level"] == "info"
    assert rec["logger"] == "ct.test"


def test_stdlib_record_internals_do_not_leak(capsys):
    """Promoting extras must not also dump lineno/pathname/process into output."""
    err = _emit(capsys, "json", lambda log: log.info("x"))
    (rec,) = _json_lines(err)
    for noise in ("pathname", "lineno", "process", "threadName", "msg", "args"):
        assert noise not in rec, f"{noise} leaked into the log line"


def test_json_tracebacks_carry_frames(capsys):
    """Regression: this rendered as the repr of a traceback object — no frames."""
    def fn(log):
        try:
            def inner():
                raise ValueError("boom")
            inner()
        except ValueError:
            log.exception("cycle failed")

    (rec,) = _json_lines(_emit(capsys, "json", fn))
    exc = rec["exception"]
    assert isinstance(exc, list) and exc, f"no structured exception: {exc!r}"
    assert exc[0]["exc_type"] == "ValueError"
    assert exc[0]["exc_value"] == "boom"
    names = [f["name"] for f in exc[0]["frames"]]
    assert "inner" in names, f"the raising frame is missing: {names}"
    assert "<traceback object at" not in json.dumps(rec)


def test_console_renders_a_real_traceback(capsys):
    def fn(log):
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("cycle failed")

    err = _emit(capsys, "console", fn)
    assert "Traceback (most recent call last)" in err
    assert "ValueError: boom" in err


def test_logs_go_to_stderr_not_stdout(capsys):
    """`qd backtest --json` writes a machine-readable payload to stdout; a log
    line in the middle of it corrupts the document. A test caught that once."""
    configure(level="INFO", fmt="json", force=True)
    logging.getLogger("ct.test").info("noise")
    out = capsys.readouterr()
    assert out.out == ""
    assert "noise" in out.err


def _our_handlers():
    """Handlers this module installed, identified by their structlog formatter.

    Not `len(root.handlers)`: pytest attaches four of its own (a live-log null
    handler, a file handler on nul, and two capture handlers), and asserting on
    the total would be asserting on pytest's internals.
    """
    import structlog

    return [h for h in logging.getLogger().handlers
            if isinstance(getattr(h, "formatter", None),
                          structlog.stdlib.ProcessorFormatter)]


def test_configure_is_idempotent(capsys):
    """`--healthcheck` imports the same module the long-running process does and
    compose runs it every 30s; re-adding handlers would multiply every line."""
    configure(level="INFO", fmt="json", force=True)
    for _ in range(5):
        configure(level="INFO", fmt="json")
    assert len(_our_handlers()) == 1
    logging.getLogger("ct.test").info("once")
    assert len(_json_lines(capsys.readouterr().err)) == 1, "line was duplicated"


def test_force_reconfigure_does_not_stack_handlers(capsys):
    for _ in range(4):
        configure(level="INFO", fmt="json", force=True)
    assert len(_our_handlers()) == 1
    logging.getLogger("ct.test").info("once")
    assert len(_json_lines(capsys.readouterr().err)) == 1


def test_configure_leaves_foreign_handlers_alone():
    """Regression: clearing root.handlers wholesale deleted pytest's caplog
    handler, so a later test in another file captured nothing and failed only in
    the full run — the worst shape a failure can have."""
    root = logging.getLogger()
    marker = logging.NullHandler()
    root.addHandler(marker)
    try:
        configure(level="INFO", fmt="json", force=True)
        assert marker in root.handlers, "configure() removed a handler it did not add"
    finally:
        root.removeHandler(marker)


def test_noisy_third_parties_are_capped(capsys):
    configure(level="DEBUG", fmt="json", force=True)
    logging.getLogger("apscheduler.scheduler").info("added job")
    logging.getLogger("httpx").info("HTTP Request: GET ...")
    assert capsys.readouterr().err.strip() == ""


def test_level_is_honoured(capsys):
    configure(level="WARNING", fmt="json", force=True)
    log = logging.getLogger("ct.test")
    log.info("suppressed")
    log.warning("kept")
    lines = _json_lines(capsys.readouterr().err)
    assert [r["event"] for r in lines] == ["kept"]


def test_auto_format_picks_json_when_not_a_tty(capsys):
    """Containers are not TTYs, and the NAS ships logs to json-file."""
    configure(level="INFO", fmt="auto", force=True)
    logging.getLogger("ct.test").info("hello")
    err = capsys.readouterr().err
    assert json.loads(err.strip())["event"] == "hello"


def test_existing_call_sites_need_no_change(capsys):
    """~40 modules use `logging.getLogger(__name__)`. That must keep working —
    structlog is installed as a formatter, not as a replacement API."""
    configure(level="INFO", fmt="json", force=True)
    logging.getLogger("ct.risk.engine").info("gate passed: %s", "exposure")
    (rec,) = _json_lines(capsys.readouterr().err)
    assert rec["event"] == "gate passed: exposure"
    assert rec["logger"] == "ct.risk.engine"
