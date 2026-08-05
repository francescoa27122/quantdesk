"""Config layer: pydantic-settings over env/secret-files, YAML strategy params.

The most valuable test here is `test_settings_defaults_match_the_call_sites`.
A Settings class is only an improvement if it is faithful; one that quietly
changes a default moves the kill-switch path or the account id without anybody
editing a deployment. Three defaults were wrong on the first pass — the kill
switch pointed at HALT instead of KILL, the account id was a string, and the SSE
cap was 900 instead of 600 — so this asserts them against the real values rather
than trusting a second reading.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ct.config import (
    Settings,
    StrategyConfig,
    StrategyConfigError,
    load_strategy_config,
)
from ct.strategies import REGISTRY

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "strategies.yml"


# ── Settings ─────────────────────────────────────────────────────────────


def test_settings_defaults_match_the_call_sites():
    """Each default equals what the pre-existing os.environ.get() used."""
    s = Settings()
    assert s.mode == "paper"
    assert s.exchange == "kraken"
    assert s.account_id == 1
    assert s.symbols == ("BTC/USD", "ETH/USD")
    assert s.timeframe == "4h"
    assert s.timeframes == ("1h", "4h", "1d")
    assert s.strategies == ("donchian_breakout",)
    assert s.initial_equity == 10_000.0
    assert s.kill_switch_file == "/app/state/KILL"
    assert s.sse_max_seconds == 600
    assert s.host_health == "/hosthealth/metrics.json"
    assert s.log_level == "INFO"


def test_no_env_default_drifted_from_source():
    """Catch a default that changes in ct/ but not here, and vice versa.

    Greps the remaining os.environ.get("QD_…", "default") call sites and compares
    each against the Settings field of the same name. This is the guard that
    makes an incremental migration safe: until every call site is converted, the
    two sources of truth must agree.
    """
    pat = re.compile(r'os\.environ\.get\(\s*"QD_([A-Z_]+)"\s*,\s*"([^"]*)"\s*\)')
    mismatches = []
    for py in (REPO / "ct").rglob("*.py"):
        for var, literal in pat.findall(py.read_text(encoding="utf-8")):
            field = var.lower()
            if field not in Settings.model_fields:
                continue
            # The DECLARED default, not an instantiated Settings(). Other tests
            # in this suite set QD_* in the environment (test_web sets a session
            # secret), and instantiating here would compare the source literal
            # against whatever those left behind rather than against the default.
            actual = Settings.model_fields[field].default
            if isinstance(actual, tuple):
                expected = tuple(x.strip() for x in literal.split(",") if x.strip())
            elif isinstance(actual, bool):
                continue
            elif isinstance(actual, int) and not isinstance(actual, bool):
                expected = int(literal)
            elif isinstance(actual, float):
                expected = float(literal)
            else:
                expected = literal
            if actual != expected:
                mismatches.append(
                    f"{py.relative_to(REPO)}: QD_{var} defaults to {literal!r} "
                    f"but Settings.{field} is {actual!r}")
    assert not mismatches, "\n".join(mismatches)


def test_csv_env_vars_parse_as_lists(monkeypatch):
    """`QD_SYMBOLS=A,B` is the format every deployment already uses.

    Without the mode="before" validator pydantic tries to JSON-decode a complex
    field, and a plain comma-separated value fails validation at import time —
    i.e. the container would not start.
    """
    monkeypatch.setenv("QD_SYMBOLS", "SOL/USD, BTC/USD ,")
    monkeypatch.setenv("QD_STRATEGIES", "ema_cross")
    s = Settings()
    assert s.symbols == ("SOL/USD", "BTC/USD")
    assert s.strategies == ("ema_cross",)


def test_secret_file_indirection(monkeypatch, tmp_path):
    """QD_<NAME>_FILE is read from disk — the Docker-secrets path."""
    f = tmp_path / "sec"
    f.write_text("  s3cret-value\n", encoding="utf-8")
    monkeypatch.setenv("QD_SESSION_SECRET_FILE", str(f))
    monkeypatch.delenv("QD_SESSION_SECRET", raising=False)
    assert Settings().session_secret == "s3cret-value"


def test_direct_env_beats_the_file(monkeypatch, tmp_path):
    f = tmp_path / "sec"
    f.write_text("from-file", encoding="utf-8")
    monkeypatch.setenv("QD_SESSION_SECRET_FILE", str(f))
    monkeypatch.setenv("QD_SESSION_SECRET", "from-env")
    assert Settings().session_secret == "from-env"


@pytest.mark.parametrize("content", ["", "   \n"])
def test_empty_secret_file_leaves_the_value_unset(monkeypatch, tmp_path, content):
    """An empty secret that looks *set* is worse than one obviously absent.

    ct/web/security.py refuses every login when no password hash is configured.
    That only works if unset stays unset.
    """
    f = tmp_path / "sec"
    f.write_text(content, encoding="utf-8")
    monkeypatch.setenv("QD_SESSION_SECRET_FILE", str(f))
    monkeypatch.delenv("QD_SESSION_SECRET", raising=False)
    assert Settings().session_secret == ""


def test_missing_secret_file_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setenv("QD_SESSION_SECRET_FILE", str(tmp_path / "nope"))
    monkeypatch.delenv("QD_SESSION_SECRET", raising=False)
    assert Settings().session_secret == ""


def test_unrelated_container_env_is_ignored(monkeypatch):
    """The containers carry TZ, PATH, PGHOST and more. extra='ignore' or the
    process refuses to start on a variable that is none of its business."""
    monkeypatch.setenv("QD_SOMETHING_NOBODY_DECLARED", "x")
    assert Settings().mode == "paper"


def test_mode_is_constrained(monkeypatch):
    monkeypatch.setenv("QD_MODE", "definitely-not-a-mode")
    with pytest.raises(ValidationError):
        Settings()


# ── strategy YAML ────────────────────────────────────────────────────────


def test_shipped_config_is_valid_and_every_param_is_real():
    """A parameter the strategy does not read is a walk-forward optimising a
    number nothing consumes. Construct each strategy for real — that runs the
    strategy's own unknown-parameter check."""
    cfg = load_strategy_config(CONFIG)
    assert cfg.version >= 1
    assert cfg.names(), "shipped config lists no strategies"
    for name in cfg.names():
        assert name in REGISTRY, f"config names unknown strategy {name!r}"
        REGISTRY[name](**cfg.params_for(name))


def test_shipped_config_covers_every_registered_strategy():
    """A strategy missing from the file silently runs on code defaults, which
    is the unattributable-parameter problem the file exists to solve."""
    cfg = load_strategy_config(CONFIG)
    assert set(cfg.names()) == set(REGISTRY), (
        f"missing from YAML: {sorted(set(REGISTRY) - set(cfg.names()))}; "
        f"unknown in YAML: {sorted(set(cfg.names()) - set(REGISTRY))}")


def test_config_hash_ignores_formatting_but_not_content(tmp_path):
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    c = tmp_path / "c.yml"
    a.write_text(
        "version: 1\n"
        "strategies:\n  ema_cross:\n    params:\n      fast: 20\n      slow: 50\n")
    # Same content: keys reordered, one of them quoted, blocks swapped.
    b.write_text(
        "strategies:\n  ema_cross:\n    params:\n"
        '      slow: 50\n      "fast": 20\n'
        "version: 1\n")
    # A real change: fast 20 -> 21.
    c.write_text(
        "version: 1\n"
        "strategies:\n  ema_cross:\n    params:\n      fast: 21\n      slow: 50\n")
    ha = load_strategy_config(a).config_hash()
    assert ha == load_strategy_config(b).config_hash(), "reformatting changed the hash"
    assert ha != load_strategy_config(c).config_hash(), "a real change did not change the hash"


def test_missing_file_is_not_an_error(tmp_path):
    """Research on a laptop and the test suite both run without one."""
    cfg = load_strategy_config(tmp_path / "absent.yml")
    assert cfg.version == 0
    assert cfg.names() == []
    assert cfg.params_for("ema_cross") == {}


@pytest.mark.parametrize("body", [
    "strategies: {}",                                   # no version
    "version: 'one'\nstrategies: {}",                   # version not an int
    "version: 1\nstrategies: [a, b]",                   # strategies not a mapping
    "version: 1\nstrategies:\n  ema_cross: 12",         # entry not a mapping
    "version: 1\nstrategies:\n  ema_cross:\n    params: 7",   # params not a mapping
])
def test_malformed_config_is_fatal(tmp_path, body):
    """Falling back to defaults would trade a parameter set nobody chose, and
    attribute the result to a file that was never read."""
    p = tmp_path / "bad.yml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(StrategyConfigError):
        load_strategy_config(p)


def test_invalid_yaml_is_fatal(tmp_path):
    p = tmp_path / "bad.yml"
    p.write_text("version: 1\nstrategies:\n  - [unclosed\n", encoding="utf-8")
    with pytest.raises(StrategyConfigError):
        load_strategy_config(p)


def test_params_are_not_merged_with_code_defaults():
    """The hash must cover decisions, not whatever the code defaults to.

    If DEFAULTS were folded in, editing a default in Python would change the
    hash of every historical record and silently invalidate the reproducibility
    guarantee that `backtests` UNIQUE constraint rests on.
    """
    cfg = StrategyConfig({"version": 1, "strategies": {
        "ema_cross": {"params": {"fast": 5}}}})
    assert cfg.params_for("ema_cross") == {"fast": 5}
    assert "slow" not in cfg.params_for("ema_cross")
    # ...but constructing still fills the rest in from DEFAULTS.
    assert REGISTRY["ema_cross"](**cfg.params_for("ema_cross")).params["slow"] == 50


def test_enabled_is_config_opinion_only():
    cfg = load_strategy_config(CONFIG)
    assert "donchian_breakout" in cfg.enabled_names()
    # Everything else ships off: none of them has passed the promotion gate.
    assert cfg.enabled_names() == ["donchian_breakout"]


def test_accessors_fall_back_to_the_supplied_default():
    cfg = StrategyConfig({"version": 3, "strategies": {"ema_cross": {}}})
    assert cfg.symbols_for("ema_cross", ("BTC/USD",)) == ("BTC/USD",)
    assert cfg.timeframe_for("ema_cross", "1h") == "1h"
    assert cfg.weight_for("ema_cross") == 0.0
    assert cfg.symbols_for("not-present", ("X",)) == ("X",)


def test_shipped_yaml_parses_as_plain_yaml():
    """Guards against a tab or a smart quote sneaking into the file."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict) and "strategies" in raw
