"""CLI tests.

The load-bearing ones are the **refusals**. A CLI that runs a backtest is easy;
the parts worth testing are the ones that decline: `arm-live` without a
terminal, `arm-live` with a mistyped phrase, `arm-live` past a promotion gate
that never ran, a backtest against fewer bars than the strategy's warmup. Every
one of those is a place where exiting 0 with a cheerful message would be worse
than failing.

No database and no network. `ct.cli.database` is monkeypatched to a `FakeDB`
that records SQL, so the *arming write path* is exercised without any real
account existing anywhere. Nothing in this file can arm anything real.

Market data comes from generated Kraken-format CSVs, which is also the `--csv`
path a person would use on a laptop with no NAS.
"""

from __future__ import annotations

import contextlib
import math
import sys
import types
import zipfile
from pathlib import Path

import pytest

from ct import cli

# ── fixtures ─────────────────────────────────────────────────────────────

# Deterministic, no RNG: a test that depends on the stdlib's random
# implementation is a test that can change meaning on a Python upgrade.
FOUR_HOURS = 14_400
EPOCH = 1_600_000_000


def _csv_rows(n: int, base: float, phase: float) -> str:
    rows = []
    for i in range(n):
        drift = 1.0 + 0.0009 * i
        wave = 1.0 + 0.22 * math.sin(i / 17.0 + phase) + 0.06 * math.sin(i / 3.7)
        close = base * drift * wave
        open_ = close * 0.999
        high = close * 1.006
        low = close * 0.994
        rows.append(f"{EPOCH + i * FOUR_HOURS},{open_:.2f},{high:.2f},{low:.2f},"
                    f"{close:.2f},{120.0 + (i % 40):.4f},{50 + i % 200}")
    return "\n".join(rows) + "\n"


def write_kraken_dir(root: Path, bars: int = 300) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "XBTUSD_240.csv").write_text(_csv_rows(bars, 30_000.0, 0.0),
                                         encoding="utf-8")
    (root / "ETHUSD_240.csv").write_text(_csv_rows(bars, 2_000.0, 1.3),
                                         encoding="utf-8")
    return root


@pytest.fixture
def kraken(tmp_path: Path) -> Path:
    return write_kraken_dir(tmp_path / "kraken")


@pytest.fixture
def btc_csv(kraken: Path) -> Path:
    return kraken / "XBTUSD_240.csv"


class FakeDB:
    """Records every statement and answers by SQL substring.

    Deliberately dumb: the point is to assert *which* statements a command
    issues and in what shape, not to re-implement Postgres. Anything the real
    database enforces (the arming CHECK constraints, `arming_is_valid`) is
    tested in `tests/test_store.py` against a real TimescaleDB.
    """

    def __init__(self, rows: dict[str, list] | None = None,
                 vals: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.rows = rows or {}
        self.vals = vals or {}

    def _match(self, table: dict, sql: str):
        for needle, value in table.items():
            if needle in " ".join(sql.split()):
                return value
        return None

    def sql_containing(self, needle: str) -> list[tuple[str, tuple]]:
        return [c for c in self.calls if needle in " ".join(c[0].split())]

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self._match(self.rows, sql) or []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        got = self._match(self.rows, sql)
        return got[0] if got else None

    async def fetchval(self, sql, *args):
        self.calls.append((sql, args))
        return self._match(self.vals, sql)

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "OK"

    async def executemany(self, sql, rows):
        self.calls.append((sql, tuple(rows)))

    async def bootstrap(self, schema_path):
        self.calls.append((f"BOOTSTRAP {schema_path}", ()))
        return bool(self.vals.get("BOOTSTRAP_APPLIED", False))

    async def table_names(self):
        return {"orders", "fills", "candles"}


@pytest.fixture
def fake_db(monkeypatch):
    # Every INSERT ... RETURNING id in the codebase is wrapped in int(), so a
    # fake that answers None turns into a TypeError three frames away.
    db = FakeDB(vals={"INSERT INTO control_commands": 1,
                      "INSERT INTO notifications": 2,
                      "INSERT INTO live_arming": 3})

    @contextlib.asynccontextmanager
    async def _database(**_kw):
        yield db

    monkeypatch.setattr(cli, "database", _database)
    return db


@pytest.fixture
def no_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


@pytest.fixture
def a_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))


# ── parsing helpers ──────────────────────────────────────────────────────


def test_parse_value_types():
    assert cli.parse_value("10") == 10
    assert isinstance(cli.parse_value("10"), int)
    assert cli.parse_value("0.5") == 0.5
    assert cli.parse_value("true") is True
    assert cli.parse_value("FALSE") is False
    assert cli.parse_value("none") is None
    assert cli.parse_value("BTC/USD") == "BTC/USD"


def test_split_list_accepts_both_forms():
    assert cli.split_list(["A,B", "C"]) == ["A", "B", "C"]
    assert cli.split_list(None, "BTC/USD") == ["BTC/USD"]
    assert cli.split_list([]) == []


def test_parse_params_and_space():
    assert cli.parse_params(["fast=5", "long_only=false"]) == {
        "fast": 5, "long_only": False}
    assert cli.parse_space(["fast=5,10,20"]) == {"fast": [5, 10, 20]}


def test_parse_params_rejects_a_bare_word():
    with pytest.raises(cli.CliError) as exc:
        cli.parse_params(["fast"])
    assert exc.value.code == cli.EXIT_USAGE


def test_parse_ts_assumes_utc():
    ts = cli.parse_ts("2026-07-01")
    assert ts is not None and ts.tzinfo is not None
    assert cli.parse_ts(None) is None
    with pytest.raises(cli.CliError):
        cli.parse_ts("last tuesday")


# ── the parser surface ───────────────────────────────────────────────────


DOCUMENTED = {
    "backtest", "walkforward", "promotion-status", "arm-live", "disarm",
    "hash-password", "backfill", "import-history", "status", "halt", "resume",
    "flatten", "kill", "coverage", "init-db", "arming-status",
    "pause-strategy", "resume-strategy", "strategies", "enable-strategy",
}


def test_every_documented_command_exists():
    """S4-1: README documented commands that could not run. This is the guard
    against that regressing one command at a time."""
    sub = next(a for a in cli.build_parser()._subparsers._group_actions
               if hasattr(a, "choices"))
    assert set(sub.choices) >= DOCUMENTED


def test_no_command_prints_help_and_exits_usage(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["--json", "backtest", "--strategy", "ema_cross"],
    ["backtest", "--json", "--strategy", "ema_cross"],
])
def test_json_flag_survives_on_either_side_of_the_subcommand(argv):
    """argparse shares parent actions between subparsers and `set_defaults`
    mutates them in place, so this silently broke the pre-subcommand form."""
    assert cli.build_parser().parse_args(argv).json is True


def test_json_defaults_to_false():
    assert cli.build_parser().parse_args(["status"]).json is False


def test_arm_live_has_no_flag_that_could_supply_the_phrase():
    """The structural half of the money boundary: if no flag can carry the
    confirmation, no script can pass one."""
    parser = cli.build_parser()
    sub = next(a for a in parser._subparsers._group_actions
               if hasattr(a, "choices"))
    options = {o for act in sub.choices["arm-live"]._actions
               for o in act.option_strings}
    assert not {o for o in options
                if any(w in o for w in ("confirm", "phrase", "yes", "force-yes"))}
    for bad in (["arm-live", "--confirm", "ARM LIVE main 250.00 USD",
                 "--max-notional", "250"],
                ["arm-live", "--phrase", "x", "--max-notional", "250"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(bad)
        assert exc.value.code == cli.EXIT_USAGE


# ── backtest ─────────────────────────────────────────────────────────────


def test_backtest_single_symbol_from_csv(btc_csv, capsys):
    code = cli.main(["backtest", "--strategy", "ema_cross", "--symbol",
                     "BTC/USD", "--csv", str(btc_csv)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "ema_cross on BTC/USD 4h" in out
    assert "VERDICT" in out


def test_backtest_json_is_parseable(btc_csv, capsys):
    import json as _json

    assert cli.main(["backtest", "--json", "--strategy", "ema_cross",
                     "--csv", str(btc_csv)]) == cli.EXIT_OK
    payload = _json.loads(capsys.readouterr().out)
    assert payload["kind"] == "single"
    assert payload["strategy"] == "ema_cross"
    assert "sharpe" in payload["metrics"]


def test_backtest_two_strategies_two_symbols_is_a_portfolio(kraken, capsys):
    code = cli.main(["backtest", "--strategy", "ema_cross,macd_momentum",
                     "--symbol", "BTC/USD,ETH/USD", "--csv", str(kraken)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "portfolio: 2 strategies" in out
    # Portfolio-only numbers: the caps that a single-symbol run cannot exercise.
    assert "peak committed" in out


def test_backtest_refuses_fewer_bars_than_warmup(tmp_path, capsys):
    """donchian_breakout needs 205 bars; Kraken's 1h window serves 168. The
    message has to say what to do about it, not just fail."""
    small = write_kraken_dir(tmp_path / "small", bars=100)
    code = cli.main(["backtest", "--strategy", "donchian_breakout",
                     "--csv", str(small / "XBTUSD_240.csv")])
    err = capsys.readouterr().err
    assert code == cli.EXIT_FAIL
    assert "207 bars" in err and "100 are available" in err
    assert "import-history" in err and "7B" in err


def test_backtest_unknown_strategy_lists_the_real_ones(btc_csv, capsys):
    code = cli.main(["backtest", "--strategy", "moon_cross",
                     "--csv", str(btc_csv)])
    err = capsys.readouterr().err
    assert code == cli.EXIT_USAGE
    assert "ema_cross" in err and "donchian_breakout" in err


def test_backtest_param_override_reaches_the_strategy(btc_csv, capsys):
    """Proved by an observable consequence, not by exit 0.

    `slow` drives `warmup`, so a value larger than the file can support has to
    change the outcome. Asserting on returns instead would be vacuous: these
    are 4h/1d strategies and 300 bars routinely produce zero closed trades,
    which is the design working (ARCHITECTURE.md §0.1), not a broken fixture.
    """
    assert cli.main(["backtest", "--strategy", "ema_cross",
                     "--csv", str(btc_csv)]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["backtest", "--strategy", "ema_cross", "--param", "slow=400",
                     "--csv", str(btc_csv)]) == cli.EXIT_FAIL
    assert "407 bars" in capsys.readouterr().err


def test_backtest_rejects_an_unknown_param(btc_csv, capsys):
    code = cli.main(["backtest", "--strategy", "ema_cross",
                     "--param", "nonsense=1", "--csv", str(btc_csv)])
    assert code == cli.EXIT_FAIL
    assert "unknown parameters" in capsys.readouterr().err


def test_backtest_refuses_param_with_all(btc_csv, capsys):
    code = cli.main(["backtest", "--all", "--param", "fast=8",
                     "--csv", str(btc_csv)])
    assert code == cli.EXIT_USAGE
    assert "one strategy" in capsys.readouterr().err


def test_backtest_needs_a_strategy(btc_csv, capsys):
    assert cli.main(["backtest", "--csv", str(btc_csv)]) == cli.EXIT_USAGE
    assert "--strategy" in capsys.readouterr().err


def test_csv_timeframe_mismatch_is_caught(btc_csv, capsys):
    code = cli.main(["backtest", "--strategy", "ema_cross", "--timeframe", "1d",
                     "--csv", str(btc_csv)])
    assert code == cli.EXIT_USAGE
    assert "4h data" in capsys.readouterr().err


def test_bars_from_zip(tmp_path, kraken):
    archive = tmp_path / "kraken.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for csv in kraken.glob("*.csv"):
            zf.write(csv, csv.name)
    bars = cli.bars_from_path(archive, ["BTC/USD"], "4h")
    assert len(bars["BTC/USD"]) == 300
    assert bars["BTC/USD"] == sorted(bars["BTC/USD"], key=lambda b: b.ts)


# ── walk-forward ─────────────────────────────────────────────────────────


def test_walkforward_runs_and_reports_trials(btc_csv, capsys):
    code = cli.main(["walkforward", "--strategy", "macd_momentum",
                     "--folds", "3", "--csv", str(btc_csv)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "OUT-OF-SAMPLE" in out and "trials evaluated" in out


def test_walkforward_montecarlo_says_so_when_there_are_too_few_trades(
        btc_csv, capsys):
    """300 bars of 4h data yields no closed out-of-sample trades here, so this
    pins the *skip* branch specifically. If a future fixture produces five or
    more, this fails loudly rather than passing on whichever branch ran."""
    code = cli.main(["walkforward", "--strategy", "macd_momentum", "--folds", "3",
                     "--montecarlo", "--csv", str(btc_csv)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "Monte Carlo skipped: 0 out-of-sample trade(s)" in out
    assert "block-bootstrap paths" not in out


def test_walkforward_is_one_strategy_at_a_time(btc_csv, capsys):
    code = cli.main(["walkforward", "--strategy", "ema_cross,macd_momentum",
                     "--csv", str(btc_csv)])
    assert code == cli.EXIT_USAGE
    assert "deflated Sharpe counts trials" in capsys.readouterr().err


# ── the human gate ───────────────────────────────────────────────────────


ARM = ["arm-live", "--strategy", "ema_cross", "--symbol", "BTC/USD",
       "--max-notional", "250"]


def test_arm_live_refuses_without_a_terminal(no_tty, capsys):
    code = cli.main(ARM)
    err = capsys.readouterr().err
    assert code == cli.EXIT_REFUSED
    assert "interactive terminal" in err


def test_arm_live_refuses_before_touching_the_database(no_tty, fake_db):
    """The refusal must not depend on Postgres being reachable — otherwise a
    database outage would turn a safety gate into a different error message."""
    assert cli.main(ARM) == cli.EXIT_REFUSED
    assert fake_db.calls == []


def _armable_db() -> FakeDB:
    return FakeDB(
        rows={"FROM accounts WHERE id": [{"id": 1, "name": "main"}],
              "FROM promotion_status WHERE strategy": []},
        vals={"FROM risk_config": 7,
              "INSERT INTO live_arming": 42,
              "INSERT INTO notifications": 9},
    )


def _run_arm(db, monkeypatch, argv, typed):
    """Drive `_arm_live` with an injected reader. Never uses a real database."""
    import asyncio

    @contextlib.asynccontextmanager
    async def _database(**_kw):
        yield db

    monkeypatch.setattr(cli, "database", _database)
    args = cli.build_parser().parse_args(argv)
    return asyncio.run(cli._arm_live(args, ask=lambda _prompt: typed))


def test_arm_live_refuses_when_the_promotion_gate_never_ran(monkeypatch, capsys):
    db = _armable_db()
    with pytest.raises(cli.Refused) as exc:
        _run_arm(db, monkeypatch, ARM, "ARM LIVE main 250.00 USD")
    assert "never evaluated" in str(exc.value)
    assert not db.sql_containing("INSERT INTO live_arming")


def test_arm_live_refuses_a_mistyped_phrase(monkeypatch, capsys):
    db = _armable_db()
    with pytest.raises(cli.Refused) as exc:
        _run_arm(db, monkeypatch, [*ARM, "--force"], "arm live main 250 usd")
    assert "did not match" in str(exc.value)
    assert not db.sql_containing("INSERT INTO live_arming")


def test_the_phrase_carries_the_cap_so_it_cannot_be_muscle_memory():
    from decimal import Decimal

    a = cli.confirmation_phrase("main", Decimal("250"))
    b = cli.confirmation_phrase("main", Decimal("2500"))
    assert a != b and "250.00" in a


def test_arm_live_forced_arms_and_leaves_a_permanent_trail(monkeypatch, capsys):
    db = _armable_db()
    code = _run_arm(db, monkeypatch, [*ARM, "--force"],
                    "ARM LIVE main 250.00 USD")
    assert code == cli.EXIT_OK

    armed = db.sql_containing("INSERT INTO live_arming")
    assert len(armed) == 1
    # gate_forced is the last bound parameter of create_human's INSERT.
    assert armed[0][1][-1] is True

    audit = db.sql_containing("INSERT INTO audit_log")
    assert len(audit) == 1 and audit[0][1][1] == "arm_live_forced"
    notify = db.sql_containing("INSERT INTO notifications")
    assert len(notify) == 1 and notify[0][1][1] == "loud"

    out = capsys.readouterr().out
    assert "PROMOTION GATE   : FORCED PAST" in out
    assert "renews this indefinitely" in out


def test_arm_live_arms_cleanly_when_the_gate_passed(monkeypatch, capsys):
    db = _armable_db()
    db.rows["FROM promotion_status WHERE strategy"] = [
        {"strategy": "ema_cross", "passed": True, "failures": []}]
    assert _run_arm(db, monkeypatch, ARM, "ARM LIVE main 250.00 USD") == cli.EXIT_OK
    armed = db.sql_containing("INSERT INTO live_arming")
    assert len(armed) == 1 and armed[0][1][-1] is False
    assert db.sql_containing("INSERT INTO audit_log")[0][1][1] == "arm_live"


def test_arm_live_needs_an_explicit_scope(monkeypatch):
    db = _armable_db()
    with pytest.raises(cli.CliError) as exc:
        _run_arm(db, monkeypatch, ["arm-live", "--max-notional", "250"], "x")
    assert exc.value.code == cli.EXIT_USAGE
    assert not db.sql_containing("INSERT INTO live_arming")


def test_disarm_needs_no_gate(fake_db, capsys):
    code = cli.main(["disarm", "--reason", "testing"])
    assert code == cli.EXIT_OK
    assert fake_db.sql_containing("UPDATE live_arming SET disarmed_at")
    assert "poisons the whole lineage" in capsys.readouterr().out


def test_arming_status_when_nothing_is_armed(fake_db, capsys):
    assert cli.main(["arming-status"]) == cli.EXIT_OK
    assert "NOT armed" in capsys.readouterr().out


def test_arming_status_flags_a_changed_envelope(fake_db, capsys):
    fake_db.rows["FROM active_arming WHERE account_id"] = [{
        "id": 3, "origin": "auto_renew", "expires_at": "2026-08-01",
        "allowed_symbols": ["BTC/USD"], "allowed_strategies": ["ema_cross"],
        "max_notional_usd": "250", "max_daily_notional_usd": "250",
        "gate_forced": False, "envelope_hash": "stale-hash-from-another-envelope",
    }]
    assert cli.main(["arming-status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "CHANGED" in out and "human tap is required" in out


# ── code fingerprint ─────────────────────────────────────────────────────


def _tree(root: Path, body: str) -> Path:
    (root / "ct").mkdir(parents=True, exist_ok=True)
    (root / "ct" / "mod.py").write_text(body, encoding="utf-8")
    return root


def test_code_fingerprint_is_stable_and_change_sensitive(tmp_path):
    """It is folded into `envelope_hash`, so a code edit must break renewal."""
    a = _tree(tmp_path / "a", "x = 1\n")
    first = cli.code_fingerprint(a)
    assert first == cli.code_fingerprint(a)
    (a / "ct" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    assert cli.code_fingerprint(a) != first


def test_code_fingerprint_notices_a_new_file(tmp_path):
    a = _tree(tmp_path / "b", "x = 1\n")
    before = cli.code_fingerprint(a)
    (a / "ct" / "extra.py").write_text("y = 1\n", encoding="utf-8")
    assert cli.code_fingerprint(a) != before


# ── control plane ────────────────────────────────────────────────────────


def test_halt_queues_intent_not_an_action(fake_db, capsys):
    assert cli.main(["halt"]) == cli.EXIT_OK
    inserted = fake_db.sql_containing("INSERT INTO control_commands")
    assert len(inserted) == 1 and inserted[0][1][1] == "halt"
    assert "next cycle" in capsys.readouterr().out


def test_flatten_says_what_happens_and_how_to_verify(fake_db, capsys):
    """Flatten is real since 2026-07-29, and the message changed with it:
    queued is still not done, so it must say where completion shows up
    (FLATTEN_COMPLETE) rather than claiming positions are closed."""
    assert cli.main(["flatten"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "reduce-only market orders" in out
    assert "FLATTEN_COMPLETE" in out
    assert "does NOT liquidate" not in out, "the old warning must be gone"
    assert fake_db.sql_containing("INSERT INTO control_commands")[0][1][1] == \
        "flatten_all"


def test_flatten_one_symbol_carries_the_symbol(fake_db):
    assert cli.main(["flatten", "--symbol", "BTC/USD"]) == cli.EXIT_OK
    call = fake_db.sql_containing("INSERT INTO control_commands")[0]
    assert call[1][1] == "flatten_symbol" and "BTC/USD" in call[1][2]


def test_pause_and_resume_strategy(fake_db):
    assert cli.main(["pause-strategy", "ema_cross"]) == cli.EXIT_OK
    assert cli.main(["resume-strategy", "ema_cross"]) == cli.EXIT_OK
    commands = [c[1][1]
                for c in fake_db.sql_containing("INSERT INTO control_commands")]
    assert commands == ["pause_strategy", "resume_strategy"]


def test_kill_switch_is_just_a_file(tmp_path, capsys):
    path = tmp_path / "state" / "KILL"
    assert cli.main(["kill", "--file", str(path), "--reason", "smoke"]) == cli.EXIT_OK
    assert path.exists() and path.read_text(encoding="utf-8") == "smoke"
    assert "blocks ALL orders including exits" in capsys.readouterr().out
    assert cli.main(["kill", "--file", str(path), "--release"]) == cli.EXIT_OK
    assert not path.exists()


# ── status and promotion ─────────────────────────────────────────────────


def test_promotion_status_distinguishes_never_evaluated_from_failed(
        fake_db, capsys):
    assert cli.main(["promotion-status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "No strategy has ever been evaluated" in out
    assert "not measured" in out and "not 'failed'" in out


def test_promotion_status_prints_rows_when_they_exist(fake_db, capsys):
    fake_db.rows["FROM promotion_status ORDER BY"] = [{
        "strategy": "ema_cross", "passed": False, "paper_days": 12,
        "paper_trades": 4, "paper_sharpe": 0.31, "failures": ["paper_days<90"]}]
    assert cli.main(["promotion-status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "ema_cross" in out and "paper_days<90" in out


def test_status_on_an_empty_system_says_so(fake_db, capsys):
    assert cli.main(["status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "no equity snapshot yet" in out
    assert "not armed for live trading" in out


def test_status_without_a_dsn_explains_the_variable(monkeypatch, capsys):
    monkeypatch.delenv("QD_DB_DSN", raising=False)
    assert cli.main(["status"]) == cli.EXIT_FAIL
    assert "QD_DB_DSN" in capsys.readouterr().err


def test_coverage_reports_what_is_stored(fake_db, capsys):
    # Needles have to be specific: `find_gaps` and `coverage` both select
    # FROM candles, and a loose match feeds one query's rows to the other.
    fake_db.rows["count(*) FILTER (WHERE is_gap_fill) imported"] = [{
        "lo": None, "hi": None, "n": 0, "imported": 0}]
    fake_db.rows["lag(ts) OVER"] = []
    assert cli.main(["coverage", "--symbol", "BTC/USD",
                     "--timeframe", "4h"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "BTC/USD" in out and "is_gap_fill" in out


def test_init_db_reports_whether_it_applied_anything(fake_db, capsys, tmp_path):
    schema = tmp_path / "schema.sql"
    schema.write_text("CREATE TABLE orders (id int);", encoding="utf-8")
    assert cli.main(["init-db", "--schema", str(schema)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "already present" in out and "orders" in out
    assert fake_db.sql_containing("BOOTSTRAP")


def test_init_db_refuses_a_missing_schema(fake_db, capsys):
    assert cli.main(["init-db", "--schema", "/nope/schema.sql"]) == cli.EXIT_FAIL
    assert "schema not found" in capsys.readouterr().err


# ── history import ───────────────────────────────────────────────────────


def test_import_history_dry_run_touches_no_database(kraken, capsys):
    code = cli.main(["import-history", str(kraken), "--dry-run"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "600 bars from 2 files" in out
    assert "nothing written" in out


def test_import_history_can_filter_to_one_symbol(kraken, capsys):
    assert cli.main(["import-history", str(kraken), "--dry-run",
                     "--symbol", "BTC/USD"]) == cli.EXIT_OK
    assert "300 bars from 1 files" in capsys.readouterr().out


def test_import_history_writes_and_marks_bars_as_imported(kraken, fake_db,
                                                          capsys):
    assert cli.main(["import-history", str(kraken)]) == cli.EXIT_OK
    upserts = fake_db.sql_containing("INSERT INTO candles")
    assert upserts, "nothing was written"
    # is_gap_fill is the last column of the upsert tuple.
    assert all(row[-1] is True for row in upserts[0][1])
    assert "is_gap_fill" in capsys.readouterr().out


def test_hash_password_hands_over_a_paste_ready_env_line(monkeypatch, capsys):
    """Regression, measured against the real docker binary.

    An argon2id hash is full of `$` and a compose `.env` interpolates `$`, so
    pasting the raw hash yields `=19=65536,t=3,p=4` inside the container —
    every login then fails in a way that looks like a wrong password rather
    than a mangled file. Quoting does **not** help; only `$$` survives. This
    command used to advise quoting.
    """
    import types

    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "a-long-enough-pw")
    assert cli.main(["hash-password"]) == cli.EXIT_OK

    out = capsys.readouterr().out
    line = next(x for x in out.splitlines()
                if x.startswith("QD_ADMIN_PASSWORD_HASH="))
    value = line.split("=", 1)[1]
    assert "$$" in value, "the .env line is not escaped"
    assert "$" not in value.replace("$$", ""), "an unescaped $ survived"
    # Undoing the escaping must reproduce a valid hash the verifier accepts.
    from argon2 import PasswordHasher

    PasswordHasher().verify(value.replace("$$", "$"), "a-long-enough-pw")


# ── the strategy lifecycle surface (2026-07-29) ──────────────────────────
# Until these commands existed the FSM's human gate had no lever: the table
# was written, `proposed` blocked trading, and the only way to enable a
# strategy was raw SQL plus a container restart.


def _strategy_row(name="donchian_breakout", status="proposed", reason=""):
    return {"name": name, "status": status, "status_reason": reason,
            "enabled": status in ("paper", "live"), "paused_until": None,
            "symbols": ["BTC/USD", "ETH/USD"], "timeframe": "4h",
            "weight": 1.0, "updated_at": None}


def test_strategies_lists_states_and_points_at_the_gate(fake_db, capsys):
    fake_db.rows["FROM strategies ORDER BY name"] = [
        _strategy_row(), _strategy_row("ema_cross", "quarantined", "raised")]
    assert cli.main(["strategies"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "donchian_breakout" in out and "proposed" in out
    assert "quarantined" in out and "raised" in out
    assert "qd enable-strategy" in out, "the list must say how to enable"


def test_strategies_on_an_empty_table_says_so(fake_db, capsys):
    assert cli.main(["strategies"]) == cli.EXIT_OK
    assert "no strategies registered" in capsys.readouterr().out


def test_enable_strategy_refuses_without_a_tty(no_tty, fake_db):
    """Same automation-path closure as arm-live: no cron, pipe or compose
    `command:` may enable a strategy."""
    assert cli.main(["enable-strategy", "donchian_breakout"]) == cli.EXIT_REFUSED
    assert fake_db.sql_containing("UPDATE strategies") == []


def test_enable_strategy_moves_proposed_to_paper(a_tty, fake_db, capsys,
                                                 monkeypatch):
    fake_db.rows["FROM strategies WHERE name"] = [_strategy_row()]
    monkeypatch.setattr("builtins.input", lambda *_a: "y")
    assert cli.main(["enable-strategy", "donchian_breakout"]) == cli.EXIT_OK

    update = fake_db.sql_containing("UPDATE strategies")
    assert update and update[0][1][1] == "paper"
    audit = fake_db.sql_containing("INSERT INTO audit_log")
    assert audit and audit[0][1][0] == "human", (
        "the FSM demands a human actor for a risk-increasing transition")
    out = capsys.readouterr().out
    assert "no restart needed" in out and "90-day paper record" in out


def test_enable_strategy_not_confirmed_changes_nothing(a_tty, fake_db,
                                                       monkeypatch):
    fake_db.rows["FROM strategies WHERE name"] = [_strategy_row()]
    monkeypatch.setattr("builtins.input", lambda *_a: "n")
    assert cli.main(["enable-strategy", "donchian_breakout"]) == cli.EXIT_REFUSED
    assert fake_db.sql_containing("UPDATE strategies") == []


def test_enable_strategy_on_a_running_strategy_is_a_noop(a_tty, fake_db,
                                                         capsys):
    fake_db.rows["FROM strategies WHERE name"] = [
        _strategy_row(status="paper")]
    assert cli.main(["enable-strategy", "donchian_breakout"]) == cli.EXIT_OK
    assert "already paper" in capsys.readouterr().out
    assert fake_db.sql_containing("UPDATE strategies") == []


def test_enable_strategy_unknown_name_fails_loudly(a_tty, fake_db):
    assert cli.main(["enable-strategy", "moon_cross"]) == cli.EXIT_FAIL


def test_enable_strategy_clears_a_quarantine(a_tty, fake_db, monkeypatch):
    """QUARANTINED -> PAPER is exactly the transition the engine may never
    make itself; the CLI is the human path the FSM demands."""
    fake_db.rows["FROM strategies WHERE name"] = [
        _strategy_row(status="quarantined", reason="raised on BTC/USD")]
    monkeypatch.setattr("builtins.input", lambda *_a: "yes")
    assert cli.main(["enable-strategy", "donchian_breakout"]) == cli.EXIT_OK
    update = fake_db.sql_containing("UPDATE strategies")
    assert update and update[0][1][1] == "paper"


# ── backtest recording (2026-07-30): the reproducibility record ──────────


def test_backtest_records_a_row_when_a_database_is_reachable(
        fake_db, btc_csv, capsys):
    code = cli.main(["backtest", "--strategy", "ema_cross",
                     "--symbol", "BTC/USD", "--csv", str(btc_csv)])
    assert code == cli.EXIT_OK
    inserted = fake_db.sql_containing("INSERT INTO backtests")
    assert len(inserted) == 1
    args = inserted[0][1]
    assert args[0] == "ema_cross" and args[13] == "single"
    # The three reproducibility hashes are real digests, not placeholders.
    assert len(args[7]) == 64, "data_hash must be a sha256"
    assert args[8], "code fingerprint present"
    assert "(recorded in backtests: inserted)" in capsys.readouterr().err


def test_backtest_without_a_database_says_it_was_not_recorded(
        btc_csv, capsys, monkeypatch):
    monkeypatch.delenv("QD_DB_DSN", raising=False)
    code = cli.main(["backtest", "--strategy", "ema_cross",
                     "--symbol", "BTC/USD", "--csv", str(btc_csv)])
    assert code == cli.EXIT_OK, "recording is optional; the backtest is not"
    assert "NOT recorded: no database configured" in capsys.readouterr().err


def test_walkforward_records_with_kind_walkforward(fake_db, btc_csv, capsys):
    code = cli.main(["walkforward", "--strategy", "ema_cross",
                     "--symbol", "BTC/USD", "--csv", str(btc_csv),
                     "--folds", "2"])
    assert code == cli.EXIT_OK
    inserted = fake_db.sql_containing("INSERT INTO backtests")
    assert len(inserted) == 1
    assert inserted[0][1][13] == "walkforward"
