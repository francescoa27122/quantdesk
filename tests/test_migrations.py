"""Alembic wiring.

These tests run without a database. That is a real limit and worth stating: they
verify the revision graph, the DSN translation, the refusals, and that the
offline renderer produces the DDL it claims to — but *applying* a migration to
Postgres is only exercised by `qd migrate` against a real server, and by CI.

Two defects showed up while writing this, both in the `--sql` path, which is the
"look at what this will do to the live trading database before it does it" path
and therefore the one least forgivable to leave broken:

  * the baseline probed for an existing schema through a connection that offline
    mode does not have, and crashed on `NoneType.execute`;
  * rendering db/schema.sql to a cp1252 console raised UnicodeEncodeError mid
    write, producing a truncated file rather than an error you could trust.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MIGRATIONS = REPO / "migrations"
VERSIONS = MIGRATIONS / "versions"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_mig_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _revisions() -> dict[str, object]:
    return {m.revision: m for m in (_load(p) for p in sorted(VERSIONS.glob("*.py")))}


# ── layout ───────────────────────────────────────────────────────────────


def test_scaffolding_exists():
    assert (REPO / "alembic.ini").is_file()
    assert (MIGRATIONS / "env.py").is_file()
    assert (MIGRATIONS / "script.py.mako").is_file()
    assert list(VERSIONS.glob("*.py")), "no revisions"


def test_migrations_are_not_inside_the_initdb_mount():
    """db/ is bind-mounted as /docker-entrypoint-initdb.d, which executes every
    .sql it finds on first boot. Migrations living there would be handed to the
    Postgres entrypoint instead of to Alembic."""
    assert not (REPO / "db" / "versions").exists()
    assert not list((REPO / "db").glob("*alembic*"))
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./db:/docker-entrypoint-initdb.d" in compose
    assert "./migrations:/docker-entrypoint-initdb.d" not in compose


def test_loading_the_alembic_config_does_not_disable_existing_loggers():
    """`logging.config.fileConfig()` defaults to disable_existing_loggers=True.

    `qd migrate` runs in the same process as the rest of the CLI, so that
    default would switch off every logger created before the migration —
    leaving the engine and the risk module silent for the remainder of the run.

    Found the hard way: after this file's tests loaded alembic.ini, an unrelated
    test in test_web.py asserting that a malformed password hash gets logged saw
    no records at all, and only in the full-suite run.
    """
    src = (MIGRATIONS / "env.py").read_text(encoding="utf-8")
    assert "disable_existing_loggers=False" in src, \
        "env.py calls fileConfig() without disable_existing_loggers=False"

    # And behaviourally: a logger created first must survive the config load.
    import logging
    from logging.config import fileConfig

    canary = logging.getLogger("ct.canary.migrations")
    canary.disabled = False
    fileConfig(str(REPO / "alembic.ini"), disable_existing_loggers=False)
    assert not canary.disabled, "loading alembic.ini disabled an existing logger"


def test_alembic_ini_has_no_hardcoded_dsn():
    """A trading database password must not sit in a committed file."""
    ini = (REPO / "alembic.ini").read_text(encoding="utf-8")
    for line in ini.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert not stripped.startswith("sqlalchemy.url"), \
            "alembic.ini sets sqlalchemy.url; the DSN belongs in QD_DB_DSN"


# ── revision graph ───────────────────────────────────────────────────────


def test_revision_chain_is_linear_and_rooted():
    revs = _revisions()
    roots = [r for r, m in revs.items() if m.down_revision is None]
    assert len(roots) == 1, f"expected exactly one root, got {roots}"
    # every non-root parent must exist
    for r, m in revs.items():
        if m.down_revision is not None:
            assert m.down_revision in revs, f"{r} points at missing {m.down_revision}"
    # exactly one head
    parents = {m.down_revision for m in revs.values() if m.down_revision}
    heads = set(revs) - parents
    assert len(heads) == 1, f"expected one head, got {sorted(heads)}"


def test_every_revision_declares_upgrade_and_downgrade():
    for rev, mod in _revisions().items():
        assert callable(getattr(mod, "upgrade", None)), f"{rev} has no upgrade()"
        assert callable(getattr(mod, "downgrade", None)), f"{rev} has no downgrade()"


def test_every_revision_documents_why_and_downgrade_safety():
    """A migration with no rationale is unreviewable, and one that does not say
    whether downgrade loses data is worse than unreviewable."""
    for path in sorted(VERSIONS.glob("*.py")):
        doc = (_load(path).__doc__ or "").lower()
        assert "why this change" in doc, f"{path.name} has no rationale"
        assert "downgrade safety" in doc, f"{path.name} does not state downgrade safety"


# ── the baseline ─────────────────────────────────────────────────────────


def test_baseline_refuses_to_downgrade():
    """Downgrading the baseline drops `fills`, the only record `positions` can
    be rebuilt from. There is no situation where that is a safe automatic act."""
    base = next(m for m in _revisions().values() if m.down_revision is None)
    with pytest.raises(RuntimeError, match="refusing to downgrade"):
        base.downgrade()


def test_baseline_probes_for_an_existing_schema():
    """It must adopt an initdb-built database, not try to recreate it — that is
    what makes `qd migrate` safe to run on the live deployment."""
    src = next(p for p in VERSIONS.glob("*.py")
               if _load(p).down_revision is None).read_text(encoding="utf-8")
    assert "to_regclass" in src, "baseline does not probe for an existing schema"
    assert "is_offline_mode" in src, "baseline would crash in --sql mode"


def test_baseline_reads_the_real_schema_file():
    base = next(m for m in _revisions().values() if m.down_revision is None)
    assert base._SCHEMA_SQL.is_file(), f"{base._SCHEMA_SQL} missing"
    text = base._read_schema_sql()
    assert "CREATE TABLE accounts" in text
    assert base._PROBE_TABLE == "accounts"


# ── DSN translation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("given", [
    "postgresql://qd:pw@qd-db:5432/quantdesk",
    "postgres://qd:pw@qd-db:5432/quantdesk",
    "postgresql+asyncpg://qd:pw@qd-db:5432/quantdesk",
])
def test_dsn_is_translated_for_sqlalchemy(monkeypatch, given):
    """The codebase talks asyncpg with a bare postgresql:// DSN. SQLAlchemy
    would otherwise default to psycopg2, which is not installed (this project
    ships psycopg 3)."""
    monkeypatch.setenv("QD_DB_DSN", given)
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("_env_probe", MIGRATIONS / "env.py")
    # env.py runs migrations on import, so read _dsn out of the source instead
    # of executing it. Keeps this test honest about what it is checking.
    src = (MIGRATIONS / "env.py").read_text(encoding="utf-8")
    ns: dict = {}
    body = re.search(r"def _dsn\(\) -> str:.*?(?=\ndef )", src, re.S).group(0)
    exec(compile(body, "env._dsn", "exec"), ns)
    out = ns["_dsn"]()
    assert out.startswith("postgresql+psycopg://")
    assert out.endswith("@qd-db:5432/quantdesk")
    assert spec is not None


def test_missing_dsn_is_a_clear_refusal(monkeypatch):
    monkeypatch.setenv("QD_DB_DSN", "")
    src = (MIGRATIONS / "env.py").read_text(encoding="utf-8")
    ns: dict = {}
    body = re.search(r"def _dsn\(\) -> str:.*?(?=\ndef )", src, re.S).group(0)
    exec(compile(body, "env._dsn", "exec"), ns)
    with pytest.raises(SystemExit, match="QD_DB_DSN"):
        ns["_dsn"]()


# ── the CLI ──────────────────────────────────────────────────────────────


def test_qd_migrate_is_registered():
    from ct.cli import build_parser

    args = build_parser().parse_args(["migrate", "--status"])
    assert args.command == "migrate"
    assert args.status is True
    assert args.revision == "head"


def test_qd_migrate_sql_renders_the_whole_chain(tmp_path, monkeypatch, capsys):
    """End-to-end offline render. Exercises both defects found in review: the
    baseline's offline branch, and console encoding."""
    from alembic import command
    from alembic.config import Config

    monkeypatch.setenv("QD_DB_DSN", "postgresql://qd:pw@localhost:5432/quantdesk")
    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS))
    # Offline mode writes through the migration context's output buffer, which
    # defaults to sys.stdout — not to Config.stdout, which only covers alembic's
    # own informational commands.
    command.upgrade(cfg, "base:head", sql=True)
    sql = capsys.readouterr().out

    assert "CREATE TABLE alembic_version" in sql
    assert "CREATE TABLE accounts" in sql, "baseline did not render the schema"
    assert "CREATE TABLE fills" in sql
    assert "ADD COLUMN IF NOT EXISTS config_hash" in sql, "0002 did not render"
    assert "FRESH-INSTALL BRANCH" in sql, "the conditional baseline is unlabelled"
    assert sql.rstrip().endswith("COMMIT;")


# ── the config_hash column this chain adds ───────────────────────────────


def test_backtest_repository_accepts_config_provenance():
    """The column exists only if the writer actually populates it."""
    import inspect

    from ct.store.backtests import BacktestRepository

    params = inspect.signature(BacktestRepository.record).parameters
    assert "config_hash" in params
    assert "config_version" in params
    assert params["config_hash"].default is None, \
        "must be optional — research on a laptop has no config file"


def test_apscheduler_jobs_migration_matches_the_librarys_own_schema():
    """The 0003 DDL is hand-written to match APScheduler's SQLAlchemyJobStore.

    The store compares against its own table definition, so a library upgrade
    that changes the schema would leave the migration describing a table the
    library no longer expects — and the symptom would be a scheduler that fails
    to persist, i.e. silently loses the daily promotion job across restarts.
    """
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

    store = SQLAlchemyJobStore(url="sqlite://", tablename="apscheduler_jobs")
    cols = {c.name: (c.nullable, str(c.type)) for c in store.jobs_t.columns}
    assert set(cols) == {"id", "next_run_time", "job_state"}, cols
    assert cols["id"][0] is False and "191" in cols["id"][1]
    assert cols["next_run_time"][0] is True
    assert cols["job_state"][0] is False
    assert [c.name for c in store.jobs_t.primary_key] == ["id"]
    assert [i.name for i in store.jobs_t.indexes] == \
        ["ix_apscheduler_jobs_next_run_time"]

    mig = next(p for p in VERSIONS.glob("*apscheduler*.py")).read_text(encoding="utf-8")
    assert "VARCHAR(191)" in mig
    # FLOAT(25) is double precision in Postgres — anything wider than 24 bits of
    # mantissa is. Equivalent, not a mismatch.
    assert "DOUBLE PRECISION" in mig
    assert "BYTEA" in mig
    assert "ix_apscheduler_jobs_next_run_time" in mig


def test_config_hash_is_not_part_of_the_uniqueness_key():
    """Adding it to the UNIQUE constraint would change a run's identity, so the
    same code+data+params would stop deduplicating and the divergence check —
    the thing that shouts when identical inputs produce different numbers —
    would go quiet."""
    src = (REPO / "ct" / "store" / "backtests.py").read_text(encoding="utf-8")
    where = re.search(r"WHERE strategy = \$1.*?kind = \$5", src, re.S)
    assert where is not None
    assert "config_hash" not in where.group(0)
