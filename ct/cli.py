"""`qd` — the operator's command line.

`pyproject.toml` has declared `qd = "ct.cli:main"` since the repo was created and
this module did not exist, so every `qd …` command in README.md was documented
and undeliverable (S4-1 in PHASE2-AUDIT.md). This is that module.

## Four rules this file follows

1. **`arm-live` is the one command Claude may not run.** The confirmation phrase
   can be supplied *only* by typing it interactively: there is no `--confirm`
   flag, no environment variable and no stdin path, and a non-TTY stdin is
   refused before any database work happens. That closes every automation path.
   It does **not** prove a human is present — see `require_interactive` for
   exactly what the check is worth and what carries the rest of the weight.

2. **A command that cannot do what its name says must say so**, loudly, and
   still exit non-zero where that is the honest answer. `flatten` prints that
   completion is observed (FLATTEN_COMPLETE), not assumed. `promotion-status`
   against an empty table reports "never evaluated" rather than "not passed" —
   those are different claims and only one of them is true.

3. **Nothing here writes orders, fills or positions.** `qd-engine` is the single
   writer (ARCHITECTURE.md §1). Control commands are *intent* rows the engine
   polls; the CLI's job is to queue them and say what was queued.

4. **Exit codes are load-bearing**, because these run under cron and compose:

       0  success
       1  failure (a thing was attempted and did not work)
       2  usage error (argparse)
       3  refused by a gate — the command was understood and declined

Data for research commands comes from Postgres (`QD_DB_DSN`) or, with `--csv`,
straight from a Kraken OHLCVT file, directory or zip. The second path exists so
a backtest is runnable on a laptop with no database at all.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ct.backtest.metrics import PerformanceReport
    from ct.models import Bar
    from ct.store import Database
    from ct.strategies.base import Strategy

log = logging.getLogger("ct.cli")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

DEFAULT_SYMBOLS = "BTC/USD"
DEFAULT_TIMEFRAME = "4h"


class CliError(Exception):
    """A failure with a message the operator can act on and an exit code."""

    def __init__(self, message: str, code: int = EXIT_FAIL) -> None:
        super().__init__(message)
        self.code = code


class Refused(CliError):
    """A gate said no. Distinct from a failure: nothing broke."""

    def __init__(self, message: str) -> None:
        super().__init__(message, EXIT_REFUSED)


# ── small shared helpers ─────────────────────────────────────────────────


def repo_root() -> Path:
    """The directory holding `ct/` and `db/`."""
    return Path(__file__).resolve().parent.parent


def code_fingerprint(root: Path | None = None) -> str:
    """Identity of the code, folded into `envelope_hash` (ARCHITECTURE.md §8.4).

    `git rev-parse HEAD` when this is a git checkout; otherwise a sha256 over
    every `.py` under `ct/`, sorted by path. The fallback is not a weaker
    substitute — it is stricter, because it changes on an *uncommitted* edit,
    which a commit sha does not. This tree is currently not a git checkout at
    all, so the fallback is the live path today.
    """
    root = root or repo_root()
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=5, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, ValueError):
        pass
    digest = hashlib.sha256()
    for path in sorted((root / "ct").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def parse_value(raw: str) -> Any:
    """`"10"` → 10, `"0.5"` → 0.5, `"true"` → True, anything else → str."""
    low = raw.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def parse_params(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise CliError(f"--param expects key=value, got {pair!r}", EXIT_USAGE)
        key, _, val = pair.partition("=")
        out[key.strip()] = parse_value(val)
    return out


def parse_space(pairs: Sequence[str] | None) -> dict[str, list[Any]]:
    """`--space fast=5,10,20` → `{"fast": [5, 10, 20]}`."""
    out: dict[str, list[Any]] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise CliError(f"--space expects key=v1,v2, got {pair!r}", EXIT_USAGE)
        key, _, vals = pair.partition("=")
        out[key.strip()] = [parse_value(v) for v in vals.split(",") if v.strip()]
    return out


def split_list(values: Sequence[str] | None, default: str = "") -> list[str]:
    """Accepts `--symbol A --symbol B` and `--symbol A,B` interchangeably."""
    raw = list(values) if values else ([default] if default else [])
    out: list[str] = []
    for item in raw:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    return out


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise CliError(
            f"cannot read {raw!r} as a date; use YYYY-MM-DD or an ISO timestamp",
            EXIT_USAGE) from exc
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def emit(payload: dict[str, Any] | list[Any], text: str, as_json: bool) -> None:
    print(json.dumps(payload, default=str, indent=2) if as_json else text)


def strategy_class(name: str) -> type[Strategy]:
    from ct.strategies import REGISTRY

    if name not in REGISTRY:
        raise CliError(
            f"unknown strategy {name!r}. Available: {', '.join(sorted(REGISTRY))}",
            EXIT_USAGE)
    return REGISTRY[name]


@asynccontextmanager
async def database(*, max_size: int = 2) -> AsyncIterator[Database]:
    from ct.store import Database

    dsn = os.environ.get("QD_DB_DSN")
    if not dsn:
        raise CliError(
            "QD_DB_DSN is not set. Inside the stack it comes from compose; "
            "outside it looks like "
            "postgresql://qd:PASSWORD@127.0.0.1:5432/quantdesk")
    db = Database(dsn, max_size=max_size)
    await db.connect(retries=2)
    try:
        yield db
    finally:
        await db.close()


# ── data loading ─────────────────────────────────────────────────────────


def bars_from_path(path: str | Path, symbols: Sequence[str], timeframe: str,
                   ) -> dict[str, list[Bar]]:
    """Read Kraken OHLCVT CSVs — a file, a directory or a zip — into bars.

    The no-database path. `--csv` on a research command means the whole run is
    reproducible from one file, which is what makes a backtest reviewable by
    somebody who does not have the NAS.
    """
    from ct.data.history import ImportStats, iter_archive, parse_filename, read_csv

    p = Path(path)
    wanted = set(symbols)
    out: dict[str, list[Bar]] = {s: [] for s in symbols}

    if p.is_file() and p.suffix.lower() == ".csv":
        symbol, tf = parse_filename(p.name)
        if symbol not in wanted:
            raise CliError(
                f"{p.name} holds {symbol} but --symbol asked for "
                f"{', '.join(sorted(wanted))}", EXIT_USAGE)
        if tf != timeframe:
            raise CliError(f"{p.name} is {tf} data but --timeframe is {timeframe}",
                           EXIT_USAGE)
        with open(p, encoding="utf-8", newline="") as fh:
            out[symbol] = list(read_csv(fh, symbol, tf, ImportStats()))
    else:
        for bar in iter_archive(p, symbols=wanted, timeframes={timeframe}):
            out[bar.symbol].append(bar)

    for symbol, bars in out.items():
        bars.sort(key=lambda b: b.ts)
        if not bars:
            raise CliError(f"{p} contains no {timeframe} bars for {symbol}")
    return out


async def bars_from_db(symbols: Sequence[str], timeframe: str, *,
                       since: datetime | None, until: datetime | None,
                       exchange: str) -> dict[str, list[Bar]]:
    from ct.data.repository import CandleRepository

    async with database() as db:
        repo = CandleRepository(db, exchange=exchange)
        out: dict[str, list[Bar]] = {}
        for symbol in symbols:
            bars = await repo.load(symbol, timeframe, since=since, until=until)
            if not bars:
                cov = await repo.coverage(symbol, timeframe)
                raise CliError(
                    f"no {timeframe} candles stored for {symbol} on {exchange} "
                    f"in that window (total stored: {cov['bars']}). "
                    f"`qd coverage` shows what exists; `qd backfill` fetches "
                    f"what the venue still serves.")
            out[symbol] = bars
        return out


def load_bars(args: argparse.Namespace, symbols: Sequence[str]) -> dict[str, list[Bar]]:
    if args.csv:
        return bars_from_path(args.csv, symbols, args.timeframe)
    return asyncio.run(bars_from_db(
        symbols, args.timeframe, since=parse_ts(args.since),
        until=parse_ts(args.until), exchange=args.exchange))


def warmup_note(name: str, warmup: int, have: int) -> str:
    """The A11 message, stated where it is actually hit.

    Three strategies need 205 bars and Kraken's 1h window serves 168, so this
    is not a rare edge — it is the ordinary outcome of pointing a breakout
    strategy at live-collected 1h data before the bulk history import lands.
    """
    return (
        f"{name} needs {warmup + 2} bars and only {have} are available.\n"
        f"  Kraken's API retains ~720 intervals (30 days at 1h, ~4 months at "
        f"4h, ~2 years at 1d) — see ARCHITECTURE.md §7B.\n"
        f"  Either use a longer timeframe, or import deep history with "
        f"`qd import-history` once the bulk archive is downloaded.")


# ── research commands ────────────────────────────────────────────────────


def report_lines(name: str, report: PerformanceReport,
                 extra: Sequence[str] = ()) -> str:
    r = report
    lines = [
        f"{name}",
        f"  bars / trades    : {r.n_periods:,} / {r.n_trades}",
        f"  return           : {r.total_return:+.1%}   CAGR {r.cagr:+.1%}",
        f"  Sharpe / DSR     : {r.sharpe:.2f} / {r.deflated_sharpe:.2f}",
        f"  max drawdown     : {r.max_drawdown:.1%} over "
        f"{r.max_drawdown_duration} bars",
        f"  win rate         : {r.win_rate:.0%}   payoff {r.payoff_ratio:.2f}",
        f"  fees             : {r.total_fees:,.0f} "
        f"({r.fees_pct_of_gross:.0%} of gross profit)",
        f"  benchmark        : {r.benchmark_return:+.1%} "
        f"(beats it: {r.beats_benchmark})",
        *extra,
        f"  VERDICT          : {r.verdict()}",
    ]
    return "\n".join(lines)


def _record_run(args: argparse.Namespace, *, strategy: str,
                symbols: list[str], bars_by_symbol: dict[str, list[Bar]],
                params: dict[str, Any], metrics: dict[str, Any],
                cost_model: Any, equity_curve: list[float] | None,
                n_trials: int = 1, kind: str = "single") -> str:
    """Record a research run in `backtests` — the reproducibility record.

    Never fails the run it records: research must work on a laptop with no
    database, so an unreachable one degrades to a printed note. The note is
    printed rather than logged because the person who just ran the backtest
    is the person who needs to know it was not recorded."""
    import dataclasses

    from ct.store.backtests import BacktestRepository, data_hash, downsample
    from ct.store.strategies import params_hash

    # `is_dataclass` narrows to "instance OR class"; asdict only accepts an
    # instance. Excluding the type keeps the narrowing honest rather than
    # casting it away.
    costs = (dataclasses.asdict(cost_model)
             if dataclasses.is_dataclass(cost_model)
             and not isinstance(cost_model, type)
             else dict(vars(cost_model)))

    from ct.config import strategy_config

    cfg = strategy_config()

    async def _write() -> str:
        async with database() as db:
            series = next(iter(bars_by_symbol.values()))
            return await BacktestRepository(db).record(
                strategy=strategy, symbols=symbols, timeframe=args.timeframe,
                start_ts=series[0].ts, end_ts=series[-1].ts, params=params,
                params_hash=params_hash(params),
                data_hash=data_hash(bars_by_symbol),
                code_git_sha=code_fingerprint(),
                cost_model=costs,
                metrics=metrics,
                equity_curve=downsample(equity_curve) if equity_curve else None,
                n_trials=n_trials, kind=kind,
                # Provenance: which revision of config/strategies.yml this ran
                # against. None when there is no config file — research on a
                # laptop is a legitimate case and must not fail the record.
                config_hash=cfg.config_hash() if cfg.version else None,
                config_version=cfg.version or None)

    # Notes go to STDERR: with --json, stdout is a machine-readable payload
    # and a chatty status line would corrupt it (a test caught exactly that).
    try:
        outcome = asyncio.run(_write())
    except CliError:
        print("  (run NOT recorded: no database configured — set QD_DB_DSN "
              "to keep the reproducibility record)", file=sys.stderr)
        return "unrecorded"
    except Exception as exc:
        print(f"  (run NOT recorded: {type(exc).__name__} — the backtest "
              f"itself is unaffected)", file=sys.stderr)
        return "unrecorded"
    if outcome == "diverged":
        print("  WARNING: same strategy+params+data+code previously produced "
              "DIFFERENT results — nondeterminism somewhere; see the engine "
              "log. Every backtest number is suspect until explained.",
              file=sys.stderr)
    else:
        print(f"  (recorded in backtests: {outcome})", file=sys.stderr)
    return outcome


def cmd_backtest(args: argparse.Namespace) -> int:
    from ct.backtest.engine import BacktestEngine
    from ct.backtest.portfolio import Allocation, PortfolioBacktestEngine
    from ct.strategies import REGISTRY

    names = sorted(REGISTRY) if args.all else split_list(args.strategy)
    if not names:
        raise CliError("give --strategy NAME (repeatable) or --all", EXIT_USAGE)
    symbols = split_list(args.symbol, DEFAULT_SYMBOLS)
    params = parse_params(args.param)
    if params and len(names) > 1:
        raise CliError("--param applies to one strategy; drop --all or pass one "
                       "--strategy", EXIT_USAGE)

    bars = load_bars(args, symbols)
    strategies = [strategy_class(n)(**params) for n in names]

    # One strategy on one symbol is the single-symbol engine; anything wider is
    # a portfolio, because summing isolated single-symbol runs overstates
    # deployable P&L by roughly the number of books (ARCHITECTURE.md §7A).
    if len(strategies) == 1 and len(symbols) == 1:
        strat = strategies[0]
        series = bars[symbols[0]]
        if len(series) < strat.warmup + 2:
            raise CliError(warmup_note(strat.name, strat.warmup, len(series)))
        engine = BacktestEngine(
            strat, initial_equity=args.equity, timeframe=args.timeframe)
        result = engine.run(series, symbol=symbols[0])
        payload = {"kind": "single", "strategy": strat.name,
                   "symbols": symbols, "timeframe": args.timeframe,
                   "rejections": result.rejections,
                   "metrics": result.report.to_dict()}
        text = report_lines(
            f"{strat.name} on {symbols[0]} {args.timeframe}", result.report,
            [f"  rejections       : {result.rejection_summary()}"])
        emit(payload, text, args.json)
        payload["recorded"] = _record_run(
            args, strategy=strat.name, symbols=symbols,
            bars_by_symbol={symbols[0]: series}, params=strat.params,
            metrics=result.report.to_dict(), cost_model=engine.costs,
            equity_curve=result.equity_curve, kind="single")
        return EXIT_OK

    thin = [s.name for s in strategies
            if min(len(bars[sym]) for sym in symbols) < s.warmup + 2]
    if thin:
        have = min(len(bars[sym]) for sym in symbols)
        raise CliError("\n".join(
            warmup_note(name, strategy_class(name)().warmup, have)
            for name in thin))

    allocations = [Allocation(s, tuple(symbols), weight=1.0 / len(strategies))
                   for s in strategies]
    # Distinct name from the single-symbol `result` above: the two branches are
    # disjoint (that one returns), but they hold different result types, and
    # reusing the name made mypy read every PortfolioResult-only attribute
    # below as missing from BacktestResult — six findings from one binding.
    pf = PortfolioBacktestEngine(
        allocations, initial_equity=args.equity, timeframe=args.timeframe,
    ).run(bars)
    payload = {"kind": "portfolio", "strategies": names, "symbols": symbols,
               "timeframe": args.timeframe, "rejections": pf.rejections,
               "peak_committed_exposure": pf.peak_committed_exposure,
               "peak_open_books": pf.peak_open_books,
               "metrics": pf.report.to_dict(),
               "per_strategy": {k: v.to_dict()
                                for k, v in pf.per_strategy.items()}}
    text = report_lines(
        f"portfolio: {len(strategies)} strategies x {len(symbols)} symbols "
        f"{args.timeframe}", pf.report,
        [f"  peak committed   : {pf.peak_committed_exposure:.1%} "
         f"(books {pf.peak_open_books})",
         f"  rejections       : {pf.rejection_summary()}"],
    ) + "\n\n" + pf.attribution()
    emit(payload, text, args.json)
    return EXIT_OK


def cmd_walkforward(args: argparse.Namespace) -> int:
    from ct.backtest.engine import CostModel
    from ct.backtest.walkforward import monte_carlo, walk_forward

    names = split_list(args.strategy)
    if args.all:
        from ct.strategies import REGISTRY

        names = sorted(REGISTRY)
    if len(names) != 1:
        raise CliError("walkforward takes exactly one --strategy at a time "
                       "(the deflated Sharpe counts trials per strategy)",
                       EXIT_USAGE)
    symbols = split_list(args.symbol, DEFAULT_SYMBOLS)
    if len(symbols) != 1:
        raise CliError("walkforward is single-symbol; use `qd backtest` for a "
                       "portfolio run", EXIT_USAGE)

    cls = strategy_class(names[0])
    bars = load_bars(args, symbols)[symbols[0]]
    space = parse_space(args.space)
    result = walk_forward(cls, bars, space, timeframe=args.timeframe,
                          initial_equity=args.equity, n_splits=args.folds,
                          embargo_pct=args.embargo)

    text = result.summary()
    payload: dict[str, Any] = {
        "strategy": cls.name, "symbol": symbols[0], "timeframe": args.timeframe,
        "folds": args.folds, "n_trials": result.n_trials,
        "chosen_params": result.chosen_params,
        "metrics": result.oos_report.to_dict(),
    }
    if args.montecarlo:
        rets = [t.return_pct for t in result.oos_trades]
        if len(rets) < 5:
            # Say the real count. A bootstrap over a handful of trades produces
            # a confidence interval wider than the estimate, and printing one
            # anyway is how a number nobody should act on ends up in a report.
            text += (f"\n\nMonte Carlo skipped: {len(rets)} out-of-sample "
                     f"trade(s), and the bootstrap needs at least 5.")
        else:
            mc = monte_carlo(rets, initial_equity=args.equity)
            text += "\n\n" + mc.summary()
            payload["montecarlo"] = {
                "p05_final_equity": mc.p05_final_equity,
                "p50_final_equity": mc.p50_final_equity,
                "p95_final_equity": mc.p95_final_equity,
                "p95_max_drawdown": mc.p95_max_drawdown,
                "prob_of_loss": mc.prob_of_loss,
                "prob_of_ruin": mc.prob_of_ruin,
            }
    emit(payload, text, args.json)
    # Stored metrics carry the Monte Carlo inside them: the promotion gate's
    # criterion 4 reads metrics->'montecarlo'->>'p95_max_drawdown'.
    stored_metrics = dict(payload["metrics"])
    stored_metrics["chosen_params"] = result.chosen_params
    if "montecarlo" in payload:
        stored_metrics["montecarlo"] = payload["montecarlo"]
    payload["recorded"] = _record_run(
        args, strategy=cls.name, symbols=symbols,
        bars_by_symbol={symbols[0]: bars},
        params=result.chosen_params[-1] if result.chosen_params else {},
        metrics=stored_metrics, cost_model=CostModel(),
        equity_curve=result.oos_equity, n_trials=result.n_trials,
        kind="walkforward")
    return EXIT_OK


async def _promotion_status(args: argparse.Namespace) -> int:
    async with database() as db:
        if args.strategy:
            rows = await db.fetch(
                "SELECT * FROM promotion_status WHERE strategy = ANY($1::text[])"
                " ORDER BY strategy", split_list(args.strategy))
        else:
            rows = await db.fetch(
                "SELECT * FROM promotion_status ORDER BY strategy")

    if not rows:
        # "No row" and "did not pass" are different claims. The nightly
        # evaluator that writes this table is Phase 5 and is not built, so
        # printing `passed: false` here would be inventing a verdict.
        emit({"evaluated": False, "rows": []},
             "No strategy has ever been evaluated against the promotion gate.\n"
             "  `promotion_status` is empty. The nightly evaluator that fills "
             "it is Phase 5 and is not built yet, so this is 'not measured', "
             "not 'failed'.\n"
             "  Live arming therefore requires --force, which is recorded in "
             "`audit_log` and notified.", args.json)
        return EXIT_OK

    lines = [f"{'strategy':24s} {'passed':>7} {'days':>5} {'trades':>7} "
             f"{'sharpe':>7}  failures"]
    for r in rows:
        lines.append(
            f"{r['strategy']:24s} {r['passed']!s:>7} {r['paper_days']:5d} "
            f"{r['paper_trades']:7d} "
            f"{float(r['paper_sharpe'] or 0):7.2f}  "
            f"{', '.join(r['failures']) or '-'}")
    emit([dict(r) for r in rows], "\n".join(lines), args.json)
    return EXIT_OK


def cmd_promotion_status(args: argparse.Namespace) -> int:
    return asyncio.run(_promotion_status(args))


# ── data commands ────────────────────────────────────────────────────────


async def _init_db(args: argparse.Namespace) -> int:
    schema = Path(args.schema) if args.schema else repo_root() / "db" / "schema.sql"
    if not schema.is_file():
        raise CliError(f"schema not found at {schema}")
    async with database() as db:
        applied = await db.bootstrap(schema)
        tables = sorted(await db.table_names())
    emit({"applied": applied, "tables": tables},
         (f"schema applied from {schema}" if applied else
          "schema already present; nothing applied (drift warnings go to the log)")
         + f"\n{len(tables)} tables: {', '.join(tables)}", args.json)
    return EXIT_OK


async def _coverage(args: argparse.Namespace) -> int:
    from ct.data.repository import CandleRepository

    symbols = split_list(args.symbol, DEFAULT_SYMBOLS)
    timeframes = split_list(args.timeframe_list, DEFAULT_TIMEFRAME)
    out: list[dict[str, Any]] = []
    async with database() as db:
        repo = CandleRepository(db, exchange=args.exchange)
        for symbol in symbols:
            for tf in timeframes:
                cov = dict(await repo.coverage(symbol, tf))
                gaps = await repo.find_gaps(symbol, tf)
                cov["gaps"] = len(gaps)
                cov["missing_bars"] = sum(g[2] for g in gaps)
                out.append(cov)

    lines = [f"{'symbol':12s} {'tf':>4} {'bars':>9} {'days':>7} {'complete':>9} "
             f"{'imported':>9} {'gaps':>5}"]
    for c in out:
        lines.append(
            f"{c['symbol']:12s} {c['timeframe']:>4} {c['bars']:9,} "
            f"{c['span_days']:7.1f} {c['completeness']:9.1%} "
            f"{c['imported']:9,} {c['gaps']:5d}")
    lines.append("  'imported' counts bars marked is_gap_fill — fetched or "
                 "imported rather than streamed live.")
    emit(out, "\n".join(lines), args.json)
    return EXIT_OK


async def _backfill(args: argparse.Namespace) -> int:
    import ccxt.pro as ccxtpro

    from ct.data.backfill import Backfiller, ccxt_fetcher
    from ct.data.repository import CandleRepository

    symbols = split_list(args.symbol, DEFAULT_SYMBOLS)
    timeframes = split_list(args.timeframe_list, DEFAULT_TIMEFRAME)
    since = parse_ts(args.since)
    until = parse_ts(args.until)

    exchange = getattr(ccxtpro, args.exchange)({"enableRateLimit": True})
    results: list[dict[str, Any]] = []
    lines: list[str] = []
    try:
        backfiller = Backfiller(ccxt_fetcher(exchange))
        async with database() as db:
            repo = CandleRepository(db, exchange=args.exchange)
            for symbol in symbols:
                for tf in timeframes:
                    if since is None:
                        last = await repo.latest_ts(symbol, tf)
                        res = await backfiller.catch_up(
                            symbol, tf, last_known=last,
                            now=datetime.now(timezone.utc))
                    else:
                        res = await backfiller.fetch_range(
                            symbol, tf, since=since, until=until)
                    stored = await repo.upsert_many(res.bars, is_gap_fill=True)
                    lines.append(f"  {res.summary()} -> {stored} stored")
                    results.append({"symbol": symbol, "timeframe": tf,
                                    "fetched": len(res.bars), "stored": stored,
                                    "requests": res.requests,
                                    "truncated": res.truncated,
                                    "reason": res.reason})
                    if since is not None and res.bars and res.bars[0].ts > since:
                        # A11, stated where somebody actually hits it rather
                        # than only in a design document.
                        lines.append(
                            f"  NOTE: asked for {since.date()}, oldest served "
                            f"{res.bars[0].ts.date()}. Kraken retains ~720 "
                            f"intervals per timeframe and serves nothing older "
                            f"— deep history needs `qd import-history` "
                            f"(ARCHITECTURE.md §7B).")
    finally:
        await exchange.close()

    emit(results, "backfill:\n" + "\n".join(lines), args.json)
    return EXIT_OK


async def _import_history(args: argparse.Namespace) -> int:
    from ct.data.history import ImportStats, iter_archive
    from ct.data.repository import CandleRepository

    symbols = set(split_list(args.symbol)) or None
    timeframes = set(split_list(args.timeframe_list)) or None
    stats = ImportStats()
    bars = iter_archive(args.path, symbols=symbols, timeframes=timeframes,
                        stats=stats)

    if args.dry_run:
        count = sum(1 for _ in bars)
        emit({"dry_run": True, "bars": count, "summary": stats.summary()},
             f"dry run — nothing written\n  {stats.summary()}", args.json)
        return EXIT_OK

    async with database() as db:
        repo = CandleRepository(db, exchange=args.exchange)
        stored = await repo.upsert_stream(bars, batch=args.batch,
                                          is_gap_fill=True)
    emit({"dry_run": False, "stored": stored, "summary": stats.summary()},
         f"imported {stored:,} candles\n  {stats.summary()}\n"
         f"  Bars are marked is_gap_fill: Kraken's archive omits intervals with "
         f"no trades, so a hole in it is a quiet market, not missing data.",
         args.json)
    return EXIT_OK


def cmd_init_db(args: argparse.Namespace) -> int:
    return asyncio.run(_init_db(args))


def cmd_migrate(args: argparse.Namespace) -> int:
    """Apply pending Alembic migrations.

    Synchronous — Alembic drives its own SQLAlchemy engine, and wrapping that in
    asyncio.run() to match the neighbouring commands would buy nothing but a
    second event loop.

    Safe to run against the existing deployment: the 0001 baseline detects a
    schema that initdb already built and adopts it rather than recreating it.
    A migration tool nobody dares run on the real database is not one.
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parent.parent
    ini = root / "alembic.ini"
    if not ini.is_file():
        raise CliError(f"alembic.ini not found at {ini}", EXIT_FAIL)
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(root / "migrations"))

    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()

    if args.status:
        from sqlalchemy import create_engine

        from ct.config import sqlalchemy_dsn

        # NOT `from migrations.env import _dsn`. Alembic loads env.py with a
        # live `context` established; importing it directly raises
        # AttributeError on `context.config`, which is why --status failed on
        # every invocation until a real database was pointed at it.
        try:
            engine = create_engine(sqlalchemy_dsn())
        except ValueError as exc:
            raise CliError(str(exc), EXIT_FAIL) from None
        try:
            with engine.connect() as conn:
                current = MigrationContext.configure(conn).get_current_revision()
        finally:
            engine.dispose()
        print(f"current : {current or '(none — never migrated)'}")
        print(f"head    : {head}")
        pending = [s.revision for s in script.walk_revisions()
                   if current is None or s.revision != current]
        if current == head:
            print("status  : up to date")
        else:
            print(f"status  : {len(pending)} revision(s) pending")
        return EXIT_OK

    if args.sql:
        # Review what it will do to a live trading database before it does it.
        command.upgrade(cfg, args.revision, sql=True)
        return EXIT_OK

    command.upgrade(cfg, args.revision)
    print(f"migrated to {args.revision}")
    return EXIT_OK


def cmd_coverage(args: argparse.Namespace) -> int:
    return asyncio.run(_coverage(args))


def cmd_backfill(args: argparse.Namespace) -> int:
    return asyncio.run(_backfill(args))


def cmd_import_history(args: argparse.Namespace) -> int:
    return asyncio.run(_import_history(args))


# ── operations ───────────────────────────────────────────────────────────


async def _status(args: argparse.Namespace) -> int:
    from ct.store import PositionRepository

    async with database() as db:
        equity = await db.fetchrow(
            """SELECT ts, equity, cash, gross_exposure, drawdown
               FROM equity_curve WHERE account_id=$1
               ORDER BY ts DESC LIMIT 1""", args.account)
        books = await PositionRepository(db).load(args.account)
        arming = await db.fetchrow(
            "SELECT * FROM active_arming WHERE account_id=$1", args.account)
        pending_cmds = await db.fetch(
            """SELECT id, command, requested_by, ts FROM control_commands
               WHERE applied_at IS NULL AND (account_id=$1 OR account_id IS NULL)
               ORDER BY id""", args.account)
        risk = await db.fetch(
            """SELECT ts, severity, code, detail FROM risk_events
               WHERE resolved_at IS NULL ORDER BY ts DESC LIMIT 10""")
        outbox = await db.fetchval(
            "SELECT count(*) FROM notifications WHERE status='pending'")

    open_books = {k: v for k, v in books.items() if not v.is_flat}
    payload = {
        "account": args.account,
        "mode": os.environ.get("QD_MODE", "paper"),
        "equity": dict(equity) if equity else None,
        "open_books": {k: {"qty": str(v.qty), "avg_entry": str(v.avg_entry)}
                       for k, v in open_books.items()},
        "armed": arming is not None,
        "arming": dict(arming) if arming else None,
        "pending_commands": [dict(r) for r in pending_cmds],
        "unresolved_risk_events": [dict(r) for r in risk],
        "notifications_pending": int(outbox or 0),
    }
    lines = [f"account {args.account}   QD_MODE={payload['mode']}"]
    if equity:
        lines.append(
            f"  equity {float(equity['equity']):,.2f} "
            f"(cash {float(equity['cash']):,.2f}, gross "
            f"{float(equity['gross_exposure']):.1%}, dd "
            f"{float(equity['drawdown']):.1%}) as of {equity['ts']}")
    else:
        lines.append("  no equity snapshot yet — the engine has not marked to "
                     "market on this account")
    lines.append(f"  open books: {len(open_books)}")
    for key, book in sorted(open_books.items()):
        lines.append(f"    {key:32s} qty {book.qty} @ {book.avg_entry}")
    if arming:
        lines.append(f"  LIVE ARMED until {arming['expires_at']} "
                     f"(origin {arming['origin']}, id {arming['id']})")
    else:
        lines.append("  not armed for live trading")
    lines.append(f"  pending control commands: {len(pending_cmds)}")
    lines.append(f"  unresolved risk events: {len(risk)}")
    for r in risk[:5]:
        lines.append(f"    [{r['severity']}] {r['code']}: {r['detail']}")
    lines.append(f"  notifications pending: {int(outbox or 0)}")
    if int(outbox or 0):
        lines.append("    NOTE: the engine's sender is a stub that logs and "
                     "reports success — nothing reaches Telegram yet.")
    emit(payload, "\n".join(lines), args.json)
    return EXIT_OK


async def _control(args: argparse.Namespace, command: str,
                   cmd_args: dict[str, Any] | None = None) -> int:
    from ct.engine.control import ControlPlane

    async with database() as db:
        cid = await ControlPlane(db, args.account).request(
            command, requested_by=args.by, args=cmd_args or {})
    note = ""
    if command in {"flatten_all", "flatten_symbol"}:
        note = ("\n  The engine submits reduce-only market orders on its next "
                "cycle (~20 s) and clears the request only once the book is "
                "actually flat — watch `qd status` / risk_events for "
                "FLATTEN_COMPLETE. Blocked only by the kill switch.")
    emit({"command": command, "id": cid, "args": cmd_args or {}},
         f"queued {command} (id {cid}) for account {args.account}; the engine "
         f"applies it on its next cycle{note}", args.json)
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    return asyncio.run(_status(args))


def cmd_halt(args: argparse.Namespace) -> int:
    return asyncio.run(_control(args, "halt"))


def cmd_resume(args: argparse.Namespace) -> int:
    return asyncio.run(_control(args, "resume"))


def cmd_flatten(args: argparse.Namespace) -> int:
    if args.symbol_one:
        return asyncio.run(_control(args, "flatten_symbol",
                                    {"symbol": args.symbol_one}))
    return asyncio.run(_control(args, "flatten_all"))


def cmd_pause_strategy(args: argparse.Namespace) -> int:
    return asyncio.run(_control(args, "pause_strategy",
                                {"strategy": args.strategy_one}))


def cmd_resume_strategy(args: argparse.Namespace) -> int:
    return asyncio.run(_control(args, "resume_strategy",
                                {"strategy": args.strategy_one}))


async def _strategies(args: argparse.Namespace) -> int:
    async with database() as db:
        rows = await db.fetch(
            """SELECT name, status, status_reason, enabled, paused_until,
                      symbols, timeframe, weight, updated_at
               FROM strategies ORDER BY name""")
    if not rows:
        emit([], "no strategies registered — the engine registers its "
             "allocations on first boot with a database", args.json)
        return EXIT_OK
    data = [{
        "name": r["name"], "status": r["status"],
        "reason": r["status_reason"] or "",
        "enabled": bool(r["enabled"]),
        "paused_until": r["paused_until"].isoformat() if r["paused_until"] else None,
        "symbols": list(r["symbols"] or []), "timeframe": r["timeframe"],
    } for r in rows]
    lines = []
    for d in data:
        until = f" until {d['paused_until'][:16]}Z" if d["paused_until"] else ""
        reason = f" — {d['reason']}" if d["reason"] else ""
        lines.append(f"  {d['name']:<24} {d['status']}{until}"
                     f"  [{','.join(d['symbols'])} {d['timeframe']}]{reason}")
    hint = ""
    if any(d["status"] == "proposed" for d in data):
        hint = ("\nA `proposed` strategy never trades. Enable one for paper "
                "with `qd enable-strategy NAME` — the running engine picks it "
                "up within one cycle.")
    emit(data, "strategies:\n" + "\n".join(lines) + hint, args.json)
    return EXIT_OK


async def _enable_strategy(args: argparse.Namespace) -> int:
    """proposed/quarantined/retired -> paper. THE PAPER GATE.

    The FSM refuses this transition from the engine (`Actor.ENGINE`); this
    command is the human path it demands. Enabling to `live` is deliberately
    not offered — promotion is the Phase 5 evaluator's job, and live trading
    additionally requires `QD_MODE=live` plus `qd arm-live`.
    """
    from ct.store import StrategyRepository
    from ct.strategies.lifecycle import Actor, IllegalTransition, StrategyState

    require_interactive("enable-strategy")
    async with database() as db:
        repo = StrategyRepository(db)
        lc = await repo.load(args.strategy_one)
        if lc is None:
            raise CliError(
                f"unknown strategy {args.strategy_one!r} — the engine registers "
                f"its allocations at boot; see `qd strategies`")
        if lc.may_run:
            emit({"strategy": lc.name, "state": lc.state.value},
                 f"{lc.name} is already {lc.state.value}; nothing to do",
                 args.json)
            return EXIT_OK
        print(f"{lc.name}: {lc.state.value} -> paper")
        if lc.reason:
            print(f"  last reason: {lc.reason}")
        answer = input(f"enable {lc.name} for PAPER trading? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise Refused("not confirmed; nothing changed")
        try:
            lc.to(StrategyState.PAPER, actor=Actor.HUMAN,
                  reason=f"enabled by {args.by}")
        except IllegalTransition as exc:
            raise Refused(str(exc)) from exc
        await repo.save(lc, actor=Actor.HUMAN)
    emit({"strategy": lc.name, "state": lc.state.value},
         f"{lc.name} is now {lc.state.value}. The running engine picks this up "
         f"within one cycle (~20 s) — no restart needed — and trades on the "
         f"next closed bar. This starts the 90-day paper record the promotion "
         f"gate counts.", args.json)
    return EXIT_OK


def cmd_strategies(args: argparse.Namespace) -> int:
    return asyncio.run(_strategies(args))


def cmd_enable_strategy(args: argparse.Namespace) -> int:
    return asyncio.run(_enable_strategy(args))


def cmd_kill(args: argparse.Namespace) -> int:
    """The file kill switch — no database, no network, no engine logic.

    Deliberately the one control path that works when everything else is down.
    """
    from ct.engine.control import KillSwitch

    switch = KillSwitch(args.file)
    if args.release:
        switch.release()
        emit({"engaged": False, "path": str(switch.path)},
             f"kill switch RELEASED ({switch.path}). The engine resumes "
             f"sending orders on its next cycle.", args.json)
        return EXIT_OK
    switch.engage(args.reason)
    emit({"engaged": True, "path": str(switch.path), "reason": args.reason},
         f"kill switch ENGAGED ({switch.path}): {args.reason}\n"
         f"  This blocks ALL orders including exits — it means 'I do not trust "
         f"this system'. To close positions instead, that is `qd flatten`.",
         args.json)
    return EXIT_OK


# ── the human gate ───────────────────────────────────────────────────────


def confirmation_phrase(account_name: str, max_notional: Decimal) -> str:
    """What must be typed, exactly. Carries the numbers so it cannot be muscle
    memory: a re-arm at a different cap is a different sentence."""
    return f"ARM LIVE {account_name} {max_notional:.2f} USD"


def require_interactive(what: str) -> None:
    """Refuse unless the phrase can be typed at a terminal.

    **What this actually guarantees, stated exactly.** The arming phrase has no
    flag, no environment variable and no file: the only way to supply it is to
    type it interactively. That closes every *automation* path — a cron job, a
    pipe, a compose `command:`, a scheduler, a script — which is the class of
    accident that would otherwise arm live trading without anybody deciding to.

    **What it does not guarantee.** Measured 2026-07-28: an agent shell can have
    a TTY, so `isatty()` does not by itself distinguish a person from an AI
    agent driving a terminal. Do not read this check as making that impossible.
    The controls that make it *safe* are the ones around it: every arming writes
    a permanent `audit_log` row and a loud notification, `qd disarm` and the kill
    switch always work, arming alone does nothing without `QD_MODE=live`, and
    Claude's operating rule (CLAUDE.md stop list §2) forbids originating one.
    A cryptographic second factor — signing the envelope hash with a key whose
    private half never touches the NAS — is the only gate that would make it
    structurally impossible, and that is the operator's call to adopt.

    Checked before any database work, so the refusal never depends on whether
    Postgres happened to be reachable.
    """
    if not sys.stdin.isatty():
        raise Refused(
            f"{what} requires an interactive terminal.\n"
            "  The confirmation phrase can only be typed at a TTY: there is no "
            "flag, environment variable or stdin path that supplies it.\n"
            "  That is deliberate — it is what makes it impossible for a "
            "script, a scheduler or an AI agent to put capital at risk.\n"
            "  Run it yourself:  docker compose run --rm qd-engine qd arm-live …")


def read_typed_confirmation(prompt: str) -> str:
    """Read the arming phrase. Re-checks the TTY: defence in depth, since this
    is the single function through which the phrase can enter the process."""
    require_interactive("arm-live")
    return input(prompt)


async def _arm_live(args: argparse.Namespace,
                    ask: Callable[[str], str] = read_typed_confirmation) -> int:
    from ct.store import ArmingRepository, NotificationOutbox, envelope_hash

    symbols = split_list(args.symbol)
    strategies = split_list(args.strategy)
    if not symbols or not strategies:
        raise CliError("arm-live needs --strategy and --symbol; arming is "
                       "scoped to exactly what you approve", EXIT_USAGE)
    max_notional = Decimal(str(args.max_notional))
    max_daily = Decimal(str(args.max_daily_notional
                            if args.max_daily_notional is not None
                            else args.max_notional))
    sha = code_fingerprint()

    async with database() as db:
        account = await db.fetchrow("SELECT * FROM accounts WHERE id=$1",
                                    args.account)
        name = account["name"] if account else f"account-{args.account}"
        risk_config_id = int(await db.fetchval(
            """SELECT id FROM risk_config WHERE account_id=$1
               ORDER BY effective_from DESC LIMIT 1""", args.account) or 0)

        gate = await db.fetch(
            "SELECT * FROM promotion_status WHERE strategy = ANY($1::text[])",
            strategies)
        by_name = {r["strategy"]: r for r in gate}
        blocked = []
        for s in strategies:
            row = by_name.get(s)
            if row is None:
                blocked.append(f"{s}: never evaluated (promotion_status has no "
                               f"row; the nightly evaluator is not built)")
            elif not row["passed"]:
                blocked.append(f"{s}: gate failed — "
                               f"{', '.join(row['failures']) or 'no reason recorded'}")
        if blocked and not args.force:
            raise Refused(
                "promotion gate has not passed:\n  " + "\n  ".join(blocked) +
                "\n  The gate exists to make going live a decision rather than "
                "a drift (ARCHITECTURE.md §8.3). Override with --force, which "
                "writes a permanent audit_log row and notifies.")

        envelope = envelope_hash(
            account_id=args.account, symbols=symbols, strategies=strategies,
            max_notional_usd=max_notional, max_daily_notional_usd=max_daily,
            risk_config_id=risk_config_id, code_git_sha=sha)

        print("You are about to authorise LIVE TRADING with real money.")
        print(f"  account          : {name} (id {args.account})")
        print(f"  mode env         : QD_MODE={os.environ.get('QD_MODE', 'paper')}"
              + ("" if os.environ.get("QD_MODE") == "live" else
                 "   <- NOT live; arming alone does not start live trading"))
        print(f"  strategies       : {', '.join(strategies)}")
        print(f"  symbols          : {', '.join(symbols)}")
        print(f"  max notional     : {max_notional} USD per position")
        print(f"  max daily        : {max_daily} USD")
        print(f"  risk config      : id {risk_config_id}")
        print(f"  code fingerprint : {sha}")
        print(f"  envelope hash    : {envelope}")
        print(f"  expires in       : {args.hours} h "
              f"(the engine renews it indefinitely while the envelope holds)")
        if blocked:
            print("  PROMOTION GATE   : FORCED PAST —")
            for b in blocked:
                print(f"    {b}")
        print()
        print("Nothing is armed until you type this line exactly:")
        phrase = confirmation_phrase(name, max_notional)
        print(f"  {phrase}")
        typed = ask("> ").strip()
        if typed != phrase:
            raise Refused("phrase did not match; nothing was armed")

        arming_id = await ArmingRepository(db).create_human(
            args.account, armed_by=args.by, confirmation_phrase=typed,
            envelope=envelope, symbols=symbols, strategies=strategies,
            max_notional_usd=max_notional, max_daily_notional_usd=max_daily,
            hours=args.hours, gate_forced=bool(blocked))

        await db.execute(
            """INSERT INTO audit_log (actor, action, target, after)
               VALUES ($1,$2,$3,$4::jsonb)""",
            args.by, "arm_live_forced" if blocked else "arm_live",
            f"account:{args.account}",
            json.dumps({"arming_id": arming_id, "envelope_hash": envelope,
                        "strategies": strategies, "symbols": symbols,
                        "max_notional_usd": str(max_notional),
                        "code_git_sha": sha, "gate_forced": bool(blocked),
                        "gate_failures": blocked}))
        await NotificationOutbox(db).enqueue(
            "telegram",
            f"LIVE ARMED (id {arming_id}) — {', '.join(strategies)} on "
            f"{', '.join(symbols)}, max {max_notional} USD"
            + (f". PROMOTION GATE FORCED: {'; '.join(blocked)}" if blocked else ""),
            subject="QuantDesk armed", priority="loud")

    emit({"arming_id": arming_id, "envelope_hash": envelope,
          "gate_forced": bool(blocked)},
         f"ARMED — live_arming id {arming_id}.\n"
         f"  The engine renews this indefinitely while the envelope hash holds. "
         f"Change a cap, a symbol, a strategy, the risk config or a line of "
         f"code and renewal stops and asks you again.\n"
         f"  `qd disarm` at any time; the kill switch always works.", args.json)
    return EXIT_OK


async def _disarm(args: argparse.Namespace) -> int:
    from ct.store import ArmingRepository

    async with database() as db:
        n = await ArmingRepository(db).disarm(
            args.account, reason=args.reason, disarmed_by=args.by)
        await db.execute(
            """INSERT INTO audit_log (actor, action, target, after)
               VALUES ($1,'disarm',$2,$3::jsonb)""",
            args.by, f"account:{args.account}",
            json.dumps({"rows": n, "reason": args.reason}))
    emit({"disarmed_rows": n, "reason": args.reason},
         f"disarmed {n} arming row(s) for account {args.account}: {args.reason}\n"
         f"  Disarm poisons the whole lineage — every renewal descended from a "
         f"disarmed row is invalid too.", args.json)
    return EXIT_OK


async def _arming_status(args: argparse.Namespace) -> int:
    from ct.store import ArmingRepository, envelope_hash

    async with database() as db:
        repo = ArmingRepository(db)
        active = await repo.active(args.account)
        chain = await repo.chain(int(active["id"])) if active else []
        risk_config_id = int(await db.fetchval(
            """SELECT id FROM risk_config WHERE account_id=$1
               ORDER BY effective_from DESC LIMIT 1""", args.account) or 0)

    if active is None:
        emit({"armed": False},
             f"account {args.account} is NOT armed for live trading.\n"
             f"  In paper mode that is irrelevant — arming gates live only.",
             args.json)
        return EXIT_OK

    # Recompute the envelope from what is true right now: if it no longer
    # matches the stored hash, renewal will stop and a human tap is coming.
    current = envelope_hash(
        account_id=args.account, symbols=list(active["allowed_symbols"]),
        strategies=list(active["allowed_strategies"]),
        max_notional_usd=Decimal(active["max_notional_usd"]),
        max_daily_notional_usd=Decimal(active["max_daily_notional_usd"]),
        risk_config_id=risk_config_id, code_git_sha=code_fingerprint())
    matches = current == active["envelope_hash"]
    lines = [
        f"account {args.account} is ARMED (id {active['id']}, "
        f"origin {active['origin']})",
        f"  expires        : {active['expires_at']}",
        f"  strategies     : {', '.join(active['allowed_strategies'])}",
        f"  symbols        : {', '.join(active['allowed_symbols'])}",
        f"  max notional   : {active['max_notional_usd']} USD",
        f"  gate forced    : {active['gate_forced']}",
        f"  renewal chain  : {len(chain)} row(s) back to the human tap",
        "  envelope       : "
        + ("unchanged — renewal will continue" if matches else
           "CHANGED — renewal will be refused and a human tap is required"),
    ]
    emit({"armed": True, "arming": dict(active), "chain_length": len(chain),
          "envelope_matches": matches, "envelope_now": current},
         "\n".join(lines), args.json)
    return EXIT_OK


def cmd_arm_live(args: argparse.Namespace) -> int:
    # Before the database, before argument validation, before anything that
    # could fail for an unrelated reason and mask this refusal.
    require_interactive("arm-live")
    return asyncio.run(_arm_live(args))


def cmd_disarm(args: argparse.Namespace) -> int:
    return asyncio.run(_disarm(args))


def cmd_arming_status(args: argparse.Namespace) -> int:
    return asyncio.run(_arming_status(args))


# ── utility ──────────────────────────────────────────────────────────────


def cmd_hash_password(args: argparse.Namespace) -> int:
    """argon2id hash for `QD_ADMIN_PASSWORD_HASH`.

    Typed twice, never echoed, never taken from a flag — a password on a command
    line lands in shell history and in `ps`.
    """
    import getpass

    try:
        from argon2 import PasswordHasher
    except ImportError as exc:
        raise CliError(
            "argon2-cffi is not installed in this environment. It is a declared "
            "dependency, so this means an incomplete install: "
            "pip install -e '.[dev]'") from exc

    require_interactive("hash-password")
    first = getpass.getpass("password: ")
    if len(first) < 12:
        raise Refused("use at least 12 characters; this guards the control "
                      "plane of a trading system")
    if first != getpass.getpass("again: "):
        raise Refused("passwords did not match")
    digest = PasswordHasher().hash(first)
    # An argon2id hash is full of `$`, and a compose `.env` interpolates `$`.
    # Measured against the real binary 2026-07-29:
    #     raw     $argon2id$v=19$m=...  ->  container sees "=19=65536,t=3,p=4"
    #     quoted  "$argon2id$v=19$..."  ->  identical destruction
    #     $$      $$argon2id$$v=19$...  ->  container sees the correct hash
    # This command previously said "quote it", which is exactly the form that
    # does NOT work — the hash arrives mangled, every login fails, and it looks
    # like a wrong password rather than a wrong file. So hand over the line
    # that can be pasted verbatim instead of describing an escaping rule.
    env_line = f"QD_ADMIN_PASSWORD_HASH={digest.replace('$', '$$')}"
    emit({"hash": digest, "env_line": env_line},
         f"{digest}\n\n"
         f"Paste this line into .env exactly as shown — every `$` is doubled,\n"
         f"because docker compose interpolates `$` in .env and quoting does "
         f"not stop it:\n\n"
         f"{env_line}", args.json)
    return EXIT_OK


# ── argument parsing ─────────────────────────────────────────────────────


def _common() -> argparse.ArgumentParser:
    """Flags every subcommand accepts, as a FRESH parser each call.

    Two argparse details, both found by running it rather than reading it:

    * `default=SUPPRESS` so a flag given *before* the subcommand survives. A
      subparser parses into its own namespace and copies every key onto the
      parent's, so an ordinary default would silently overwrite `--json`.
    * A new instance per subparser. `parents=[...]` shares the *same action
      objects*, and `set_defaults()` mutates `action.default` in place — so one
      shared parent parser lets the main parser's defaults reach back into every
      subparser and re-break the case above.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="machine-readable output")
    p.add_argument("--account", type=int, default=argparse.SUPPRESS,
                   help="account id (default $QD_ACCOUNT_ID or 1)")
    p.add_argument("--exchange", default=argparse.SUPPRESS,
                   help="exchange name (default $QD_EXCHANGE or kraken)")
    return p


def _research_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", action="append",
                   help="strategy name; repeatable or comma-separated")
    p.add_argument("--symbol", action="append", help="e.g. BTC/USD; repeatable")
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    p.add_argument("--since", help="ISO date; start of the window")
    p.add_argument("--until", help="ISO date; end of the window")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--csv", help="Kraken OHLCVT file, directory or zip — "
                                 "runs with no database at all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qd", parents=[_common()],
        description="QuantDesk operator CLI. Exit codes: 0 ok, 1 failure, "
                    "2 usage, 3 refused by a gate.")
    parser.set_defaults(
        json=False,
        account=int(os.environ.get("QD_ACCOUNT_ID", "1")),
        exchange=os.environ.get("QD_EXCHANGE", "kraken"),
        by=os.environ.get("QD_OPERATOR") or os.environ.get("USER") or "cli",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # research
    p = sub.add_parser("backtest", parents=[_common()],
                       help="backtest one strategy, or a portfolio")
    _research_args(p)
    p.add_argument("--all", action="store_true", help="every registered strategy")
    p.add_argument("--param", action="append", help="key=value strategy override")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("walkforward", parents=[_common()],
                       help="purged walk-forward — the honest evaluation")
    _research_args(p)
    p.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--space", action="append",
                   help="key=v1,v2,v3 parameter grid; repeatable")
    p.add_argument("--montecarlo", action="store_true",
                   help="block-bootstrap the out-of-sample trades")
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("promotion-status", parents=[_common()],
                       help="the paper->live gate")
    p.add_argument("--strategy", action="append")
    p.set_defaults(func=cmd_promotion_status)

    # data
    p = sub.add_parser("init-db", parents=[_common()],
                       help="apply db/schema.sql to an empty database")
    p.add_argument("--schema", help="path to schema.sql")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("migrate", parents=[_common()],
                       help="apply pending Alembic migrations")
    p.add_argument("--status", action="store_true",
                   help="show current revision vs head; apply nothing")
    p.add_argument("--sql", action="store_true",
                   help="print the DDL instead of running it")
    p.add_argument("--revision", default="head",
                   help="target revision (default: head)")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("coverage", parents=[_common()],
                       help="what candle data actually exists")
    p.add_argument("--symbol", action="append")
    p.add_argument("--timeframe", dest="timeframe_list", action="append")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("backfill", parents=[_common()],
                       help="REST catch-up inside the venue's retained window")
    p.add_argument("--symbol", action="append")
    p.add_argument("--timeframe", dest="timeframe_list", action="append")
    p.add_argument("--since", help="ISO date. NOTE: Kraken serves ~720 "
                                   "intervals and nothing older (§7B)")
    p.add_argument("--until")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("import-history", parents=[_common()],
                       help="import Kraken bulk OHLCVT CSVs (zip or directory)")
    p.add_argument("path")
    p.add_argument("--symbol", action="append", help="filter; default all mapped")
    p.add_argument("--timeframe", dest="timeframe_list", action="append")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true",
                   help="count what would be imported; touches no database")
    p.set_defaults(func=cmd_import_history)

    # operations
    p = sub.add_parser("status", parents=[_common()],
                       help="what the system believes right now")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("halt", parents=[_common()],
                       help="stop opening; exits still work")
    p.set_defaults(func=cmd_halt)

    p = sub.add_parser("resume", parents=[_common()],
                       help="resume opening (refused while unreconciled)")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("flatten", parents=[_common()],
                       help="request liquidation (see the warning it prints)")
    p.add_argument("--symbol", dest="symbol_one",
                   help="one symbol; omit to request all")
    p.set_defaults(func=cmd_flatten)

    p = sub.add_parser("pause-strategy", parents=[_common()], help="pause one strategy")
    p.add_argument("strategy_one", metavar="STRATEGY")
    p.set_defaults(func=cmd_pause_strategy)

    p = sub.add_parser("resume-strategy", parents=[_common()],
                       help="resume a paused strategy")
    p.add_argument("strategy_one", metavar="STRATEGY")
    p.set_defaults(func=cmd_resume_strategy)

    p = sub.add_parser("strategies", parents=[_common()],
                       help="lifecycle state of every registered strategy")
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser(
        "enable-strategy", parents=[_common()],
        help="THE PAPER GATE — move a strategy to `paper` so it trades "
             "(interactive TTY only; `live` is the promotion gate's job)")
    p.add_argument("strategy_one", metavar="STRATEGY")
    p.set_defaults(func=cmd_enable_strategy)

    p = sub.add_parser("kill", parents=[_common()],
                       help="the file kill switch — blocks ALL orders, exits too")
    p.add_argument("--file", default=os.environ.get("QD_KILL_SWITCH_FILE",
                                                    "/app/state/KILL"))
    p.add_argument("--reason", default="manual")
    p.add_argument("--release", action="store_true", help="remove it instead")
    p.set_defaults(func=cmd_kill)

    # live
    p = sub.add_parser(
        "arm-live", parents=[_common()],
        help="THE HUMAN GATE — authorise live trading (interactive TTY only)")
    p.add_argument("--strategy", action="append")
    p.add_argument("--symbol", action="append")
    p.add_argument("--max-notional", type=float, required=True,
                   dest="max_notional")
    p.add_argument("--max-daily-notional", type=float, default=None,
                   dest="max_daily_notional")
    p.add_argument("--hours", type=int, default=168,
                   help="initial validity; the engine renews it indefinitely")
    p.add_argument("--force", action="store_true",
                   help="bypass the promotion gate; permanently audited")
    p.set_defaults(func=cmd_arm_live)

    p = sub.add_parser("disarm", parents=[_common()],
                       help="revoke live arming; always permitted")
    p.add_argument("--reason", default="manual")
    p.set_defaults(func=cmd_disarm)

    p = sub.add_parser("arming-status", parents=[_common()],
                       help="what is armed, and whether renewal will continue")
    p.set_defaults(func=cmd_arming_status)

    # utility
    p = sub.add_parser("hash-password", parents=[_common()],
                       help="argon2id hash for QD_ADMIN_PASSWORD_HASH")
    p.set_defaults(func=cmd_hash_password)

    return parser


def _survive_a_narrow_console() -> None:
    """Never let an em dash kill a command.

    Half the messages in this project — including every `verdict()` and every
    backfill summary — carry `—` and `→`. On a cp1252 Windows console that is a
    `UnicodeEncodeError` *inside the print*, so the command fails after doing
    its work and reports an encoding error instead of its result. Found by
    running `qd import-history --dry-run`, which imported nothing and exited 1
    for a reason that had nothing to do with importing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # An already-detached stream (pytest, a closed pipe) cannot be
            # reconfigured, and failing here would break every command.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _survive_a_narrow_console()
    # WARNING is asked for explicitly rather than by reading QD_LOG_LEVEL with a
    # different default from the engine's. A second default for one variable is
    # invisible drift; test_no_env_default_drifted_from_source now fails if one
    # reappears. An operator who sets QD_LOG_LEVEL still wins.
    from ct.config import settings
    from ct.logging import configure as _configure_logging

    _configure_logging(level=os.environ.get("QD_LOG_LEVEL") or "WARNING",
                       fmt=settings().log_format)
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return exc.code
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_FAIL
    except Exception as exc:
        # One line for the operator, the traceback for the log. A CLI that
        # dumps a stack trace at 3 a.m. buries the sentence that matters.
        log.exception("%s failed", args.command)
        print(f"error: {args.command}: {exc}", file=sys.stderr)
        return EXIT_FAIL


__all__ = [
    "CliError",
    "Refused",
    "build_parser",
    "code_fingerprint",
    "confirmation_phrase",
    "main",
    "read_typed_confirmation",
    "require_interactive",
]


if __name__ == "__main__":
    raise SystemExit(main())
