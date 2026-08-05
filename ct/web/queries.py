"""Every read the dashboard performs, in one file.

Two rules, both from ARCHITECTURE.md §10, and both load-bearing:

1. **`qd-web` computes nothing that `qd-engine` also computes.** Equity, P&L,
   drawdown and exposure are read, never recalculated. Two implementations of
   P&L that disagree is a support nightmare, and the one on the dashboard is
   the one you would believe.

2. **Every number on a page comes from a query in this module.** If a value
   cannot be traced to a `SELECT` here, it does not go on a page. That is what
   makes "where does this figure come from?" answerable in one grep.

Everything returns plain dicts and lists so templates never hold a database
handle and a page cannot accidentally issue a query while rendering.

**Engine liveness comes from the event log, not from a socket.** The two
containers share no channel by design (§1), so the dashboard learns what the
engine believes by reading the `heartbeat` events it publishes. The age of the
newest heartbeat *is* the liveness signal, and it degrades honestly: a dead
engine produces an old timestamp rather than a stale-but-plausible status.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# Anything older than this and the engine is not running, whatever the last
# heartbeat said. The engine's default cycle is 20 s and it beats every 15
# cycles — so the NORMAL gap between beats is ~300 s, and a 300 s threshold
# flaps to "quiet" on ordinary jitter right before each beat (equal-to-interval
# is a boundary error, found the day a human read the verdict). Stale means at
# least one whole beat was MISSED: 2 × interval + a minute of grace.
HEARTBEAT_STALE_S = 660


def _f(value: Any, default: float = 0.0) -> float:
    """Numerics arrive as Decimal; templates want float. Never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return {}
    return {}


class Queries:
    """Read-only view of the trading database."""

    def __init__(self, db, account_id: int = 1) -> None:
        self.db = db
        self.account_id = account_id

    # ── engine state ─────────────────────────────────────────────────────

    async def engine(self) -> dict[str, Any]:
        """What the engine last said about itself, and how long ago.

        `stale` is computed from the timestamp rather than trusted from the
        payload: a process that has stopped cannot tell you it has stopped.
        """
        # The type string comes from the enum the engine publishes with, not a
        # literal: this line said 'system.heartbeat' for two days while the
        # engine wrote 'ops.heartbeat', so the dashboard NEVER saw a heartbeat
        # and reported a healthy engine as "gone quiet". The test double
        # matched this query by a different substring, which is why no test
        # caught it — found by a human reading the Home page.
        from ct.events.types import EventType
        hb = EventType.HEARTBEAT.value
        row = await self.db.fetchrow(
            f"""SELECT ts, payload FROM events
               WHERE type = '{hb}' AND payload->>'component' = 'engine'
               ORDER BY seq DESC LIMIT 1""")
        collector = await self.db.fetchrow(
            f"""SELECT ts, payload FROM events
               WHERE type = '{hb}'
                 AND payload->>'component' = 'collector'
               ORDER BY seq DESC LIMIT 1""")
        now = datetime.now(timezone.utc)

        def _age(r) -> float | None:
            return None if r is None else (now - r["ts"]).total_seconds()

        payload = _payload(row["payload"]) if row else {}
        age = _age(row)
        return {
            "seen": row is not None,
            "age_s": age,
            "stale": age is None or age > HEARTBEAT_STALE_S,
            "mode": payload.get("mode", "unknown"),
            "halted": bool(payload.get("halted", True)),
            "halt_reason": payload.get("halt_reason", ""),
            "kill": bool(payload.get("kill", False)),
            "armed": bool(payload.get("armed", False)),
            "reconciled": bool(payload.get("reconciled", False)),
            "may_open": bool(payload.get("may_open", False)),
            "may_close": bool(payload.get("may_close", False)),
            "paused_strategies": list(payload.get("paused_strategies") or []),
            "cycles": int(payload.get("cycles") or 0),
            "signals": int(payload.get("signals") or 0),
            "orders": int(payload.get("orders") or 0),
            "open_books": int(payload.get("open_books") or 0),
            "collector_seen": collector is not None,
            "collector_age_s": _age(collector),
            "collector": _payload(collector["payload"]) if collector else {},
        }

    async def arming(self) -> dict[str, Any] | None:
        row = await self.db.fetchrow(
            "SELECT * FROM active_arming WHERE account_id = $1", self.account_id)
        if row is None:
            return None
        out = dict(row)
        out["expires_in_s"] = (out["expires_at"]
                               - datetime.now(timezone.utc)).total_seconds()
        return out

    # ── equity and performance ───────────────────────────────────────────

    async def equity_latest(self) -> dict[str, Any] | None:
        row = await self.db.fetchrow(
            """SELECT ts, equity, cash, positions_value, gross_exposure, drawdown
               FROM equity_curve WHERE account_id = $1
               ORDER BY ts DESC LIMIT 1""", self.account_id)
        return dict(row) if row else None

    async def equity_series(self, hours: int = 168) -> list[dict[str, Any]]:
        """Down-sampled so a year of minute marks does not reach the browser.

        `time_bucket` width scales with the window: the chart is ~900 px wide
        and there is no point shipping more points than it has pixels.
        """
        bucket = ("1 minute" if hours <= 6 else
                  "5 minutes" if hours <= 48 else
                  "1 hour" if hours <= 24 * 60 else "1 day")
        rows = await self.db.fetch(
            f"""SELECT time_bucket(interval '{bucket}', ts) AS bucket,
                       last(equity, ts)   AS equity,
                       max(drawdown)      AS drawdown
                FROM equity_curve
                WHERE account_id = $1 AND ts > now() - ($2 || ' hours')::interval
                GROUP BY bucket ORDER BY bucket""",
            self.account_id, str(hours))
        return [{"ts": r["bucket"], "equity": _f(r["equity"]),
                 "drawdown": _f(r["drawdown"])} for r in rows]

    async def performance(self) -> dict[str, Any]:
        """Point-in-time changes, computed by the database over stored marks.

        Deliberately not a re-implementation of `ct/backtest/metrics.py`: these
        are differences between recorded equity values, not risk statistics.
        Anything that deserves a Sharpe comes from a backtest record.
        """
        row = await self.db.fetchrow(
            """WITH cur AS (
                   SELECT equity, ts FROM equity_curve WHERE account_id = $1
                   ORDER BY ts DESC LIMIT 1)
               SELECT (SELECT equity FROM cur) AS equity,
                      (SELECT ts FROM cur)     AS ts,
                      (SELECT equity FROM equity_curve WHERE account_id = $1
                         AND ts <= now() - interval '1 day'
                       ORDER BY ts DESC LIMIT 1) AS day_ago,
                      (SELECT equity FROM equity_curve WHERE account_id = $1
                         AND ts <= now() - interval '7 days'
                       ORDER BY ts DESC LIMIT 1) AS week_ago,
                      (SELECT equity FROM equity_curve WHERE account_id = $1
                       ORDER BY ts ASC LIMIT 1)  AS first_equity,
                      (SELECT max(equity) FROM equity_curve WHERE account_id = $1)
                                                 AS peak,
                      (SELECT max(drawdown) FROM equity_curve WHERE account_id = $1)
                                                 AS max_dd,
                      (SELECT count(*) FROM equity_curve WHERE account_id = $1)
                                                 AS marks""",
            self.account_id)
        empty = {"has_data": False, "equity": None, "ts": None, "day": None,
                 "week": None, "since_start": None, "peak": None,
                 "max_drawdown": None, "marks": 0}
        if row is None or row["equity"] is None:
            # Every key present, all None. A template that reaches for a
            # missing key gets an em dash from the formatter; one that reaches
            # for an *undefined* key raises during render, and a dashboard that
            # 500s because nothing has happened yet is the worst possible
            # failure mode for a system whose normal state is "no data".
            return empty

        equity = _f(row["equity"])

        def _chg(before) -> float | None:
            base = _f(before, 0.0)
            return None if base <= 0 else equity / base - 1.0

        return {
            "has_data": True,
            "equity": equity,
            "ts": row["ts"],
            "day": _chg(row["day_ago"]),
            "week": _chg(row["week_ago"]),
            "since_start": _chg(row["first_equity"]),
            "peak": _f(row["peak"]),
            "max_drawdown": _f(row["max_dd"]),
            "marks": int(row["marks"] or 0),
        }

    # ── positions, orders, fills ─────────────────────────────────────────

    async def positions(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT strategy, symbol, qty, avg_entry, realized_pnl, fees_paid,
                      stop_price, target_price, trail_atr_mult, opened_at,
                      updated_at
               FROM positions WHERE account_id = $1 AND qty <> 0
               ORDER BY symbol, strategy""", self.account_id)
        return [dict(r) for r in rows]

    async def symbol_exposure(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT symbol, net_qty, gross_qty, open_books, realized_pnl,
                      fees_paid
               FROM positions_by_symbol
               WHERE account_id = $1 AND gross_qty <> 0 ORDER BY symbol""",
            self.account_id)
        return [dict(r) for r in rows]

    async def open_orders(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT client_order_id, strategy, symbol, side, type, tif, qty,
                      filled_qty, limit_price, status, reduce_only, created_at,
                      submitted_at
               FROM orders
               WHERE account_id = $1 AND status IN ('pending','open','partial')
               ORDER BY created_at DESC""", self.account_id)
        return [dict(r) for r in rows]

    async def strategy_states(self) -> list[dict[str, Any]]:
        """Lifecycle rows, for surfaces that phrase rather than tabulate."""
        rows = await self.db.fetch(
            """SELECT name, status, status_reason, updated_at
               FROM strategies ORDER BY name""")
        return [dict(r) for r in rows]

    async def fills_count(self) -> int:
        return int(await self.db.fetchval(
            """SELECT count(*) FROM fills f JOIN orders o ON o.id = f.order_id
               WHERE o.account_id = $1""", self.account_id) or 0)

    async def recent_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT client_order_id, strategy, symbol, side, type, qty,
                      filled_qty, avg_fill_price, fee_paid, status, created_at,
                      terminal_at, error_code
               FROM orders WHERE account_id = $1
               ORDER BY created_at DESC LIMIT $2""", self.account_id, limit)
        return [dict(r) for r in rows]

    async def recent_fills(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT f.ts, f.qty, f.price, f.fee, f.is_maker, o.symbol,
                      o.strategy, o.side, o.client_order_id
               FROM fills f JOIN orders o ON o.id = f.order_id
               WHERE o.account_id = $1
               ORDER BY f.ts DESC LIMIT $2""", self.account_id, limit)
        return [dict(r) for r in rows]

    async def execution_quality(self, days: int = 30) -> dict[str, Any]:
        """Maker ratio and fees — the direct measure of the largest claimed edge.

        ARCHITECTURE.md §0.1 puts the maker-first advantage at 30 bps per round
        trip, which is bigger than most published retail edges. If it is not
        actually being realised, this is where that shows up first.
        """
        row = await self.db.fetchrow(
            """SELECT count(*)                                        AS fills,
                      coalesce(sum(f.qty * f.price), 0)               AS notional,
                      coalesce(sum(f.fee), 0)                         AS fees,
                      coalesce(sum(f.qty) FILTER (WHERE f.is_maker), 0)
                          / nullif(sum(f.qty), 0)                     AS maker_ratio
               FROM fills f JOIN orders o ON o.id = f.order_id
               WHERE o.account_id = $1 AND f.ts > now() - ($2 || ' days')::interval""",
            self.account_id, str(days))
        return {
            "fills": int(row["fills"] or 0) if row else 0,
            "notional": _f(row["notional"]) if row else 0.0,
            "fees": _f(row["fees"]) if row else 0.0,
            # NULL means "no fills", not "no maker fills" — the distinction the
            # trade_lifecycles view got wrong once already (§8A).
            "maker_ratio": (None if row is None or row["maker_ratio"] is None
                            else _f(row["maker_ratio"])),
        }

    # ── trade lifecycles (the honest "trade history") ────────────────────

    async def lifecycles(self, *, limit: int = 100, closed_only: bool = False,
                         strategy: str | None = None, symbol: str | None = None,
                         ) -> list[dict[str, Any]]:
        """From the `trade_lifecycles` view over the event log.

        The `trades` table has **no writer anywhere in the codebase**, so a page
        built on it would be permanently empty and look like "no trades yet"
        rather than "nothing records this". Lifecycles are derived from events
        that are definitely written, and they carry the maker ratio, the
        rejection reason and the causal chain — more than the table would.
        """
        rows = await self.db.fetch(
            """SELECT * FROM trade_lifecycles
               WHERE ($1::boolean IS NOT TRUE OR is_closed)
                 AND ($2::text IS NULL OR strategy = $2)
                 AND ($3::text IS NULL OR symbol = $3)
               ORDER BY first_seq DESC LIMIT $4""",
            closed_only, strategy, symbol, limit)
        return [dict(r) for r in rows]

    async def lifecycle_events(self, correlation_id: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT seq, ts, type, severity, source, causation_id, payload
               FROM events WHERE correlation_id = $1 ORDER BY seq""",
            correlation_id)
        return [{**dict(r), "payload": _payload(r["payload"])} for r in rows]

    # ── strategies ───────────────────────────────────────────────────────

    async def strategy_stats(self, days: int = 90) -> list[dict[str, Any]]:
        """Per-strategy reality: signals emitted, how many risk let through,
        fills, fees, and what it currently holds.

        Built from `signals` + `fills` + `positions` rather than the
        `strategies` table, which nothing populates — the engine's roster comes
        from `QD_STRATEGIES` at boot, so the truthful answer to "what is running"
        is "what has produced rows".
        """
        rows = await self.db.fetch(
            """WITH sig AS (
                   SELECT strategy,
                          count(*)                                  AS signals,
                          count(*) FILTER (WHERE accepted)          AS accepted,
                          max(ts)                                   AS last_signal
                   FROM signals
                   WHERE account_id = $1 AND ts > now() - ($2||' days')::interval
                   GROUP BY strategy),
               fil AS (
                   SELECT o.strategy,
                          count(*)                                  AS fills,
                          coalesce(sum(f.fee), 0)                   AS fees,
                          coalesce(sum(f.qty) FILTER (WHERE f.is_maker), 0)
                              / nullif(sum(f.qty), 0)               AS maker_ratio
                   FROM fills f JOIN orders o ON o.id = f.order_id
                   WHERE o.account_id = $1
                   GROUP BY o.strategy),
               pos AS (
                   SELECT strategy,
                          count(*) FILTER (WHERE qty <> 0)          AS open_books,
                          coalesce(sum(realized_pnl), 0)            AS realized
                   FROM positions WHERE account_id = $1 GROUP BY strategy)
               SELECT coalesce(sig.strategy, fil.strategy, pos.strategy) AS strategy,
                      coalesce(sig.signals, 0)      AS signals,
                      coalesce(sig.accepted, 0)     AS accepted,
                      sig.last_signal,
                      coalesce(fil.fills, 0)        AS fills,
                      coalesce(fil.fees, 0)         AS fees,
                      fil.maker_ratio,
                      coalesce(pos.open_books, 0)   AS open_books,
                      coalesce(pos.realized, 0)     AS realized
               FROM sig FULL OUTER JOIN fil ON sig.strategy = fil.strategy
                        FULL OUTER JOIN pos ON pos.strategy
                             = coalesce(sig.strategy, fil.strategy)
               ORDER BY 1""",
            self.account_id, str(days))
        return [dict(r) for r in rows]

    async def rejections(self, days: int = 7) -> list[dict[str, Any]]:
        """Why risk said no, most frequent first.

        `EDGE_BELOW_FEES` at the top of this list is the system working as
        designed, not a fault — that distinction is printed next to it.
        """
        rows = await self.db.fetch(
            """SELECT reject_code AS code, count(*) AS n, max(ts) AS last_ts
               FROM signals
               WHERE account_id = $1 AND NOT accepted
                 AND ts > now() - ($2 || ' days')::interval
               GROUP BY 1 ORDER BY n DESC""", self.account_id, str(days))
        return [dict(r) for r in rows]

    async def recent_signals(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT ts, strategy, symbol, direction, strength, ref_price,
                      accepted, reject_code, reject_detail, sized_qty
               FROM signals WHERE account_id = $1
               ORDER BY ts DESC LIMIT $2""", self.account_id, limit)
        return [dict(r) for r in rows]

    # ── risk ─────────────────────────────────────────────────────────────

    async def risk_events(self, *, unresolved_only: bool = True,
                          limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT id, ts, severity, code, detail, payload, resolved_at
               FROM risk_events
               WHERE (account_id = $1 OR account_id IS NULL)
                 AND ($2::boolean IS NOT TRUE OR resolved_at IS NULL)
               ORDER BY ts DESC LIMIT $3""",
            self.account_id, unresolved_only, limit)
        return [{**dict(r), "payload": _payload(r["payload"])} for r in rows]

    async def risk_config(self) -> dict[str, Any]:
        row = await self.db.fetchrow(
            """SELECT id, limits, effective_from, created_by, note
               FROM risk_config WHERE account_id = $1
               ORDER BY effective_from DESC LIMIT 1""", self.account_id)
        if row is None:
            return {}
        return {**dict(row), "limits": _payload(row["limits"])}

    async def promotion(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            "SELECT * FROM promotion_status ORDER BY strategy")
        return [dict(r) for r in rows]

    # ── data health ──────────────────────────────────────────────────────

    async def data_coverage(self, exchange: str = "kraken",
                            ) -> list[dict[str, Any]]:
        """One row per stored series, newest bar first.

        `age_s` is the number the STALE_DATA gate is really about, so it is
        shown rather than a green tick: the operator should see the same figure
        the risk engine sees.
        """
        rows = await self.db.fetch(
            """SELECT symbol, timeframe, count(*) AS bars, min(ts) AS earliest,
                      max(ts) AS latest,
                      count(*) FILTER (WHERE is_gap_fill) AS imported
               FROM candles WHERE exchange = $1
               GROUP BY symbol, timeframe ORDER BY symbol, timeframe""",
            exchange)
        now = datetime.now(timezone.utc)
        return [{**dict(r),
                 "age_s": (now - r["latest"]).total_seconds() if r["latest"]
                          else None} for r in rows]

    # ── activity ─────────────────────────────────────────────────────────

    async def events(self, *, limit: int = 100, after_seq: int | None = None,
                     severity: str | None = None, kind: str | None = None,
                     ) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT seq, ts, type, source, severity, correlation_id, payload
               FROM events
               WHERE ($1::bigint IS NULL OR seq > $1)
                 AND ($2::text IS NULL OR severity = $2)
                 AND ($3::text IS NULL OR type LIKE $3 || '%')
               ORDER BY seq DESC LIMIT $4""",
            after_seq, severity, kind, limit)
        return [{**dict(r), "payload": _payload(r["payload"])} for r in rows]

    async def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT id, ts, actor, action, target, before, after
               FROM audit_log ORDER BY ts DESC LIMIT $1""", limit)
        return [{**dict(r), "after": _payload(r["after"]),
                 "before": _payload(r["before"])} for r in rows]

    async def control_commands(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """SELECT id, ts, command, args, requested_by, applied_at, result
               FROM control_commands
               WHERE account_id = $1 OR account_id IS NULL
               ORDER BY id DESC LIMIT $2""", self.account_id, limit)
        return [{**dict(r), "args": _payload(r["args"])} for r in rows]

    async def notifications(self, limit: int = 25) -> dict[str, Any]:
        rows = await self.db.fetch(
            """SELECT status, count(*) AS n FROM notifications GROUP BY status""")
        recent = await self.db.fetch(
            """SELECT ts, channel, priority, subject, body, status, attempts,
                      last_error
               FROM notifications ORDER BY id DESC LIMIT $1""", limit)
        return {"counts": {r["status"]: int(r["n"]) for r in rows},
                "recent": [dict(r) for r in recent]}

    # ── system ───────────────────────────────────────────────────────────

    async def db_stats(self) -> dict[str, Any]:
        size = await self.db.fetchval(
            "SELECT pg_size_pretty(pg_database_size(current_database()))")
        try:
            version = await self.db.fetchrow(
                """SELECT sha256, applied_at, source FROM schema_version
                   ORDER BY applied_at DESC LIMIT 1""")
        except Exception:
            # A database initialised by the Postgres entrypoint (schema.sql in
            # /docker-entrypoint-initdb.d) has no `schema_version` — only
            # `bootstrap()` creates that marker. The System page 500'd on
            # exactly such a database (found in production, 2026-07-30). A
            # missing provenance marker is a fact to display, not an error;
            # the template renders it as "not stamped".
            version = None
        counts = await self.db.fetchrow(
            """SELECT (SELECT count(*) FROM events)   AS events,
                      (SELECT count(*) FROM candles)  AS candles,
                      (SELECT count(*) FROM orders)   AS orders,
                      (SELECT count(*) FROM fills)    AS fills,
                      (SELECT count(*) FROM signals)  AS signals""")
        rate = await self.db.fetchval(
            """SELECT count(*) FROM events WHERE ts > now() - interval '1 hour'""")
        return {
            "size": size,
            "schema_sha": (version["sha256"][:12] if version else None),
            "schema_applied": version["applied_at"] if version else None,
            "counts": dict(counts) if counts else {},
            "events_last_hour": int(rate or 0),
        }

    async def accounts(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            "SELECT id, name, mode, exchange, initial_equity FROM accounts "
            "ORDER BY id")
        return [dict(r) for r in rows]


def host_health(path: str = "/hosthealth/metrics.json") -> dict[str, Any]:
    """The NAS's own metrics, produced by the existing `health-collector`.

    Read from a file rather than re-implemented, because that container already
    solves it and two collectors disagreeing about CPU load is a support call
    nobody needs. Absent file is the normal case off-NAS and returns a marker
    the page renders as "not mounted" rather than as zeroes — zeroes would look
    like a healthy idle machine.
    """
    from pathlib import Path

    p = Path(path)
    try:
        if not p.is_file():
            return {"available": False, "reason": f"{path} is not mounted"}
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"unreadable: {exc}"}

    ram_total = _f(data.get("ram_total_kb"))
    ram_avail = _f(data.get("ram_avail_kb"))
    disk_total = _f(data.get("disk_total"))
    disk_used = _f(data.get("disk_used"))
    stamp = str(data.get("ts", ""))
    return {
        "available": True,
        "ts": stamp,
        "cpu_pct": _f(data.get("cpu_pct")),
        "load1": _f(data.get("load1")),
        "cores": int(_f(data.get("cores"), 1)),
        "cpu_temp": _f(data.get("cpu_temp")),
        "ram_pct": (100.0 * (1 - ram_avail / ram_total)) if ram_total else 0.0,
        "ram_avail_mb": ram_avail / 1024.0,
        "ram_total_mb": ram_total / 1024.0,
        "disk_pct": (100.0 * disk_used / disk_total) if disk_total else 0.0,
        "disk_free_tb": (disk_total - disk_used) / 1e12 if disk_total else 0.0,
        "ping_ms": _f(data.get("ping_ms")),
        "uptime_days": _f(data.get("uptime_s")) / 86400.0,
        "hist": data.get("hist") or {},
        "containers": data.get("containers") or [],
        "smart": data.get("smart") or [],
    }


def freshness(age_s: float | None, timeframe: str = "4h") -> str:
    """`ok` | `slow` | `stale` for a data series, in bar units not seconds.

    Bars, because that is the unit the risk engine's `max_data_age_bars` gate
    uses — a page that grades freshness in seconds would disagree with the gate
    that actually blocks trading.
    """
    from ct.data.integrity import timeframe_seconds

    if age_s is None:
        return "stale"
    step = timeframe_seconds(timeframe)
    bars = age_s / step
    return "ok" if bars <= 1.5 else "slow" if bars <= 3 else "stale"


def timedelta_words(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    seconds = int(abs(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"


__all__ = [
    "HEARTBEAT_STALE_S",
    "Queries",
    "freshness",
    "host_health",
    "timedelta_words",
]

