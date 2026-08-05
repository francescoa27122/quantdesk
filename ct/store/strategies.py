"""Strategy lifecycle persistence — the `strategies` table's first writer.

Until this existed the table had no writer anywhere in the codebase, which had
a consequence beyond an empty dashboard panel: the engine kept quarantines in
`RuntimeState.paused_strategies`, a set rebuilt on every boot, so **a strategy
that crashed and was stopped came back alive after any restart**. With
`restart: unless-stopped` on the container, that is not a rare path.

The FSM in `ct/strategies/lifecycle.py` decides; this stores. Every transition
also writes an `audit_log` row, because "who stopped this strategy and when" is
exactly the question asked after an incident.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from ct.store.db import Database
from ct.strategies.lifecycle import Actor, StrategyLifecycle, StrategyState

log = logging.getLogger(__name__)


def params_hash(params: dict[str, Any]) -> str:
    """Stable hash of a parameter set.

    Sorted keys and a canonical separator, so the same parameters hash the same
    whatever order they were built in — this feeds backtest reproducibility and
    would otherwise be a silent source of cache misses.
    """
    blob = json.dumps(params or {}, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class StrategyRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert(self, name: str, *, class_path: str,
                     params: dict[str, Any] | None = None,
                     symbols: list[str] | None = None, timeframe: str = "4h",
                     weight: float = 0.0) -> None:
        """Register a strategy without disturbing its state.

        `ON CONFLICT` deliberately does **not** touch `status`, `enabled` or
        `status_reason`: a redeploy re-registers every strategy, and if that
        reset the status it would silently un-quarantine the one thing somebody
        had stopped on purpose. Registration is not a lifecycle transition.
        """
        await self.db.execute(
            """INSERT INTO strategies (name, class_path, params, params_hash,
                                       symbols, timeframe, weight, status)
               VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7,$8)
               ON CONFLICT (name) DO UPDATE SET
                   class_path = EXCLUDED.class_path,
                   params     = EXCLUDED.params,
                   params_hash = EXCLUDED.params_hash,
                   symbols    = EXCLUDED.symbols,
                   timeframe  = EXCLUDED.timeframe,
                   weight     = EXCLUDED.weight,
                   updated_at = now()""",
            name, class_path, json.dumps(params or {}, default=str),
            params_hash(params or {}), symbols or [], timeframe, weight,
            StrategyState.PROPOSED.value)

    async def load(self, name: str) -> StrategyLifecycle | None:
        row = await self.db.fetchrow(
            """SELECT name, status, status_reason, paused_until, updated_at
               FROM strategies WHERE name = $1""", name)
        return _from_row(row) if row else None

    async def load_all(self) -> dict[str, StrategyLifecycle]:
        rows = await self.db.fetch(
            """SELECT name, status, status_reason, paused_until, updated_at
               FROM strategies ORDER BY name""")
        return {r["name"]: _from_row(r) for r in rows}

    async def save(self, lc: StrategyLifecycle, *, actor: Actor,
                   ) -> None:
        """Persist a state change and audit it in one place.

        Not a transaction: `strategies` and `audit_log` are independent, and a
        lost audit row must never prevent a strategy from being stopped. The
        order is deliberate — stop first, record second.
        """
        await self.db.execute(
            """UPDATE strategies
               SET status = $2, status_reason = $3, paused_until = $4,
                   enabled = $5, updated_at = now()
               WHERE name = $1""",
            lc.name, lc.state.value, lc.reason[:500], lc.paused_until,
            lc.may_run)
        try:
            await self.db.execute(
                """INSERT INTO audit_log (actor, action, target, after)
                   VALUES ($1, $2, $3, $4::jsonb)""",
                actor.value, f"strategy:{lc.state.value}", lc.name,
                json.dumps({"state": lc.state.value, "reason": lc.reason,
                            "paused_until": (lc.paused_until.isoformat()
                                             if lc.paused_until else None)}))
        except Exception:
            log.exception("could not audit %s -> %s", lc.name, lc.state.value)

    async def running(self) -> list[str]:
        """Names the engine may call `on_bar` on."""
        rows = await self.db.fetch(
            """SELECT name FROM strategies
               WHERE status = ANY($1::text[]) ORDER BY name""",
            [StrategyState.PAPER.value, StrategyState.LIVE.value])
        return [r["name"] for r in rows]


def _from_row(row: Any) -> StrategyLifecycle:
    raw = str(row["status"] or StrategyState.PROPOSED.value)
    try:
        state = StrategyState(raw)
    except ValueError:
        # An unknown status is not a reason to start trading. The schema's
        # original vocabulary was active/quarantined/paused, so anything this
        # code does not recognise is treated as needing a human.
        log.warning("strategy %s has unknown status %r; treating as quarantined",
                    row["name"], raw)
        state = StrategyState.QUARANTINED
    paused_until: datetime | None = row["paused_until"]
    return StrategyLifecycle(
        name=row["name"], state=state, reason=row["status_reason"] or "",
        paused_until=paused_until,
        resume_to=StrategyState.PAPER if state is StrategyState.PAUSED else None,
        updated_at=row["updated_at"])


__all__ = ["StrategyRepository", "params_hash"]
