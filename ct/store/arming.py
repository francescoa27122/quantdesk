"""Live-trading arming — the code half of the invariant in ARCHITECTURE.md §8.4.

**Every arming chain must terminate in a human tap.** Enforced in three
independent places, because the code-level check is the one most likely to be
refactored away:

1. schema `CHECK` constraints — a renewal cannot be inserted without a parent;
   a human arming cannot be inserted without a typed confirmation
2. `arming_is_valid(id)` in SQL — walks the lineage, refuses if any ancestor is
   disarmed
3. this module — computes the envelope hash and refuses to renew a changed one

Claude may auto-renew, disarm, and re-arm after its *own* disarm for a verified
transient cause. Claude may **not** originate the first arm; `create_human()`
requires a confirmation phrase that only a person can supply.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import asyncpg

from ct.store.db import Database

log = logging.getLogger(__name__)

# Renewal is refused after any of these; a fresh human tap is required.
HARD_DISARM_REASONS = frozenset({
    "max_drawdown", "promotion_gate_regressed", "envelope_changed",
    "manual", "code_changed",
})


def envelope_hash(
    *, account_id: int, symbols: Sequence[str], strategies: Sequence[str],
    max_notional_usd: Decimal, max_daily_notional_usd: Decimal,
    risk_config_id: int, code_git_sha: str,
) -> str:
    """Binds a renewal to unchanged terms.

    Any change to caps, symbols, strategies, risk config or strategy code breaks
    the hash, renewal stops, and a human is asked again. That is precisely what
    makes indefinite renewal safe: it renews *the same decision*, never a new
    one. Sorted so ordering cannot produce a spurious mismatch.
    """
    parts = [
        str(account_id),
        ",".join(sorted(symbols)),
        ",".join(sorted(strategies)),
        f"{Decimal(max_notional_usd):.8f}",
        f"{Decimal(max_daily_notional_usd):.8f}",
        str(risk_config_id),
        code_git_sha,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class ArmingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ── the human gate ───────────────────────────────────────────────────

    async def create_human(
        self, account_id: int, *, armed_by: str, confirmation_phrase: str,
        envelope: str, symbols: Sequence[str], strategies: Sequence[str],
        max_notional_usd: Decimal, max_daily_notional_usd: Decimal,
        hours: int = 168, gate_forced: bool = False,
    ) -> int:
        """Root a new arming chain. Requires a typed confirmation phrase.

        This is the only function in the codebase that can put capital at risk,
        and it cannot be called meaningfully without a human having typed
        something. Claude does not call it.
        """
        if not confirmation_phrase or not confirmation_phrase.strip():
            raise ValueError(
                "arming requires a typed confirmation phrase; this gate exists "
                "so going live is a decision rather than a drift")
        digest = hashlib.sha256(confirmation_phrase.strip().encode()).hexdigest()
        return int(await self.db.fetchval(
            """INSERT INTO live_arming (account_id, expires_at, armed_by, origin,
                   envelope_hash, confirmation_hash, max_notional_usd,
                   max_daily_notional_usd, allowed_symbols, allowed_strategies,
                   gate_forced)
               VALUES ($1, now() + ($2 || ' hours')::interval, $3, 'human', $4,
                       $5, $6, $7, $8, $9, $10)
               RETURNING id""",
            account_id, str(hours), armed_by, envelope, digest,
            max_notional_usd, max_daily_notional_usd, list(symbols),
            list(strategies), gate_forced))

    # ── what the engine may do on its own ────────────────────────────────

    async def active(self, account_id: int) -> asyncpg.Record | None:
        """The single question asked before every live order."""
        return await self.db.fetchrow(
            "SELECT * FROM active_arming WHERE account_id = $1", account_id)

    async def is_armed(self, account_id: int) -> bool:
        return await self.active(account_id) is not None

    async def renew(self, account_id: int, envelope: str, *,
                    hours: int = 168, renewed_by: str = "qd-engine",
                    ) -> int | None:
        """Extend a live arming whose envelope is unchanged. Returns the new id.

        Returns None — and records why — when the envelope has changed. Silent
        failure here would mean either trading under terms nobody approved, or
        stopping without saying so.
        """
        current = await self.active(account_id)
        if current is None:
            log.info("no active arming for account %s; nothing to renew", account_id)
            return None
        if current["envelope_hash"] != envelope:
            log.warning("envelope changed for account %s; renewal refused",
                        account_id)
            return None
        return int(await self.db.fetchval(
            """INSERT INTO live_arming (account_id, expires_at, armed_by, origin,
                   renews_id, envelope_hash, max_notional_usd,
                   max_daily_notional_usd, allowed_symbols, allowed_strategies,
                   gate_forced)
               VALUES ($1, now() + ($2 || ' hours')::interval, $3, 'auto_renew',
                       $4, $5, $6, $7, $8, $9, $10)
               RETURNING id""",
            account_id, str(hours), renewed_by, current["id"], envelope,
            current["max_notional_usd"], current["max_daily_notional_usd"],
            list(current["allowed_symbols"]), list(current["allowed_strategies"]),
            current["gate_forced"]))

    async def due_for_renewal(self, account_id: int,
                              within: timedelta = timedelta(hours=24)) -> bool:
        row = await self.active(account_id)
        if row is None:
            return False
        expires: datetime = row["expires_at"]
        return expires - datetime.now(expires.tzinfo) <= within

    async def disarm(self, account_id: int, *, reason: str,
                     disarmed_by: str = "qd-engine") -> int:
        """Disarm every live row for the account. Always permitted, never gated.

        Disarming a link poisons its whole lineage (`arming_is_valid` refuses if
        any ancestor is disarmed), but every row is marked anyway so the audit
        trail says plainly what was stopped and when.
        """
        rows = await self.db.fetch(
            """UPDATE live_arming SET disarmed_at = now(), disarmed_by = $2,
                   disarm_reason = $3
               WHERE account_id = $1 AND disarmed_at IS NULL
               RETURNING id""",
            account_id, disarmed_by, reason)
        if rows:
            log.warning("disarmed %d arming rows for account %s: %s",
                        len(rows), account_id, reason)
        return len(rows)

    async def may_auto_rearm(self, account_id: int) -> bool:
        """True only if the most recent disarm was a verified transient cause.

        Drawdown, gate regression and any code or config change need a fresh
        human tap — those are judgements about whether to keep risking capital,
        not faults that resolve themselves.
        """
        reason = await self.db.fetchval(
            """SELECT disarm_reason FROM live_arming
               WHERE account_id = $1 AND disarmed_at IS NOT NULL
               ORDER BY disarmed_at DESC LIMIT 1""", account_id)
        if reason is None:
            return False
        return reason not in HARD_DISARM_REASONS

    async def chain(self, arming_id: int) -> list[asyncpg.Record]:
        """The lineage from `arming_id` back to its root, oldest first."""
        return await self.db.fetch(
            """WITH RECURSIVE up AS (
                   SELECT * FROM live_arming WHERE id = $1
                   UNION ALL
                   SELECT a.* FROM live_arming a JOIN up ON up.renews_id = a.id)
               SELECT * FROM up ORDER BY id""", arming_id)

    async def is_human_rooted(self, arming_id: int) -> bool:
        return bool(await self.db.fetchval(
            "SELECT arming_is_valid($1)", arming_id))


__all__ = ["HARD_DISARM_REASONS", "ArmingRepository", "envelope_hash"]
