"""Transactional notification outbox.

The failure this exists to prevent: **a trade executed and you were never told.**

A separate notification service consuming a queue adds a second thing that can
be down during exactly that event. Instead the notification row is written in
the SAME database transaction as the fill it announces (see
`OrderRepository.record_fill`). If the drain is down the message is not lost —
it is a row you can see. If the engine is down, no trade happened.

Delivery is at-least-once by design. A duplicate Telegram message is a mild
annoyance; a missed fill notification is not. When those two trade off, take
the annoyance.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

import asyncpg

from ct.store.db import Database

log = logging.getLogger(__name__)

# Exponential, capped. A dead Telegram must not turn into a tight retry loop
# hammering a 4-core NAS.
BACKOFF_SECONDS = [5, 30, 120, 600, 1800, 3600]
MAX_ATTEMPTS = len(BACKOFF_SECONDS) + 1

Sender = Callable[[str, str | None, str], Awaitable[bool]]
"""(channel, subject, body) -> delivered?  Must not raise; return False."""


class NotificationOutbox:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.sent = 0
        self.failed = 0
        self.dropped = 0

    async def enqueue(self, channel: str, body: str, *, subject: str | None = None,
                      priority: str = "normal") -> int:
        """Standalone enqueue. Prefer writing the row inside the transaction
        that produced the event — this is for notifications with no such
        transaction (health alerts, daily summaries)."""
        return int(await self.db.fetchval(
            """INSERT INTO notifications (channel, priority, subject, body)
               VALUES ($1,$2,$3,$4) RETURNING id""",
            channel, priority, subject, body))

    async def pending(self, limit: int = 50) -> list[asyncpg.Record]:
        return await self.db.fetch(
            """SELECT * FROM notifications
               WHERE status = 'pending' AND next_attempt_at <= now()
               ORDER BY
                   CASE priority WHEN 'loud' THEN 0 WHEN 'normal' THEN 1
                                 ELSE 2 END,
                   id
               LIMIT $1""", limit)

    async def drain(self, send: Sender, *, limit: int = 50) -> tuple[int, int]:
        """Attempt every due notification once. Returns (sent, failed).

        Never raises: a notification channel being down must not propagate into
        whatever called the drain. Each row is claimed with `FOR UPDATE
        SKIP LOCKED` so two drains cannot double-send the same row — which
        matters the moment there is more than one process, and costs nothing
        when there is one.
        """
        sent = failed = 0
        for _ in range(limit):
            async with self.db.transaction() as conn:
                row = await conn.fetchrow(
                    """SELECT * FROM notifications
                       WHERE status='pending' AND next_attempt_at <= now()
                       ORDER BY CASE priority WHEN 'loud' THEN 0
                                              WHEN 'normal' THEN 1 ELSE 2 END, id
                       FOR UPDATE SKIP LOCKED LIMIT 1""")
                if row is None:
                    break
                try:
                    ok = await send(row["channel"], row["subject"], row["body"])
                    err = None
                except Exception as exc:
                    ok, err = False, repr(exc)
                    log.exception("notification sender raised")

                if ok:
                    await conn.execute(
                        """UPDATE notifications SET status='sent', sent_at=now(),
                               attempts = attempts + 1 WHERE id=$1""", row["id"])
                    sent += 1
                    self.sent += 1
                    continue

                attempts = int(row["attempts"]) + 1
                failed += 1
                self.failed += 1
                if attempts >= MAX_ATTEMPTS:
                    # Give up loudly. A row stuck retrying forever is a queue
                    # that never drains and hides everything behind it.
                    await conn.execute(
                        """UPDATE notifications SET status='failed',
                               attempts=$2, last_error=$3 WHERE id=$1""",
                        row["id"], attempts, err or "delivery refused")
                    self.dropped += 1
                    log.error("notification %s failed permanently after %d "
                              "attempts: %s", row["id"], attempts, err)
                else:
                    delay = BACKOFF_SECONDS[min(attempts - 1,
                                                len(BACKOFF_SECONDS) - 1)]
                    await conn.execute(
                        """UPDATE notifications SET attempts=$2, last_error=$3,
                               next_attempt_at = now() + ($4 || ' seconds')::interval
                           WHERE id=$1""",
                        row["id"], attempts, err or "delivery refused", str(delay))
        return sent, failed

    async def backlog(self) -> dict[str, int]:
        """Health signal. A growing pending count means nobody is being told
        anything, which is indistinguishable from 'nothing happened' until you
        look here."""
        rows = await self.db.fetch(
            "SELECT status, count(*) n FROM notifications GROUP BY 1")
        return {r["status"]: int(r["n"]) for r in rows}

    async def oldest_pending_age_seconds(self) -> float | None:
        ts: datetime | None = await self.db.fetchval(
            "SELECT min(ts) FROM notifications WHERE status = 'pending'")
        if ts is None:
            return None
        return (datetime.now(ts.tzinfo) - ts).total_seconds()


__all__ = ["BACKOFF_SECONDS", "MAX_ATTEMPTS", "NotificationOutbox", "Sender"]
