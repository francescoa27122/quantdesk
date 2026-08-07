"""Connection pool, transactions, bootstrap, health.

Everything that touches Postgres goes through `Database`. Two rules it exists
to enforce:

1. **The order path never waits on the database longer than it can afford.**
   Every call carries a timeout, and a slow database surfaces as a timeout to be
   handled rather than an engine that has quietly stopped.
2. **Reconnection is automatic and observable.** A restarted `qd-db` container
   must not require restarting `qd-engine` — the pool re-establishes itself and
   the failure is counted, not swallowed.

Sizing note: `max_size=8` against a `max_connections=40` Postgres tuned for
~1 GB on a 4-core N100. Three containers × 8 connections leaves headroom for
`psql` and the backup job. Raising it will not make anything faster; the
bottleneck is the exchange, not the database.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import asyncpg

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
# A query that outlives this is not slow, it is stuck. Failing loudly beats an
# engine blocked on a lock it will never get.
STATEMENT_TIMEOUT_MS = 30_000


class DatabaseError(RuntimeError):
    """Database failure the engine may retry. Counted by the circuit breaker."""


class IntegrityError(DatabaseError):
    """A constraint refused the write — a UNIQUE, CHECK or FK violation.

    Split out from `DatabaseError` because the correct response is opposite:
    a connection failure should be retried, an integrity violation never should.
    Retrying "this renewal has no human ancestor" forever would turn a caught
    logic error into a hot loop, and would pollute the circuit breaker that is
    supposed to detect a sick database.
    """


class Database:
    """asyncpg pool with bootstrap, health and a transaction helper."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        command_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self._pool: asyncpg.Pool | None = None
        self.connect_failures = 0
        self.query_failures = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    async def connect(self, *, retries: int = 5, backoff: float = 1.0) -> None:
        """Open the pool, retrying with backoff.

        Retries because `qd-engine` and `qd-db` start together and Postgres
        takes a few seconds to accept connections. Crashing on the first refusal
        would make container start order load-bearing, which is exactly the kind
        of fragility `restart: unless-stopped` is supposed to remove.
        """
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self._pool = await asyncpg.create_pool(
                    self.dsn, min_size=self.min_size, max_size=self.max_size,
                    command_timeout=self.command_timeout,
                    server_settings={"statement_timeout": str(STATEMENT_TIMEOUT_MS),
                                     "application_name": "quantdesk"},
                )
                log.info("database pool open (min=%d max=%d)", self.min_size,
                         self.max_size)
                return
            except Exception as exc:
                last = exc
                self.connect_failures += 1
                log.warning("database connect attempt %d/%d failed: %s",
                            attempt, retries, exc)
                if attempt < retries:
                    await asyncio.sleep(backoff * attempt)
        raise DatabaseError(f"could not connect after {retries} attempts: {last}")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def is_connected(self) -> bool:
        return self._pool is not None and not self._pool._closed

    async def healthy(self) -> bool:
        try:
            return bool(await self.fetchval("SELECT 1") == 1)
        except Exception:
            return False

    # ── execution ────────────────────────────────────────────────────────

    def _require(self) -> asyncpg.Pool:
        if self._pool is None:
            raise DatabaseError("database pool is not open; call connect() first")
        return self._pool

    async def execute(self, sql: str, *args: Any) -> str:
        try:
            # asyncpg ships no stubs, so the pool is Any. These casts record
            # the documented return types at the one boundary the rest of the
            # codebase reads them through.
            return cast("str", await self._require().execute(sql, *args))
        except asyncpg.IntegrityConstraintViolationError as exc:
            raise IntegrityError(str(exc)) from exc
        except asyncpg.PostgresError as exc:
            self.query_failures += 1
            raise DatabaseError(str(exc)) from exc

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        try:
            return cast("list[asyncpg.Record]",
                        await self._require().fetch(sql, *args))
        except asyncpg.IntegrityConstraintViolationError as exc:
            raise IntegrityError(str(exc)) from exc
        except asyncpg.PostgresError as exc:
            self.query_failures += 1
            raise DatabaseError(str(exc)) from exc

    async def fetchrow(self, sql: str, *args: Any) -> asyncpg.Record | None:
        try:
            return await self._require().fetchrow(sql, *args)
        except asyncpg.IntegrityConstraintViolationError as exc:
            raise IntegrityError(str(exc)) from exc
        except asyncpg.PostgresError as exc:
            self.query_failures += 1
            raise DatabaseError(str(exc)) from exc

    async def fetchval(self, sql: str, *args: Any) -> Any:
        try:
            return await self._require().fetchval(sql, *args)
        except asyncpg.IntegrityConstraintViolationError as exc:
            raise IntegrityError(str(exc)) from exc
        except asyncpg.PostgresError as exc:
            self.query_failures += 1
            raise DatabaseError(str(exc)) from exc

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        try:
            await self._require().executemany(sql, rows)
        except asyncpg.IntegrityConstraintViolationError as exc:
            raise IntegrityError(str(exc)) from exc
        except asyncpg.PostgresError as exc:
            self.query_failures += 1
            raise DatabaseError(str(exc)) from exc

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """One atomic unit.

        This is what makes the outbox guarantee real: a fill and the
        notification announcing it are written in the same transaction, so
        "traded but never told you" cannot happen. Use it for any write that
        spans more than one table.
        """
        pool = self._require()
        async with pool.acquire() as conn, conn.transaction():
            yield conn

    # ── bootstrap ────────────────────────────────────────────────────────

    async def bootstrap(self, schema_path: str | Path) -> bool:
        """Apply `schema.sql` if the database is empty. Returns True if applied.

        Idempotent by inspection, not by `IF NOT EXISTS` everywhere: the schema
        creates hypertables, views and functions that do not all support it, so
        the check is "does the spine exist?" rather than statement-by-statement.
        Migrations after this point are Alembic's job — see `migrations/`.

        **Runs on a dedicated connection, not a pooled one.** `CREATE EXTENSION
        timescaledb` must be the first statement in its session; on a pooled
        connection that has already touched the extension Postgres refuses with
        *"extension has already been loaded with another version"*. That failure
        is connection-dependent, so it shows up intermittently — which is far
        worse than failing every time. Found running these tests against a real
        database on 2026-07-27.
        """
        exists = await self.fetchval(
            "SELECT to_regclass('public.orders') IS NOT NULL")
        sql = Path(schema_path).read_text(encoding="utf-8")
        digest = hashlib.sha256(sql.encode()).hexdigest()
        if exists:
            log.info("schema already present; skipping bootstrap")
            # A database initialised by the Postgres entrypoint mount ran
            # schema.sql without ever creating the marker, so drift detection
            # was silently disabled and anything reading schema_version broke.
            # Register a baseline late, and SAY it is late: the sha describes
            # the file now, not necessarily the file the entrypoint ran.
            registered = await self.fetchval(
                "SELECT to_regclass('public.schema_version') IS NOT NULL")
            if not registered:
                await self.execute(
                    """CREATE TABLE IF NOT EXISTS schema_version (
                           applied_at timestamptz NOT NULL DEFAULT now(),
                           sha256     text NOT NULL,
                           source     text NOT NULL)""")
                await self.execute(
                    "INSERT INTO schema_version (sha256, source) VALUES ($1, $2)",
                    digest,
                    f"{schema_path} (late registration on an already-"
                    f"initialised database; sha is of the file at registration "
                    f"time, not initialisation time)")
                log.warning("schema_version was missing (entrypoint-initialised "
                            "database); registered baseline %s", digest[:12])
            await self._warn_on_drift(digest)
            return False
        conn = await asyncpg.connect(self.dsn, timeout=self.command_timeout)
        try:
            async with conn.transaction():
                await conn.execute(sql)
        finally:
            await conn.close()
        await self.execute(
            """CREATE TABLE IF NOT EXISTS schema_version (
                   applied_at timestamptz NOT NULL DEFAULT now(),
                   sha256     text NOT NULL,
                   source     text NOT NULL)""")
        await self.execute(
            "INSERT INTO schema_version (sha256, source) VALUES ($1, $2)",
            digest, str(schema_path))
        log.info("schema bootstrapped from %s (sha256 %s)", schema_path,
                 digest[:12])
        return True

    async def _warn_on_drift(self, digest: str) -> None:
        """Warn when `schema.sql` has changed since this database was created.

        **Why this instead of Alembic, for now.** Alembic earns its keep when
        there is data you cannot afford to drop. There is none yet — the schema
        is still moving weekly and nothing has been deployed — so maintaining a
        migration chain would be pure carrying cost for a database that is
        currently recreated at will. What actually has value today is *knowing*
        the running database no longer matches the code, which is one hash.

        Adopt Alembic at the first deploy that holds real trades: stamp this
        baseline and migrate from there. `schema_version` is the marker that
        makes that switch clean rather than archaeological.
        """
        try:
            current = await self.fetchval(
                "SELECT sha256 FROM schema_version ORDER BY applied_at DESC LIMIT 1")
        except DatabaseError:
            return                      # pre-dates the marker; nothing to compare
        if current and current != digest:
            log.warning(
                "SCHEMA DRIFT: db was built from schema.sql sha256 %s but the "
                "file is now %s. The running database does not match the code.",
                current[:12], digest[:12])

    async def table_names(self) -> set[str]:
        rows = await self.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {r["tablename"] for r in rows}

    async def stats(self) -> dict[str, Any]:
        """Cheap numbers for the health endpoint."""
        pool = self._pool
        return {
            "connected": self.is_connected,
            "pool_size": pool.get_size() if pool else 0,
            "pool_idle": pool.get_idle_size() if pool else 0,
            "connect_failures": self.connect_failures,
            "query_failures": self.query_failures,
        }


__all__ = ["Database", "DatabaseError", "IntegrityError"]
