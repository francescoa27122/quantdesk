"""Persistence layer.

`ct.store` is the only package that talks to Postgres. Everything above it —
risk, strategies, the backtester — stays pure and testable without a database.

    db = Database(dsn)
    await db.connect()
    await db.bootstrap("db/schema.sql")

    recon = ReconciliationService(db)
    result = await recon.run(account_id, exchange_balances)
    if not result.ok:
        ...   # engine stays halted; this is not optional

See ARCHITECTURE.md §6.3 (reconciliation) and §8.4 (arming).
"""

from ct.store.arming import HARD_DISARM_REASONS, ArmingRepository, envelope_hash
from ct.store.db import Database, DatabaseError, IntegrityError
from ct.store.events import PostgresEventSink, TeeEventLog, backfill_from_jsonl
from ct.store.outbox import NotificationOutbox
from ct.store.reconcile import (
    DUST,
    Discrepancy,
    ReconciliationResult,
    ReconciliationService,
)
from ct.store.repositories import (
    EquityRepository,
    OrderRepository,
    PositionRepository,
    RiskEventRepository,
    SignalRepository,
    fill_from_record,
)
from ct.store.strategies import StrategyRepository, params_hash

__all__ = [
    "DUST",
    "HARD_DISARM_REASONS",
    "ArmingRepository",
    "Database",
    "DatabaseError",
    "Discrepancy",
    "EquityRepository",
    "IntegrityError",
    "NotificationOutbox",
    "OrderRepository",
    "PositionRepository",
    "PostgresEventSink",
    "ReconciliationResult",
    "ReconciliationService",
    "RiskEventRepository",
    "SignalRepository",
    "StrategyRepository",
    "TeeEventLog",
    "backfill_from_jsonl",
    "envelope_hash",
    "fill_from_record",
    "params_hash",
]
