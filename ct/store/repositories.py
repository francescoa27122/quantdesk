"""Repositories — the only code that writes trading state.

Three invariants live here, and each is enforced by the database rather than by
discipline:

1. **Idempotent order submission.** `orders.client_order_id` is UNIQUE, so
   resubmitting after a restart or a retry is a no-op, not a second position.
   `insert_order` returns the existing row instead of raising.
2. **Fills are the ledger; positions are a projection.** `positions` is
   derivable by replaying `fills`, and `ReconciliationService` checks that on
   every boot. A fill is never edited, only appended.
3. **A fill and the notification announcing it are one transaction.** "Traded
   but never told you" is not a race that can happen — the outbox row is
   written or the fill is not.

Money is `Decimal` end to end. asyncpg maps `numeric` to `Decimal`, so nothing
here rounds through float.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

import asyncpg

from ct.models import Fill, Order, OrderStatus, Position, Side
from ct.store.db import Database

log = logging.getLogger(__name__)


class OrderRepository:
    """Orders and fills. The single writer is `qd-engine`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert_order(
        self, account_id: int, order: Order, *, signal_id: int | None = None,
    ) -> int:
        """Persist an intent. Idempotent on `client_order_id`.

        Returns the order's database id. If the id already exists — a retry, or
        a restart replaying its own in-flight work — the existing row is
        returned untouched. This is the difference between a retry and a
        duplicate position.
        """
        row = await self.db.fetchrow(
            """
            INSERT INTO orders (client_order_id, account_id, strategy, signal_id,
                                symbol, side, type, tif, qty, limit_price,
                                stop_price, status, reduce_only, attached_stop,
                                attached_target, trail_atr_mult, created_at, meta)
            VALUES ($1,$2,$3,$4,$5,$6::order_side,$7::order_type,$8::time_in_force,
                    $9,$10,$11,'pending'::order_status,$12,$13,$14,$15,$16,$17::jsonb)
            ON CONFLICT (client_order_id) DO NOTHING
            RETURNING id
            """,
            order.client_order_id, account_id, order.strategy, signal_id,
            order.symbol, order.side.value, order.type.value, order.tif.value,
            order.qty, order.limit_price, order.stop_price, order.reduce_only,
            order.attached_stop, order.attached_target,
            Decimal(str(order.trail_atr_mult)) if order.trail_atr_mult else None,
            order.created_at, _json(order.meta),
        )
        if row is not None:
            return int(row["id"])
        existing = await self.db.fetchval(
            "SELECT id FROM orders WHERE client_order_id = $1",
            order.client_order_id)
        log.info("order %s already persisted (id=%s); treating as idempotent",
                 order.client_order_id, existing)
        return int(existing)

    async def mark_submitted(self, client_order_id: str,
                             exchange_order_id: str | None) -> None:
        await self.db.execute(
            """UPDATE orders SET status = 'open'::order_status,
                   exchange_order_id = $2, submitted_at = now()
               WHERE client_order_id = $1 AND status = 'pending'::order_status""",
            client_order_id, exchange_order_id)

    async def mark_terminal(self, client_order_id: str, status: OrderStatus,
                            *, error_code: str | None = None,
                            error_detail: str | None = None) -> None:
        await self.db.execute(
            """UPDATE orders SET status = $2::order_status, terminal_at = now(),
                   error_code = $3, error_detail = $4
               WHERE client_order_id = $1""",
            client_order_id, status.value, error_code, error_detail)

    async def record_fill(
        self, account_id: int, fill: Fill, *,
        notification: tuple[str, str, str] | None = None,
    ) -> bool:
        """Append a fill, update its order, upsert the book — atomically.

        `notification` is `(channel, subject, body)`; when given it is written
        in the SAME transaction, which is what makes "traded but never told you"
        impossible rather than merely unlikely.

        Returns False if the fill was already recorded (duplicate delivery from
        the exchange, or a replay after restart). Exchanges do re-deliver, so
        this must be a normal outcome, not an error.
        """
        async with self.db.transaction() as conn:
            order_id = await conn.fetchval(
                "SELECT id FROM orders WHERE client_order_id = $1", fill.order_id)
            if order_id is None:
                raise ValueError(
                    f"fill references unknown order {fill.order_id}; the order "
                    f"must be persisted before it can fill")

            inserted = await conn.fetchval(
                """
                INSERT INTO fills (order_id, exchange_fill_id, ts, qty, price,
                                   fee, fee_ccy, is_maker)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (order_id, exchange_fill_id) DO NOTHING
                RETURNING id
                """,
                order_id, fill.exchange_fill_id, fill.ts, fill.qty, fill.price,
                fill.fee, None, fill.is_maker)
            if inserted is None:
                return False

            # filled_qty and avg_fill_price are DERIVED from fills, never
            # incremented in place — an increment that runs twice is a phantom
            # position, and this recomputation cannot drift.
            await conn.execute(
                """
                UPDATE orders o SET
                    filled_qty = agg.q,
                    avg_fill_price = agg.p,
                    fee_paid = agg.f,
                    status = CASE WHEN agg.q >= o.qty THEN 'filled'::order_status
                                  ELSE 'partial'::order_status END,
                    terminal_at = CASE WHEN agg.q >= o.qty THEN now()
                                       ELSE o.terminal_at END
                FROM (SELECT sum(qty) q,
                             sum(qty*price)/NULLIF(sum(qty),0) p,
                             sum(fee) f
                      FROM fills WHERE order_id = $1) agg
                WHERE o.id = $1
                """, order_id)

            await self._apply_to_book(conn, account_id, fill)

            if notification is not None:
                channel, subject, body = notification
                await conn.execute(
                    """INSERT INTO notifications (channel, priority, subject, body)
                       VALUES ($1, 'normal', $2, $3)""",
                    channel, subject, body)
            return True

    @staticmethod
    async def _apply_to_book(conn: asyncpg.Connection, account_id: int,
                             fill: Fill) -> None:
        """Apply a fill to (account, strategy, symbol) using the same
        open/add/reduce/flip arithmetic as `Position.apply_fill`.

        Kept in SQL rather than round-tripping the position into Python so the
        update is inside the fill's transaction and cannot interleave with
        another writer.
        """
        signed = fill.qty * Decimal(fill.side.sign)
        row = await conn.fetchrow(
            """SELECT qty, avg_entry, realized_pnl, fees_paid FROM positions
               WHERE account_id=$1 AND strategy=$2 AND symbol=$3 FOR UPDATE""",
            account_id, fill.strategy, fill.symbol)

        if row is None or row["qty"] == 0:
            qty, avg_entry, realized = signed, fill.price, Decimal(0)
            opened = fill.ts
        else:
            cur_qty, cur_avg = row["qty"], row["avg_entry"]
            opened = None
            if (cur_qty > 0) == (signed > 0):                     # add
                qty = cur_qty + signed
                avg_entry = (cur_avg * cur_qty + fill.price * signed) / qty
                realized = Decimal(0)
            else:
                closing = min(abs(signed), abs(cur_qty))
                direction = Decimal(1 if cur_qty > 0 else -1)
                realized = (fill.price - cur_avg) * direction * closing
                remaining = cur_qty + signed
                if remaining == 0:                                 # flat
                    qty, avg_entry = Decimal(0), Decimal(0)
                elif (remaining > 0) == (cur_qty > 0):             # partial
                    qty, avg_entry = remaining, cur_avg
                else:                                              # flip
                    qty, avg_entry, opened = remaining, fill.price, fill.ts
        await conn.execute(
            """
            INSERT INTO positions (account_id, strategy, symbol, qty, avg_entry,
                                   realized_pnl, fees_paid, opened_at, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,now())
            ON CONFLICT (account_id, strategy, symbol) DO UPDATE SET
                qty = EXCLUDED.qty,
                avg_entry = EXCLUDED.avg_entry,
                realized_pnl = positions.realized_pnl + EXCLUDED.realized_pnl,
                fees_paid = positions.fees_paid + EXCLUDED.fees_paid,
                opened_at = CASE WHEN EXCLUDED.qty = 0 THEN NULL
                                 WHEN positions.qty = 0 THEN EXCLUDED.opened_at
                                 ELSE positions.opened_at END,
                stop_price = CASE WHEN EXCLUDED.qty = 0 THEN NULL
                                  ELSE positions.stop_price END,
                target_price = CASE WHEN EXCLUDED.qty = 0 THEN NULL
                                    ELSE positions.target_price END,
                updated_at = now()
            """,
            account_id, fill.strategy, fill.symbol, qty, avg_entry,
            realized - fill.fee, fill.fee, opened)

    async def open_orders(self, account_id: int) -> list[asyncpg.Record]:
        """Non-terminal orders. On boot these are re-queried at the exchange by
        `client_order_id` before anything new is sent."""
        return await self.db.fetch(
            """SELECT * FROM orders WHERE account_id = $1
               AND status IN ('pending','open','partial')
               ORDER BY created_at""", account_id)

    async def fills_for(self, account_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch(
            """SELECT f.*, o.symbol, o.side, o.strategy, o.account_id
               FROM fills f JOIN orders o ON o.id = f.order_id
               WHERE o.account_id = $1 ORDER BY f.ts, f.id""", account_id)


class PositionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def load(self, account_id: int) -> dict[str, Position]:
        """Every book, keyed exactly as `AccountState.positions` is."""
        rows = await self.db.fetch(
            "SELECT * FROM positions WHERE account_id = $1", account_id)
        out: dict[str, Position] = {}
        for r in rows:
            out[f"{r['strategy']}|{r['symbol']}"] = Position(
                symbol=r["symbol"], strategy=r["strategy"], qty=r["qty"],
                avg_entry=r["avg_entry"], realized_pnl=r["realized_pnl"],
                fees_paid=r["fees_paid"], stop_price=r["stop_price"],
                target_price=r["target_price"],
                trail_atr_mult=float(r["trail_atr_mult"])
                if r["trail_atr_mult"] is not None else None,
                highest_close=float(r["highest_close"])
                if r["highest_close"] is not None else None,
                lowest_close=float(r["lowest_close"])
                if r["lowest_close"] is not None else None,
                opened_at=r["opened_at"])
        return out

    async def save_protective_levels(
        self, account_id: int, pos: Position,
    ) -> None:
        """Persist stop/target/trail so a restart does not lose them.

        Losing a stop across a restart is the single most expensive thing this
        layer can get wrong: the position survives, its protection does not.
        """
        await self.db.execute(
            """UPDATE positions SET stop_price=$4, target_price=$5,
                   trail_atr_mult=$6, highest_close=$7, lowest_close=$8,
                   updated_at=now()
               WHERE account_id=$1 AND strategy=$2 AND symbol=$3""",
            account_id, pos.strategy, pos.symbol, pos.stop_price,
            pos.target_price,
            Decimal(str(pos.trail_atr_mult)) if pos.trail_atr_mult else None,
            Decimal(str(pos.highest_close)) if pos.highest_close else None,
            Decimal(str(pos.lowest_close)) if pos.lowest_close else None)

    async def net_by_symbol(self, account_id: int) -> dict[str, Decimal]:
        rows = await self.db.fetch(
            "SELECT symbol, net_qty FROM positions_by_symbol WHERE account_id=$1",
            account_id)
        return {r["symbol"]: r["net_qty"] for r in rows if r["net_qty"] != 0}


class EquityRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(self, account_id: int, ts: datetime, *, equity: Decimal,
                     cash: Decimal, positions_value: Decimal,
                     unrealized: Decimal = Decimal(0),
                     gross_exposure: Decimal = Decimal(0),
                     drawdown: Decimal = Decimal(0)) -> None:
        await self.db.execute(
            """INSERT INTO equity_curve (account_id, ts, equity, cash,
                   positions_value, unrealized_pnl, gross_exposure, drawdown)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
               ON CONFLICT (account_id, ts) DO UPDATE SET
                   equity = EXCLUDED.equity, cash = EXCLUDED.cash,
                   positions_value = EXCLUDED.positions_value,
                   unrealized_pnl = EXCLUDED.unrealized_pnl,
                   gross_exposure = EXCLUDED.gross_exposure,
                   drawdown = EXCLUDED.drawdown""",
            account_id, ts, equity, cash, positions_value, unrealized,
            gross_exposure, drawdown)

    async def high_water_mark(self, account_id: int) -> Decimal:
        v = await self.db.fetchval(
            "SELECT max(equity) FROM equity_curve WHERE account_id=$1", account_id)
        return v if v is not None else Decimal(0)


class SignalRepository:
    """Every signal, INCLUDING the rejected ones.

    Without the rejections you cannot tell "the strategy is bad" from "risk is
    strangling it" — which is a SQL query, not an archaeology project.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(
        self, account_id: int, ts: datetime, strategy: str, symbol: str,
        direction: int, *, accepted: bool, strength: float = 1.0,
        ref_price: Decimal | None = None, stop_price: Decimal | None = None,
        regime: str = "unknown", reject_code: str | None = None,
        reject_detail: str | None = None, sized_qty: Decimal | None = None,
        ml_confidence: float | None = None, meta: dict[str, Any] | None = None,
    ) -> int:
        return int(await self.db.fetchval(
            """INSERT INTO signals (ts, account_id, strategy, symbol, direction,
                   strength, ref_price, stop_price, regime, ml_confidence,
                   accepted, reject_code, reject_detail, sized_qty, meta)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::market_regime,$10,$11,$12,$13,
                       $14,$15::jsonb)
               RETURNING id""",
            ts, account_id, strategy, symbol, direction, Decimal(str(strength)),
            ref_price, stop_price, regime,
            Decimal(str(ml_confidence)) if ml_confidence is not None else None,
            accepted, reject_code, reject_detail, sized_qty, _json(meta or {})))

    async def rejection_counts(self, account_id: int, days: int = 7,
                               ) -> dict[str, int]:
        rows = await self.db.fetch(
            """SELECT reject_code, count(*) n FROM signals
               WHERE account_id=$1 AND NOT accepted
                 AND ts > now() - ($2 || ' days')::interval
               GROUP BY 1 ORDER BY 2 DESC""", account_id, str(days))
        return {r["reject_code"]: int(r["n"]) for r in rows}


class RiskEventRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(self, account_id: int | None, severity: str, code: str,
                     detail: str = "", payload: dict[str, Any] | None = None,
                     ) -> int:
        return int(await self.db.fetchval(
            """INSERT INTO risk_events (account_id, severity, code, detail, payload)
               VALUES ($1,$2,$3,$4,$5::jsonb) RETURNING id""",
            account_id, severity, code, detail, _json(payload or {})))

    async def unresolved(self, account_id: int | None = None,
                         ) -> list[asyncpg.Record]:
        return await self.db.fetch(
            """SELECT * FROM risk_events
               WHERE resolved_at IS NULL AND ($1::int IS NULL OR account_id = $1)
               ORDER BY ts DESC""", account_id)

    async def resolve(self, event_id: int) -> None:
        await self.db.execute(
            "UPDATE risk_events SET resolved_at = now() WHERE id = $1", event_id)


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str)


def fill_from_record(r: asyncpg.Record) -> Fill:
    """Rebuild a domain Fill from a joined fills/orders row."""
    return Fill(order_id=str(r["order_id"]), ts=r["ts"], symbol=r["symbol"],
                side=Side(r["side"]), qty=r["qty"], price=r["price"],
                fee=r["fee"], is_maker=bool(r["is_maker"]),
                exchange_fill_id=r["exchange_fill_id"],
                strategy=r["strategy"] or "")


__all__ = [
    "EquityRepository",
    "OrderRepository",
    "PositionRepository",
    "RiskEventRepository",
    "SignalRepository",
    "fill_from_record",
]
