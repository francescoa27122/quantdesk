"""Startup reconciliation — the check that must pass before anything trades.

**The engine boots halted and stays halted until this passes clean.**
Disagreeing with the exchange about what you own is the most expensive bug class
in this domain: every risk gate downstream is computed from a position number,
so a wrong position silently invalidates all of them at once.

Three independent sources are compared:

  A. `positions` — the projection the engine maintains
  B. **replay of `fills`** — recomputed from the ledger, sharing no code with A
  C. the exchange's own balances

A vs B catches a bug in our bookkeeping. B vs C catches a fill we never saw, or
one the exchange forgot. Two sources would tell you *that* something is wrong;
three tell you *which side* is wrong, which is the difference between a
five-minute fix and an afternoon.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from ct.models import Position, Side
from ct.store.db import Database
from ct.store.repositories import OrderRepository, PositionRepository

log = logging.getLogger(__name__)

# Exchanges round; so does float somewhere upstream of every API. Below this a
# difference is noise, not a discrepancy. Above it, refuse to trade.
DUST = Decimal("0.00000001")


@dataclass(frozen=True, slots=True)
class Discrepancy:
    symbol: str
    source_a: str
    source_b: str
    qty_a: Decimal
    qty_b: Decimal

    @property
    def delta(self) -> Decimal:
        return self.qty_a - self.qty_b

    def __str__(self) -> str:
        return (f"{self.symbol}: {self.source_a}={self.qty_a} vs "
                f"{self.source_b}={self.qty_b} (delta {self.delta})")


@dataclass
class ReconciliationResult:
    ok: bool
    checked_symbols: int = 0
    discrepancies: list[Discrepancy] = field(default_factory=list)
    positions_vs_fills: dict[str, Decimal] = field(default_factory=dict)
    fills_vs_exchange: dict[str, Decimal] = field(default_factory=dict)
    exchange_available: bool = True

    def summary(self) -> str:
        if self.ok:
            base = (f"reconciliation OK — {self.checked_symbols} symbols agree "
                    f"across positions, fill replay"
                    + (" and the exchange" if self.exchange_available else ""))
            return base if self.exchange_available else (
                base + "; EXCHANGE NOT CHECKED (balances unavailable)")
        lines = [f"reconciliation FAILED — {len(self.discrepancies)} discrepancies:"]
        lines += [f"  {d}" for d in self.discrepancies]
        return "\n".join(lines)


class ReconciliationService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.orders = OrderRepository(db)
        self.positions = PositionRepository(db)

    async def replay_fills(self, account_id: int) -> dict[str, Decimal]:
        """Net quantity per symbol, recomputed from the fill ledger alone.

        Deliberately does NOT reuse `Position.apply_fill` or the SQL in
        `_apply_to_book`: an independent reconstruction is the entire point. If
        it shared code, agreement would prove only that the same bug ran twice.
        """
        net: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
        for r in await self.orders.fills_for(account_id):
            sign = 1 if Side(r["side"]) is Side.BUY else -1
            net[r["symbol"]] += r["qty"] * sign
        return {s: q for s, q in net.items() if abs(q) > DUST}

    async def run(
        self, account_id: int,
        exchange_balances: dict[str, Decimal] | None = None,
    ) -> ReconciliationResult:
        """Compare all three sources. `exchange_balances` may be None when the
        venue is unreachable — that is reported, never assumed clean."""
        stored = await self.positions.net_by_symbol(account_id)
        replayed = await self.replay_fills(account_id)

        symbols = set(stored) | set(replayed)
        if exchange_balances is not None:
            symbols |= {s for s, q in exchange_balances.items() if abs(q) > DUST}

        discrepancies: list[Discrepancy] = []
        a_vs_b: dict[str, Decimal] = {}
        b_vs_c: dict[str, Decimal] = {}

        for sym in sorted(symbols):
            a = stored.get(sym, Decimal(0))
            b = replayed.get(sym, Decimal(0))
            if abs(a - b) > DUST:
                discrepancies.append(Discrepancy(sym, "positions", "fill_replay", a, b))
                a_vs_b[sym] = a - b
            if exchange_balances is not None:
                c = exchange_balances.get(sym, Decimal(0))
                if abs(b - c) > DUST:
                    discrepancies.append(
                        Discrepancy(sym, "fill_replay", "exchange", b, c))
                    b_vs_c[sym] = b - c

        result = ReconciliationResult(
            ok=not discrepancies, checked_symbols=len(symbols),
            discrepancies=discrepancies, positions_vs_fills=a_vs_b,
            fills_vs_exchange=b_vs_c,
            exchange_available=exchange_balances is not None)

        if result.ok:
            log.info(result.summary())
        else:
            log.error(result.summary())
        return result

    async def repair_positions_from_fills(self, account_id: int) -> int:
        """Rebuild `positions` from the fill ledger. Returns rows rewritten.

        Only ever run deliberately, never automatically: if positions and fills
        disagree, WHICH is wrong is a judgement. Fills are append-only and
        exchange-sourced, so they are usually right — but "usually" is not a
        basis for silently overwriting the state that governs live risk. The
        engine reports the mismatch and halts; a human runs this.
        """
        rows = await self.orders.fills_for(account_id)
        books: dict[tuple[str, str], Position] = {}
        for r in sorted(rows, key=lambda x: (x["ts"], x["id"])):
            key = (r["strategy"] or "", r["symbol"])
            pos = books.setdefault(
                key, Position(symbol=r["symbol"], strategy=key[0]))
            from ct.store.repositories import fill_from_record
            pos.apply_fill(fill_from_record(r))

        written = 0
        async with self.db.transaction() as conn:
            await conn.execute(
                "DELETE FROM positions WHERE account_id = $1", account_id)
            for (strategy, symbol), pos in books.items():
                await conn.execute(
                    """INSERT INTO positions (account_id, strategy, symbol, qty,
                           avg_entry, realized_pnl, fees_paid, opened_at, updated_at)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,now())""",
                    account_id, strategy, symbol, pos.qty, pos.avg_entry,
                    pos.realized_pnl, pos.fees_paid, pos.opened_at)
                written += 1
        log.warning("rebuilt %d position rows for account %s from the fill ledger",
                    written, account_id)
        return written

    async def orphan_orders(self, account_id: int) -> list[str]:
        """Non-terminal orders from a previous life.

        Each must be re-queried at the exchange by `client_order_id` before any
        new order is placed. Blind resubmission after a restart is how one
        intent becomes two positions.
        """
        rows = await self.orders.open_orders(account_id)
        return [r["client_order_id"] for r in rows]


__all__ = ["DUST", "Discrepancy", "ReconciliationResult", "ReconciliationService"]
