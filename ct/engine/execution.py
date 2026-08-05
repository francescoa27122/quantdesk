"""Order execution — the bridge between an approved Order and a broker.

Sits between `RiskEngine.evaluate` and `Broker.submit`, and owns the three
things that make restart-safety real:

1. **Persist before submitting.** The order row is written *first*, so a crash
   between "sent to exchange" and "recorded locally" leaves a record we can
   reconcile from. The reverse order leaves an order at the venue that this
   system has never heard of — the worst possible state.
2. **Idempotency via `client_order_id`.** Generated before the call, UNIQUE in
   the database. A retry, a restart replay, or a duplicate exchange delivery
   are all no-ops rather than second positions.
3. **Every fill is one transaction** — fill row, order aggregate, position
   book, and the notification announcing it. "Traded but never told you" is not
   a race that can happen here.

The broker is injected. `PaperBroker` and (later) `CcxtBroker` are
indistinguishable from this module's point of view, which is the same
one-code-path discipline as the backtester.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from ct.events import (
    EventBus,
    order_filled,
    order_rejected,
    order_submitted,
    position_closed,
    position_opened,
)
from ct.execution.broker import Broker, OrderUpdate
from ct.models import Fill, Order, OrderStatus, Position
from ct.store.repositories import OrderRepository, PositionRepository

log = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(
        self,
        broker: Broker,
        orders: OrderRepository,
        positions: PositionRepository,
        bus: EventBus,
        *,
        account_id: int,
        notify_channel: str = "telegram",
    ) -> None:
        self.broker = broker
        self.orders = orders
        self.positions = positions
        self.bus = bus
        self.account_id = account_id
        self.notify_channel = notify_channel
        self.submitted = 0
        self.rejected = 0
        self.fills_recorded = 0
        self.duplicate_fills = 0
        self.consecutive_broker_failures = 0

    async def submit(self, order: Order, *, correlation_id: str,
                     causation_id: int | None = None) -> OrderUpdate:
        """Persist, then submit. Never the other way round.

        A crash after `broker.submit()` but before the row exists leaves an
        order live at the exchange that this system has no record of — nothing
        downstream can reconcile it, because nothing knows to look.
        """
        await self.orders.insert_order(self.account_id, order)
        ev = self.bus.publish(order_submitted(
            order.client_order_id, order.symbol, order.side.value, order.qty,
            order.type.value, order.limit_price, correlation_id,
            causation_id=causation_id))

        try:
            update = await self.broker.submit(order)
        except Exception as exc:
            # The order row stays `pending`. Startup reconciliation will
            # re-query it by client_order_id rather than assume it never
            # happened — "we don't know" is not "it didn't".
            self.consecutive_broker_failures += 1
            log.exception("broker.submit failed for %s", order.client_order_id)
            self.bus.publish(order_rejected(
                order.client_order_id, order.symbol, repr(exc),
                correlation_id=correlation_id))
            return OrderUpdate(order.client_order_id, OrderStatus.PENDING,
                               error=repr(exc))

        self.consecutive_broker_failures = 0
        if update.status is OrderStatus.REJECTED:
            self.rejected += 1
            await self.orders.mark_terminal(
                order.client_order_id, OrderStatus.REJECTED,
                error_code="broker_rejected", error_detail=update.error)
            self.bus.publish(order_rejected(
                order.client_order_id, order.symbol,
                update.error or "rejected", correlation_id=correlation_id))
            return update

        self.submitted += 1
        await self.orders.mark_submitted(order.client_order_id, None)
        _ = ev
        return update

    async def record_fill(self, fill: Fill, *, correlation_id: str | None = None,
                          notify: bool = True) -> bool:
        """Persist a fill and everything derived from it, atomically.

        Returns False for a duplicate — which is a normal outcome, not an
        error. Exchanges re-deliver, and a restart replays.
        """
        note = None
        if notify:
            side = fill.side.value.upper()
            note = (self.notify_channel,
                    f"Filled {side} {fill.symbol}",
                    f"{side} {fill.qty} {fill.symbol} @ {fill.price} "
                    f"(fee {fill.fee}, {'maker' if fill.is_maker else 'taker'})")

        before = await self._book_qty(fill.strategy, fill.symbol)
        inserted = await self.orders.record_fill(self.account_id, fill,
                                                 notification=note)
        if not inserted:
            self.duplicate_fills += 1
            return False

        self.fills_recorded += 1
        self.bus.publish(order_filled(
            fill.order_id, fill.symbol, fill.side.value, fill.qty, fill.price,
            fill.fee, fill.is_maker, correlation_id=correlation_id))

        after = await self._book_qty(fill.strategy, fill.symbol)
        if before == 0 and after != 0:
            book = await self._book(fill.strategy, fill.symbol)
            self.bus.publish(position_opened(
                fill.symbol, after, book.avg_entry if book else fill.price,
                fill.strategy, correlation_id=correlation_id))
        elif before != 0 and after == 0:
            book = await self._book(fill.strategy, fill.symbol)
            realized = book.realized_pnl if book else Decimal(0)
            self.bus.publish(position_closed(
                fill.symbol, fill.strategy, realized, 0.0, "signal",
                correlation_id=correlation_id))
        return True

    async def poll_fills(self, *, correlation_ids: dict[str, str] | None = None,
                         ) -> int:
        """Drain fills the broker has produced since the last poll.

        `PaperBroker` accumulates them in `.fills`; a live broker would use
        `watchMyTrades`. Both arrive here, so the recording path is identical.
        """
        raw = getattr(self.broker, "fills", None)
        if not raw:
            return 0
        pending, self.broker.fills = list(raw), []      # type: ignore[attr-defined]
        n = 0
        for fill in pending:
            cid = (correlation_ids or {}).get(fill.order_id)
            if await self.record_fill(fill, correlation_id=cid):
                n += 1
        return n

    async def cancel_all_open(self) -> int:
        """Shutdown path: cancel resting orders so nothing executes unattended.

        Positions are deliberately NOT flattened — a container restart is not a
        reason to market-sell the book. The resting stops remain the protection
        and the engine re-arms them on boot.
        """
        cancelled = 0
        for row in await self.orders.open_orders(self.account_id):
            cid = row["client_order_id"]
            try:
                update = await self.broker.cancel(cid)
            except Exception:
                log.exception("cancel failed for %s", cid)
                continue
            if update.status is OrderStatus.CANCELLED:
                await self.orders.mark_terminal(cid, OrderStatus.CANCELLED)
                cancelled += 1
        return cancelled

    async def reconcile_open_orders(self) -> dict[str, str]:
        """Re-query every non-terminal order at the exchange BEFORE trading.

        Blind resubmission after a restart is how one intent becomes two
        positions. Anything the venue does not recognise is marked terminal so
        it cannot linger as a phantom.
        """
        outcome: dict[str, str] = {}
        for row in await self.orders.open_orders(self.account_id):
            cid = row["client_order_id"]
            try:
                update = await self.broker.fetch_order(cid)
            except Exception as exc:
                outcome[cid] = f"unreachable: {exc!r}"
                continue
            outcome[cid] = update.status.value
            if update.status.is_terminal:
                await self.orders.mark_terminal(cid, update.status)
        return outcome

    # ── helpers ──────────────────────────────────────────────────────────

    async def _book(self, strategy: str, symbol: str) -> Position | None:
        books = await self.positions.load(self.account_id)
        return books.get(f"{strategy}|{symbol}")

    async def _book_qty(self, strategy: str, symbol: str) -> Decimal:
        book = await self._book(strategy, symbol)
        return book.qty if book else Decimal(0)


__all__ = ["OrderExecutor"]
