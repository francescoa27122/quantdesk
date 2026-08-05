"""Broker interface + the paper implementation.

Three brokers implement this protocol and the engine cannot tell them apart
(ARCHITECTURE.md §4):

    BacktestBroker  — inside ct/backtest/engine.py, fills against historical bars
    PaperBroker     — here; fills against LIVE book snapshots, real spreads
    CcxtBroker      — Phase 6; real orders, and the ONLY class that can lose money

PaperBroker is not a toy. It fills against the same live quotes the live broker
would trade on, applies the same fee schedule, and models queue position for
post-only orders. A paper result that diverges from the eventual live result is
a bug in this file, and that is deliberate: it concentrates the simulation risk
in one small, readable place instead of spreading it across the engine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from ct.models import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Quote,
    Side,
    TimeInForce,
    utcnow,
)

log = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """Recoverable exchange error — the engine's circuit breaker counts these."""


class InsufficientFunds(BrokerError):
    pass


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    maker_bps: float = 25.0
    taker_bps: float = 40.0
    tier_volume_30d: float = 0.0

    def fee(self, notional: Decimal, is_maker: bool) -> Decimal:
        bps = self.maker_bps if is_maker else self.taker_bps
        return notional * Decimal(str(bps / 10_000.0))


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    client_order_id: str
    status: OrderStatus
    filled_qty: Decimal = Decimal(0)
    avg_price: Decimal | None = None
    error: str | None = None


class Broker(ABC):
    """Everything the engine is allowed to ask an exchange to do.

    Note what is absent: no `withdraw`, no `transfer`, no `deposit`. The
    interface cannot express moving money off the exchange, so no bug and no
    prompt can cause it. The API key itself is also created without withdrawal
    permission (ARCHITECTURE.md §9) — defence in depth, both layers cheap.
    """

    @abstractmethod
    async def submit(self, order: Order) -> OrderUpdate: ...

    @abstractmethod
    async def cancel(self, client_order_id: str) -> OrderUpdate: ...

    @abstractmethod
    async def fetch_order(self, client_order_id: str) -> OrderUpdate: ...

    @abstractmethod
    async def open_orders(self) -> list[OrderUpdate]: ...

    @abstractmethod
    async def balances(self) -> dict[str, Decimal]: ...

    @abstractmethod
    async def quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    async def fee_schedule(self) -> FeeSchedule: ...

    @property
    @abstractmethod
    def can_lose_money(self) -> bool:
        """True only for the live broker. The engine logs this at every boot so
        'I thought it was on paper' is impossible to say afterwards."""


@dataclass
class _RestingOrder:
    order: Order
    placed_at: object
    filled: Decimal = Decimal(0)
    # Simulated queue position: how much size sat ahead of us at our price when
    # we joined. Ignoring this is what makes naive paper trading over-optimistic
    # — it assumes you are always first in the queue.
    queue_ahead: Decimal = Decimal(0)


class PaperBroker(Broker):
    """Simulated fills against live quotes.

    Fill rules:
      * market / IOC  → immediate at the far touch + modelled slippage
      * post-only     → rests; fills only after the market trades THROUGH our
                        price (not merely to it) by `through_ticks`, and only
                        after `queue_ahead` has been consumed
      * unfilled post-only orders expire after `maker_timeout`
    """

    def __init__(
        self,
        starting_cash: Decimal = Decimal(10_000),
        quote_ccy: str = "USD",
        fees: FeeSchedule | None = None,
        *,
        maker_timeout: timedelta = timedelta(minutes=15),
        latency_ms: int = 250,
        through_ticks: float = 0.0005,   # 5 bps beyond our price to count a fill
    ) -> None:
        self.cash: dict[str, Decimal] = {quote_ccy: starting_cash}
        self.quote_ccy = quote_ccy
        self.fees = fees or FeeSchedule()
        self.maker_timeout = maker_timeout
        self.latency_ms = latency_ms
        self.through_ticks = through_ticks
        self._resting: dict[str, _RestingOrder] = {}
        self._terminal: dict[str, OrderUpdate] = {}
        self._quotes: dict[str, Quote] = {}
        self.fills: list[Fill] = []

    # ── the collector feeds this ─────────────────────────────────────────
    def on_quote(self, q: Quote) -> list[Fill]:
        """Update the book and return any fills it triggered."""
        self._quotes[q.symbol] = q
        return self._match(q)

    async def quote(self, symbol: str) -> Quote:
        q = self._quotes.get(symbol)
        if q is None:
            raise BrokerError(f"no quote for {symbol}; the collector is behind")
        return q

    async def fee_schedule(self) -> FeeSchedule:
        return self.fees

    @property
    def can_lose_money(self) -> bool:
        return False

    async def submit(self, order: Order) -> OrderUpdate:
        q = self._quotes.get(order.symbol)
        if q is None:
            return OrderUpdate(order.client_order_id, OrderStatus.REJECTED,
                               error=f"no quote for {order.symbol}")

        if order.type is OrderType.MARKET or order.tif is TimeInForce.IOC:
            price = Decimal(str(q.ask if order.side is Side.BUY else q.bid))
            # Latency cost: by the time we arrive the touch has moved. Charged
            # as a fraction of the spread, scaled by our latency.
            drift = Decimal(str(q.spread_bps / 10_000.0)) * \
                Decimal(str(min(1.0, self.latency_ms / 1000.0))) * \
                Decimal(order.side.sign)
            price *= (Decimal(1) + drift)
            upd, _fill = self._execute(order, order.qty, price, is_maker=False)
            return upd

        if order.limit_price is None:
            return OrderUpdate(order.client_order_id, OrderStatus.REJECTED,
                               error="limit order without a price")

        if order.tif is TimeInForce.POST_ONLY:
            crosses = (order.side is Side.BUY and order.limit_price >= Decimal(str(q.ask))) \
                or (order.side is Side.SELL and order.limit_price <= Decimal(str(q.bid)))
            if crosses:
                # A real post-only is rejected, not silently converted. If this
                # ever gets "helpfully" turned into a taker fill, the maker
                # rebate assumption in every backtest becomes a lie.
                return OrderUpdate(order.client_order_id, OrderStatus.REJECTED,
                                   error="post_only would cross the spread")

        ahead = Decimal(str(q.bid_size if order.side is Side.BUY else q.ask_size))
        self._resting[order.client_order_id] = _RestingOrder(order, utcnow(),
                                                             queue_ahead=ahead)
        return OrderUpdate(order.client_order_id, OrderStatus.OPEN)

    async def cancel(self, client_order_id: str) -> OrderUpdate:
        resting = self._resting.pop(client_order_id, None)
        if resting is None:
            return self._terminal.get(
                client_order_id,
                OrderUpdate(client_order_id, OrderStatus.REJECTED, error="unknown order"))
        upd = OrderUpdate(client_order_id, OrderStatus.CANCELLED, resting.filled)
        self._terminal[client_order_id] = upd
        return upd

    async def fetch_order(self, client_order_id: str) -> OrderUpdate:
        if client_order_id in self._terminal:
            return self._terminal[client_order_id]
        r = self._resting.get(client_order_id)
        if r is None:
            return OrderUpdate(client_order_id, OrderStatus.REJECTED, error="unknown order")
        status = OrderStatus.PARTIAL if r.filled > 0 else OrderStatus.OPEN
        return OrderUpdate(client_order_id, status, r.filled)

    async def open_orders(self) -> list[OrderUpdate]:
        return [OrderUpdate(cid, OrderStatus.PARTIAL if r.filled > 0 else OrderStatus.OPEN,
                            r.filled) for cid, r in self._resting.items()]

    async def balances(self) -> dict[str, Decimal]:
        return dict(self.cash)

    # ── internals ────────────────────────────────────────────────────────

    def _match(self, q: Quote) -> list[Fill]:
        out: list[Fill] = []
        now = utcnow()
        for cid, r in list(self._resting.items()):
            o = r.order
            if o.symbol != q.symbol:
                continue
            if now - r.placed_at > self.maker_timeout:  # type: ignore[operator]
                self._resting.pop(cid, None)
                self._terminal[cid] = OrderUpdate(cid, OrderStatus.EXPIRED, r.filled)
                continue

            limit = o.limit_price
            assert limit is not None
            thr = Decimal(str(self.through_ticks))
            traded_through = (
                (o.side is Side.BUY and Decimal(str(q.ask)) < limit * (Decimal(1) - thr))
                or (o.side is Side.SELL and Decimal(str(q.bid)) > limit * (Decimal(1) + thr))
            )
            if not traded_through:
                continue
            # Consume the queue ahead of us before we get any size.
            traded = Decimal(str(q.bid_size if o.side is Side.BUY else q.ask_size))
            if r.queue_ahead > 0:
                r.queue_ahead = max(Decimal(0), r.queue_ahead - traded)
                continue

            remaining = o.qty - r.filled
            fill_qty = min(remaining, traded) if traded > 0 else remaining
            if fill_qty <= 0:
                continue

            upd, fill = self._execute(o, fill_qty, limit, is_maker=True)

            # A rejected execution is NOT a fill. Advancing `r.filled` or
            # emitting here was the phantom-fill bug: on repeated quotes an
            # unfundable resting order reported 30 units filled when 6 were, and
            # re-emitted the same Fill object every time. `fill is None` is now
            # the single authority on whether anything happened.
            if fill is None:
                # PaperBroker does not reserve funds at placement (real venues
                # do), so an unfundable resting order can only be discovered
                # here. Kill it rather than retry forever — a live exchange
                # would have rejected it at placement, and a silently immortal
                # order is worse than a loudly dead one.
                self._resting.pop(cid, None)
                self._terminal[cid] = OrderUpdate(
                    cid, OrderStatus.REJECTED, r.filled, error=upd.error)
                log.warning("paper order %s killed at fill time: %s", cid, upd.error)
                continue

            r.filled += fill.qty
            out.append(fill)
            if r.filled >= o.qty:
                self._resting.pop(cid, None)
                self._terminal[cid] = OrderUpdate(cid, OrderStatus.FILLED, r.filled, limit)
        return out

    def _execute(self, order: Order, qty: Decimal, price: Decimal,
                 *, is_maker: bool) -> tuple[OrderUpdate, Fill | None]:
        """Attempt an execution.

        Returns `(update, fill)`. **`fill is None` iff nothing was executed** —
        callers must branch on that and never infer success from the update or
        from `self.fills`. Returning the Fill explicitly instead of letting
        callers reach for `self.fills[-1]` is what makes the phantom-fill bug
        unrepresentable rather than merely fixed.
        """
        notional = qty * price
        fee = self.fees.fee(notional, is_maker)
        if order.side is Side.BUY:
            need = notional + fee
            have = self.cash.get(self.quote_ccy, Decimal(0))
            if need > have:
                return OrderUpdate(
                    order.client_order_id, OrderStatus.REJECTED,
                    error=f"insufficient {self.quote_ccy}: "
                          f"need {need:.2f}, have {have:.2f}"), None
            self.cash[self.quote_ccy] = have - need
        else:
            self.cash[self.quote_ccy] = self.cash.get(self.quote_ccy, Decimal(0)) \
                + notional - fee

        # `strategy` MUST be carried through. Without it the fill lands in the
        # book keyed "|SYMBOL" instead of "STRATEGY|SYMBOL", so the position
        # persists under a key the engine never looks up — and a restart
        # silently loses track of an open position while reconciliation still
        # passes, because that nets by symbol. Found running the engine end to
        # end against live prices, not by any unit test.
        fill = Fill(order.client_order_id, utcnow(), order.symbol, order.side,
                    qty, price, fee, is_maker, strategy=order.strategy)
        self.fills.append(fill)
        status = OrderStatus.FILLED if qty >= order.qty else OrderStatus.PARTIAL
        upd = OrderUpdate(order.client_order_id, status, qty, price)
        if status is OrderStatus.FILLED:
            self._terminal[order.client_order_id] = upd
        log.info("paper fill %s %s %s @ %s (%s) fee %s", order.symbol,
                 order.side.value, qty, price, "maker" if is_maker else "taker", fee)
        return upd, fill


__all__ = [
    "Broker",
    "BrokerError",
    "FeeSchedule",
    "InsufficientFunds",
    "OrderUpdate",
    "PaperBroker",
]
