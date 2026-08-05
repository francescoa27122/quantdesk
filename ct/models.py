"""Domain models shared by the backtester, the paper broker and the live engine.

These are deliberately plain frozen dataclasses with no I/O and no DB coupling:
the same objects flow through all three brokers (ARCHITECTURE.md §4), and the
risk engine must be testable without a database or an exchange.

Money is `Decimal` everywhere it touches an order, a fill or a balance. Prices
and quantities in indicator/statistics code are float — that is fine, those are
estimates. It is not fine for anything the exchange will echo back to you.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

# ── Enums ────────────────────────────────────────────────────────────────

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    OCO = "oco"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    POST_ONLY = "post_only"   # maker-only; rejected if it would cross. §0.1


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                        OrderStatus.REJECTED, OrderStatus.EXPIRED)


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


class ExitReason(str, Enum):
    SIGNAL = "signal"
    STOP = "stop"
    TRAIL = "trail"
    TARGET = "target"
    FLATTEN = "flatten"
    RISK = "risk"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Market data ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Bar:
    """A CLOSED candle. `ts` is the bar's OPEN time in UTC.

    Strategies only ever receive closed bars — that is the structural half of
    look-ahead prevention (ARCHITECTURE.md §4).
    """
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def dollar_volume(self) -> float:
        """Used by the slippage model's participation cap."""
        return self.volume * self.typical


@dataclass(frozen=True, slots=True)
class Quote:
    ts: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return 0.0 if mid <= 0 else (self.ask - self.bid) / mid * 10_000.0


# ── Strategy output ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Signal:
    """What a strategy emits. Note what is NOT here: quantity.

    Strategies express *intent and conviction*; only the risk engine decides
    size. That separation is what makes it impossible for a strategy bug to
    place an oversized order.
    """
    ts: datetime
    strategy: str
    symbol: str
    direction: int                    # +1 long, -1 short, 0 = exit/flat
    strength: float = 1.0             # [0, 1] conviction, scales size
    ref_price: float = 0.0            # close of the bar that produced it
    stop_price: float | None = None
    target_price: float | None = None
    trail_atr_mult: float | None = None
    atr: float | None = None          # required for risk-budget sizing
    regime: Regime = Regime.UNKNOWN
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0 or 1, got {self.direction}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0,1], got {self.strength}")


# ── Orders, fills, positions ─────────────────────────────────────────────

def new_client_order_id(prefix: str = "qd") -> str:
    """Idempotency key, generated BEFORE the API call.

    On restart the engine re-queries every non-terminal order by this id
    instead of blindly resubmitting (ARCHITECTURE.md §6.3).
    """
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    side: Side
    type: OrderType
    qty: Decimal
    client_order_id: str = field(default_factory=new_client_order_id)
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    tif: TimeInForce = TimeInForce.GTC
    strategy: str = ""
    reduce_only: bool = False
    attached_stop: Decimal | None = None
    attached_target: Decimal | None = None
    trail_atr_mult: float | None = None
    created_at: datetime = field(default_factory=utcnow)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"order qty must be > 0, got {self.qty}")
        if self.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.type.value} order requires limit_price")
        if self.type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.type.value} order requires stop_price")

    def notional(self, mark: Decimal) -> Decimal:
        return self.qty * (self.limit_price if self.limit_price is not None else mark)


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str                  # client_order_id
    ts: datetime
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    fee: Decimal = Decimal(0)
    is_maker: bool = False
    exchange_fill_id: str | None = None
    # Which strategy's book this fill belongs to. Required once more than one
    # strategy can trade the same symbol — without it, applying a fill would
    # have to guess whose position it changed.
    strategy: str = ""

    @property
    def notional(self) -> Decimal:
        return self.qty * self.price


@dataclass(slots=True)
class Position:
    """Mutable — the engine owns exactly one of these per (account, symbol).

    Invariant: replaying every Fill for a symbol through `apply_fill` in
    timestamp order must reproduce this object. The reconciliation job checks
    that on every boot (ARCHITECTURE.md §6.3).
    """
    symbol: str
    qty: Decimal = Decimal(0)               # signed; positive = long
    avg_entry: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    fees_paid: Decimal = Decimal(0)
    stop_price: Decimal | None = None
    target_price: Decimal | None = None
    trail_atr_mult: float | None = None
    highest_close: float | None = None
    lowest_close: float | None = None
    opened_at: datetime | None = None
    strategy: str = ""

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def direction(self) -> int:
        return 0 if self.qty == 0 else (1 if self.qty > 0 else -1)

    def notional(self, mark: Decimal) -> Decimal:
        return abs(self.qty) * mark

    def unrealized_pnl(self, mark: Decimal) -> Decimal:
        if self.qty == 0:
            return Decimal(0)
        return (mark - self.avg_entry) * self.qty

    def apply_fill(self, fill: Fill) -> Decimal:
        """Apply a fill; return the realized P&L of this fill alone.

        Handles the four cases explicitly — open, add, reduce, and flip —
        because the flip case (a fill larger than the open position, reversing
        it) is the one that silently corrupts avg_entry in naive
        implementations.
        """
        self.fees_paid += fill.fee
        signed = fill.qty * fill.side.sign
        realized = Decimal(0)

        if self.qty == 0:
            # Open.
            self.qty = signed
            self.avg_entry = fill.price
            self.opened_at = fill.ts
        elif (self.qty > 0) == (signed > 0):
            # Add to the same side: weighted-average the entry.
            total = self.qty + signed
            self.avg_entry = (self.avg_entry * self.qty + fill.price * signed) / total
            self.qty = total
        else:
            closing = min(abs(signed), abs(self.qty))
            realized = (fill.price - self.avg_entry) * Decimal(self.direction) * closing
            remaining = self.qty + signed
            if remaining == 0:
                # Flat.
                self.qty = Decimal(0)
                self.avg_entry = Decimal(0)
                self.opened_at = None
                self.stop_price = self.target_price = None
                self.highest_close = self.lowest_close = None
            elif (remaining > 0) == (self.qty > 0):
                # Partial reduce: avg_entry is unchanged.
                self.qty = remaining
            else:
                # Flip: the residual is a brand-new position at the fill price.
                self.qty = remaining
                self.avg_entry = fill.price
                self.opened_at = fill.ts
                self.stop_price = self.target_price = None
                self.highest_close = self.lowest_close = None

        self.realized_pnl += realized - fill.fee
        return realized


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    strategy: str
    symbol: str
    direction: int
    qty: Decimal
    entry_ts: datetime
    exit_ts: datetime
    entry_price: Decimal
    exit_price: Decimal
    gross_pnl: Decimal
    fees: Decimal
    exit_reason: ExitReason = ExitReason.SIGNAL
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0

    @property
    def net_pnl(self) -> Decimal:
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> float:
        cost = abs(self.entry_price * self.qty)
        return 0.0 if cost == 0 else float(self.net_pnl / cost)

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


# ── Account snapshot handed to the risk engine ───────────────────────────

@dataclass(frozen=True, slots=True)
class PendingOrder:
    """An order approved but not yet filled.

    Exposure caps MUST count these. Without them, N signals evaluated in one
    engine pass each see the same zero-exposure snapshot, each passes the cap
    independently, and together they breach it by N×. Measured: a 30% gross cap
    reached 50% with six symbols, and a 12% cluster cap reached 50%.

    This is not a backtest artifact — the live engine evaluates a batch of
    signals per pass too, so the same breach would happen with real money.
    """
    strategy: str
    symbol: str
    notional: Decimal


@dataclass(frozen=True, slots=True)
class AccountState:
    """Everything the risk engine is allowed to know. Pure data, no DB handle.

    Passing a snapshot rather than a repository is what lets the entire gate
    stack be property-tested without infrastructure (tests/test_risk.py).
    """
    ts: datetime
    equity: Decimal
    cash: Decimal
    high_water_mark: Decimal
    positions: dict[str, Position] = field(default_factory=dict)
    marks: dict[str, Decimal] = field(default_factory=dict)
    day_start_equity: Decimal = Decimal(0)
    week_start_equity: Decimal = Decimal(0)
    open_order_count: int = 0
    # Approved-but-unfilled orders. Every exposure cap counts these; see
    # `PendingOrder` for what happens when they are ignored.
    pending: tuple[PendingOrder, ...] = ()
    orders_last_hour: int = 0
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""
    data_age_bars: dict[str, int] = field(default_factory=dict)
    # ρ>0.8 clusters, e.g. {"majors": ["BTC/USD", "ETH/USD"]}
    correlation_clusters: dict[str, list[str]] = field(default_factory=dict)

    # ── book addressing ──────────────────────────────────────────────────
    # `positions` is keyed by BOOK, not by symbol: "<strategy>|<symbol>". With
    # more than one strategy trading a symbol, a symbol-keyed dict forces them
    # to share one position — so a close from strategy A would flatten strategy
    # B's trade. That is exactly the "strategies fighting over capital" failure
    # the allocator exists to prevent, and it has to be impossible at the data
    # structure level, not merely discouraged.
    #
    # Risk still reasons in SYMBOL terms, because the exchange does: exposure
    # and correlation caps aggregate every book on a symbol.

    @staticmethod
    def book_key(strategy: str, symbol: str) -> str:
        return f"{strategy}|{symbol}"

    def position_for(self, strategy: str, symbol: str) -> Position | None:
        return self.positions.get(self.book_key(strategy, symbol))

    def books_for_symbol(self, symbol: str) -> list[Position]:
        return [p for p in self.positions.values()
                if p.symbol == symbol and not p.is_flat]

    def _mark(self, pos: Position) -> Decimal:
        return self.marks.get(pos.symbol, pos.avg_entry)

    def gross_exposure(self) -> Decimal:
        if self.equity <= 0:
            return Decimal(0)
        total = sum((p.notional(self._mark(p))
                     for p in self.positions.values() if not p.is_flat),
                    Decimal(0))
        return total / self.equity

    def symbol_exposure(self, symbol: str) -> Decimal:
        """Netted across every strategy's book on this symbol — the exposure
        the exchange actually sees."""
        if self.equity <= 0:
            return Decimal(0)
        total = sum((p.notional(self._mark(p)) for p in self.books_for_symbol(symbol)),
                    Decimal(0))
        return total / self.equity

    # ── committed = filled + in flight. What every cap must be checked against.
    def pending_notional(self, symbol: str | None = None) -> Decimal:
        return sum((p.notional for p in self.pending
                    if symbol is None or p.symbol == symbol), Decimal(0))

    def committed_symbol_exposure(self, symbol: str) -> Decimal:
        if self.equity <= 0:
            return Decimal(0)
        return self.symbol_exposure(symbol) + self.pending_notional(symbol) / self.equity

    def committed_gross_exposure(self) -> Decimal:
        if self.equity <= 0:
            return Decimal(0)
        return self.gross_exposure() + self.pending_notional() / self.equity

    def open_position_count(self) -> int:
        """Books that are open OR have an order in flight. A pending order is a
        position slot already spent — counting only filled books lets N orders
        past a max-positions cap of 1."""
        books = {self.book_key(p.strategy, p.symbol)
                 for p in self.positions.values() if not p.is_flat}
        books |= {self.book_key(p.strategy, p.symbol) for p in self.pending}
        return len(books)

    def holds_book(self, strategy: str, symbol: str) -> bool:
        pos = self.position_for(strategy, symbol)
        if pos is not None and not pos.is_flat:
            return True
        return any(p.strategy == strategy and p.symbol == symbol
                   for p in self.pending)

    def drawdown(self) -> Decimal:
        if self.high_water_mark <= 0:
            return Decimal(0)
        return max(Decimal(0), (self.high_water_mark - self.equity) / self.high_water_mark)

    def day_drawdown(self) -> Decimal:
        if self.day_start_equity <= 0:
            return Decimal(0)
        return max(Decimal(0), (self.day_start_equity - self.equity) / self.day_start_equity)

    def week_drawdown(self) -> Decimal:
        if self.week_start_equity <= 0:
            return Decimal(0)
        return max(Decimal(0), (self.week_start_equity - self.equity) / self.week_start_equity)


__all__ = [
    "AccountState",
    "Bar",
    "ClosedTrade",
    "ExitReason",
    "Fill",
    "Order",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
    "Regime",
    "Side",
    "Signal",
    "TimeInForce",
    "new_client_order_id",
    "replace",
    "utcnow",
]
