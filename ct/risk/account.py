"""`AccountStateBuilder` — the one place account bookkeeping happens.

**Why this file exists.** The backtester used to construct `AccountState` by hand,
and the live engine was going to construct it again, differently. That produced a
whole class of silent bug: the backtest models a *different system* than the one
that will trade. Five instances were found before this file existed —

  * the consecutive-loss breaker never expired in backtest (lifetime ban) while
    production's `paused_until` did;
  * `orders_last_hour` was hardcoded to 0, so the rate limiter was never exercised;
  * `positions` was a single-symbol dict, so gross-exposure, correlation-cluster
    and max-position gates were dead code in every backtest;
  * `armed` was hardcoded True;
  * high-water mark and day/week rollover were open-coded in the run loop.

Each was invisible, and each silently invalidated performance numbers. So the rule
now is: **nobody constructs `AccountState` directly — they drive this builder.**
The backtester, the paper engine and the live engine all use it, which makes a
backtest/live divergence a change to one file rather than an emergent property of
two.

It is multi-symbol from the start even though the current backtester runs one
symbol, so the portfolio backtester (Phase 2 item 3) extends this rather than
replacing it.

No I/O, no clock of its own — the caller supplies `ts`. That is what makes it
usable identically under a simulated clock and a real one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from ct.models import AccountState, Fill, PendingOrder, Position
from ct.risk.engine import RiskLimits

# Orders are counted against the rate limiter over this trailing window.
RATE_WINDOW = timedelta(hours=1)


class AccountStateBuilder:
    """Owns cash, positions, high-water mark, streaks and rate counters.

    Drive it in this order per bar:
        1. `apply_fill(...)`        for each fill
        2. `record_trade_result(…)` when a round trip closes
        3. `mark(ts, marks)`        once — returns equity, handles rollovers
        4. `snapshot(ts, marks)`    to hand to `RiskEngine.evaluate`
    """

    def __init__(
        self,
        initial_equity: Decimal,
        limits: RiskLimits,
        *,
        quote_ccy: str = "USD",
    ) -> None:
        self.limits = limits
        self.quote_ccy = quote_ccy
        self.cash: Decimal = initial_equity
        self.positions: dict[str, Position] = {}
        self.high_water_mark: Decimal = initial_equity
        self.equity: Decimal = initial_equity
        self.total_fees: Decimal = Decimal(0)

        self.day_start_equity: Decimal = initial_equity
        self.week_start_equity: Decimal = initial_equity
        self._cur_day: object | None = None
        self._cur_week: object | None = None

        self._order_ts: list[datetime] = []
        self._pending: dict[str, PendingOrder] = {}
        self._consecutive_losses: dict[str, int] = {}
        self._breaker_until: dict[str, datetime] = {}

        self.halted: bool = False
        self.halt_reason: str = ""

    # ── position & cash bookkeeping ──────────────────────────────────────

    def position(self, strategy: str, symbol: str) -> Position:
        """The book for one (strategy, symbol) pair, created on first touch.

        Keyed per book rather than per symbol so two strategies trading the
        same symbol cannot flatten each other — see `AccountState.book_key`.
        """
        key = AccountState.book_key(strategy, symbol)
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(symbol=symbol, strategy=strategy)
            self.positions[key] = pos
        return pos

    def apply_fill(self, fill: Fill) -> Decimal:
        """Apply a fill to cash and the owning book. Returns realized P&L.

        Cash is debited by `side.sign * notional`, so equity is always
        `cash + Σ(qty × mark)` — correct for shorts too (negative qty, cash
        credited on entry).
        """
        pos = self.position(fill.strategy, fill.symbol)
        realized = pos.apply_fill(fill)
        self.cash -= Decimal(fill.side.sign) * fill.notional
        self.cash -= fill.fee
        self.total_fees += fill.fee
        return realized

    def record_order(self, ts: datetime) -> None:
        """Count a submitted order against the rate limiter."""
        self._order_ts.append(ts)

    # ── in-flight reservations ───────────────────────────────────────────
    # An approved order consumes exposure budget from the moment it is
    # approved, not from the moment it fills. See `models.PendingOrder`.

    def reserve(self, client_order_id: str, strategy: str, symbol: str,
                notional: Decimal) -> None:
        self._pending[client_order_id] = PendingOrder(strategy, symbol, notional)

    def release(self, client_order_id: str,
                filled_notional: Decimal | None = None) -> None:
        """Free a reservation. With `filled_notional`, reduce it by that much
        (a partial fill converts part of the reservation into a real position);
        without, drop it entirely (cancel or expiry)."""
        cur = self._pending.get(client_order_id)
        if cur is None:
            return
        if filled_notional is None:
            del self._pending[client_order_id]
            return
        remaining = cur.notional - filled_notional
        if remaining <= 0:
            del self._pending[client_order_id]
        else:
            self._pending[client_order_id] = PendingOrder(
                cur.strategy, cur.symbol, remaining)

    def pending_orders(self) -> tuple[PendingOrder, ...]:
        return tuple(self._pending.values())

    def record_trade_result(self, strategy: str, net_pnl: Decimal, ts: datetime) -> None:
        """Update the consecutive-loss streak and arm the breaker if tripped."""
        if net_pnl > 0:
            self._consecutive_losses[strategy] = 0
            self._breaker_until.pop(strategy, None)
            return
        n = self._consecutive_losses.get(strategy, 0) + 1
        self._consecutive_losses[strategy] = n
        if n >= self.limits.max_consecutive_losses:
            # Wall-clock, not bar count: the same rule then means the same thing
            # on 1h and 1d, and it matches production's `strategies.paused_until`.
            self._breaker_until[strategy] = ts + timedelta(
                hours=self.limits.loss_pause_hours)

    # ── per-bar maintenance ──────────────────────────────────────────────

    def mark(self, ts: datetime, marks: dict[str, Decimal]) -> Decimal:
        """Mark to market; roll day/week; expire breakers; prune rate window.

        Call exactly once per bar, after fills and before `snapshot`.
        """
        self._expire_breakers(ts)
        self._prune_orders(ts)

        value = Decimal(0)
        for pos in self.positions.values():
            if pos.is_flat:
                continue
            mark = marks.get(pos.symbol)
            if mark is None:
                # No price for an open position: fall back to entry rather than
                # silently valuing it at zero, and let the risk engine's
                # NO_MARK_PRICE gate refuse to trade it.
                mark = pos.avg_entry
            value += pos.qty * mark

        self.equity = self.cash + value
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        day = ts.date()
        if self._cur_day is None or day != self._cur_day:
            self._cur_day, self.day_start_equity = day, self.equity

        week = ts.isocalendar()[:2]
        if self._cur_week is None or week != self._cur_week:
            self._cur_week, self.week_start_equity = week, self.equity

        return self.equity

    def _expire_breakers(self, ts: datetime) -> None:
        for strategy, until in list(self._breaker_until.items()):
            if ts >= until:
                self._breaker_until.pop(strategy, None)
                self._consecutive_losses[strategy] = 0

    def _prune_orders(self, ts: datetime) -> None:
        cutoff = ts - RATE_WINDOW
        if self._order_ts and self._order_ts[0] < cutoff:
            self._order_ts = [t for t in self._order_ts if t >= cutoff]

    def orders_last_hour(self, ts: datetime) -> int:
        cutoff = ts - RATE_WINDOW
        return sum(1 for t in self._order_ts if t >= cutoff)

    # ── the snapshot handed to the risk engine ───────────────────────────

    def snapshot(
        self,
        ts: datetime,
        marks: dict[str, Decimal],
        *,
        data_age_bars: dict[str, int] | None = None,
        correlation_clusters: dict[str, list[str]] | None = None,
        open_order_count: int = 0,
    ) -> AccountState:
        return AccountState(
            ts=ts,
            equity=self.equity,
            cash=self.cash,
            high_water_mark=self.high_water_mark,
            positions=self.positions,
            marks=marks,
            day_start_equity=self.day_start_equity,
            week_start_equity=self.week_start_equity,
            open_order_count=open_order_count or len(self._pending),
            pending=self.pending_orders(),
            orders_last_hour=self.orders_last_hour(ts),
            consecutive_losses=dict(self._consecutive_losses),
            halted=self.halted,
            halt_reason=self.halt_reason,
            data_age_bars=data_age_bars or {},
            correlation_clusters=correlation_clusters or {},
        )

    def open_position_count(self) -> int:
        return sum(1 for p in self.positions.values() if not p.is_flat)


__all__ = ["RATE_WINDOW", "AccountStateBuilder"]
