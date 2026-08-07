"""Portfolio backtester — N symbols × M strategies over ONE shared account.

## Why this exists

`BacktestEngine` runs one strategy on one symbol with one position, and that
made four risk gates dead code in every backtest ever run here: gross exposure,
per-symbol exposure, correlation clusters, and max concurrent positions. They
were written, unit-tested in isolation, and **never once executed against real
price data** — because a single-symbol account can't breach a portfolio cap.

It also makes the allocator impossible to validate: an allocator arbitrates
between strategies competing for capital, and until now there was no simulation
in which two strategies could compete.

## The two decisions that shape it

**1. Books, not positions.** Every (strategy, symbol) pair gets its own book
(`AccountState.book_key`). Two strategies both long BTC hold two independent
books, so a close from one cannot flatten the other. Risk still reasons in
symbol terms — exposure and correlation caps aggregate every book on a symbol,
because that is what the exchange holds. This is the data-structure-level
answer to "never let strategies fight over capital."

**2. One clock, merged.** Symbols do not share a bar timeline: one may be
missing bars the other has. The engine walks the **union** of bar timestamps and
gives each strategy only the bars for its own symbol that have closed at or
before the current instant. A symbol with no bar at time T is simply not
consulted at T — it is never forward-filled, because a forward-filled bar is a
price that did not happen and every gate downstream would treat it as real.

Look-ahead prevention is unchanged and still structural: `Context` arrays are
slices that physically end at the current bar.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np

from ct.backtest.engine import CostModel
from ct.backtest.metrics import PerformanceReport, build_report
from ct.events import (
    EventBus,
    NullEventBus,
    new_correlation_id,
    order_filled,
    order_submitted,
    portfolio_updated,
    position_closed,
    position_opened,
    risk_approved,
    signal_generated,
    signal_rejected,
)
from ct.models import (
    AccountState,
    Bar,
    ClosedTrade,
    ExitReason,
    Fill,
    Order,
    OrderType,
    Position,
    Regime,
    Side,
    Signal,
    TimeInForce,
    replace,
)
from ct.risk.account import AccountStateBuilder
from ct.risk.allocator import Allocator, Weights
from ct.risk.engine import Decision, RiskEngine, RiskLimits
from ct.risk.sizing import annualize_vol, ewma_volatility
from ct.strategies.base import Context, Strategy
from ct.strategies.features import FeatureSet
from ct.strategies.indicators import atr

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Allocation:
    """One strategy's mandate: which symbols, at what portfolio weight.

    `weight` feeds the volatility-target sizer, so it scales risk rather than
    notional — halving a weight halves the risk contribution, not the dollars.
    """
    strategy: Strategy
    symbols: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError(f"{self.strategy.name}: no symbols allocated")
        if self.weight < 0:
            raise ValueError(f"{self.strategy.name}: negative weight")


@dataclass
class _Leg:
    strategy: str
    symbol: str
    direction: int
    qty: Decimal
    entry_ts: datetime
    entry_price: Decimal
    entry_index: int
    correlation_id: str
    fees: Decimal = Decimal(0)
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class PortfolioResult:
    report: PerformanceReport
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    signals: list[tuple[Signal, Decision]] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    per_strategy: dict[str, PerformanceReport] = field(default_factory=dict)
    # Mark-to-market peak. This CAN legitimately exceed the gross cap: caps
    # bind when an order is approved, and a held position that appreciates
    # carries exposure up with it. The system does not force-deleverage on
    # appreciation — that would sell winners to satisfy a bookkeeping limit.
    peak_gross_exposure: float = 0.0
    # Committed exposure (filled + in flight) measured at the instant of every
    # approval. THIS is the number the gross cap actually governs, and it must
    # never exceed it.
    peak_committed_exposure: float = 0.0
    peak_open_books: int = 0

    def rejection_summary(self) -> str:
        if not self.rejections:
            return "no signals were rejected"
        rows = sorted(self.rejections.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{code}x{n}" for code, n in rows)

    def attribution(self) -> str:
        lines = [f"{'strategy':24s} {'trades':>6} {'net':>10} {'sharpe':>7} {'maxDD':>7}"]
        for name, rep in sorted(self.per_strategy.items(),
                                key=lambda kv: -kv[1].net_profit):
            lines.append(f"{name:24s} {rep.n_trades:6d} {rep.net_profit:10,.0f} "
                         f"{rep.sharpe:7.2f} {rep.max_drawdown:6.1%}")
        return "\n".join(lines)


class PortfolioBacktestEngine:
    """Multi-symbol, multi-strategy, one account, one risk engine."""

    def __init__(
        self,
        allocations: Sequence[Allocation],
        limits: RiskLimits | None = None,
        costs: CostModel | None = None,
        *,
        initial_equity: float = 10_000.0,
        timeframe: str = "4h",
        vol_lookback: int = 30,
        atr_period: int = 14,
        maker_ttl_bars: int = 3,
        market_ttl_bars: int = 3,
        armed: bool = True,
        correlation_clusters: dict[str, list[str]] | None = None,
        bus: EventBus | None = None,
        allocator: Allocator | None = None,
        weights: Weights | None = None,
    ) -> None:
        if not allocations:
            raise ValueError("no allocations given")
        self.allocations = list(allocations)
        self.limits = limits or RiskLimits()
        self.costs = costs or CostModel()
        self.initial_equity = Decimal(str(initial_equity))
        self.timeframe = timeframe
        self.vol_lookback = vol_lookback
        self.atr_period = atr_period
        self.maker_ttl_bars = maker_ttl_bars
        self.market_ttl_bars = market_ttl_bars
        self.armed = armed
        # Correlated groups (rho > 0.8) get a joint cap. Without this, "5 alt
        # positions" is one bet wearing five hats.
        self.correlation_clusters = correlation_clusters or {}
        # NullEventBus by default: a portfolio backtest emits millions of events
        # and nobody wants them on disk. Same type as the real bus so there is
        # exactly one code path.
        self.bus = bus if bus is not None else NullEventBus(source="backtest")
        self.risk = RiskEngine(self.limits)
        # Arbitrates signals competing for the same capacity within one pass.
        self.allocator = allocator or Allocator()
        # Static weights for now: per-strategy return streams do not exist
        # until a run has finished, so a run cannot weight by its own output
        # without looking ahead. Feed last run's trades to `Allocator.weights`
        # and pass the result in.
        self._weights = weights

    # ── main loop ────────────────────────────────────────────────────────

    def run(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        *,
        regimes: dict[str, list[Regime]] | None = None,
        n_trials: int = 1,
    ) -> PortfolioResult:
        symbols = self._validate(bars_by_symbol)
        series = {s: _SymbolSeries(bars_by_symbol[s], self.atr_period)
                  for s in symbols}

        # Union of every symbol's bar timestamps, in order. A symbol with no bar
        # at time T is skipped at T rather than forward-filled — a forward-filled
        # bar is a price that never traded, and every gate downstream would
        # treat it as real.
        clock: list[datetime] = sorted({b.ts for s in symbols
                                        for b in bars_by_symbol[s]})

        acct = AccountStateBuilder(self.initial_equity, self.limits)
        legs: dict[str, _Leg] = {}
        pending: list[tuple[Order, int, str]] = []   # (order, ttl, correlation_id)

        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        trades: list[ClosedTrade] = []
        signal_log: list[tuple[Signal, Decision]] = []
        rejections: dict[str, int] = {}
        bars_in_market = 0
        peak_gross = 0.0
        peak_books = 0
        self._peak_committed = 0.0
        marks: dict[str, Decimal] = {}

        warmup_by_alloc = {a.strategy.name: a.strategy.warmup for a in self.allocations}
        max_warmup = max(warmup_by_alloc.values())

        for tick, now in enumerate(clock):
            live = {s: series[s].at(now) for s in symbols}
            live = {s: i for s, i in live.items() if i is not None}
            for s, idx in live.items():
                marks[s] = Decimal(str(series[s].close[idx]))

            # ── 1. fills for orders raised earlier ───────────────────────
            still: list[tuple[Order, int, str]] = []
            for order, ttl, cor in pending:
                idx = live.get(order.symbol)
                if idx is None:                       # symbol had no bar here
                    still.append((order, ttl, cor))
                    continue
                bar = bars_by_symbol[order.symbol][idx]
                fill = self._simulate_fill(order, bar)
                if fill is None:
                    if ttl > 1:
                        still.append((order, ttl - 1, cor))
                    else:
                        acct.release(order.client_order_id)   # expired unfilled
                    continue

                # A fill converts part of the reservation into a real position.
                acct.release(order.client_order_id, fill.qty * fill.price)
                book = acct.position(order.strategy, order.symbol)
                realized = acct.apply_fill(fill)
                self.bus.publish(order_filled(
                    fill.order_id, fill.symbol, fill.side.value, fill.qty,
                    fill.price, fill.fee, fill.is_maker, correlation_id=cor))

                leg_key = f"{order.strategy}|{order.symbol}"
                leg = legs.get(leg_key)
                if leg is None and not book.is_flat:
                    legs[leg_key] = _Leg(order.strategy, order.symbol,
                                         book.direction, abs(book.qty), bar.ts,
                                         book.avg_entry, tick, cor, fill.fee)
                    book.stop_price = order.attached_stop
                    book.target_price = order.attached_target
                    book.trail_atr_mult = order.trail_atr_mult
                    book.highest_close = book.lowest_close = bar.close
                    self.bus.publish(position_opened(
                        order.symbol, book.qty, book.avg_entry, order.strategy,
                        correlation_id=cor))
                elif leg is not None:
                    leg.fees += fill.fee
                    if book.is_flat:
                        reason = (ExitReason(order.meta["reason_enum"])
                                  if "reason_enum" in order.meta else ExitReason.SIGNAL)
                        self._settle(legs, leg_key, fill.price, realized, reason,
                                     tick, bar.ts, trades, acct)

                remainder = order.qty - fill.qty
                if remainder > 0 and ttl > 1:
                    still.append((replace(order, qty=remainder), ttl - 1, cor))
                elif remainder > 0:
                    acct.release(order.client_order_id)
            pending = still

            # ── 2. protective exits, per book ────────────────────────────
            for leg_key, book in list(acct.positions.items()):
                if book.is_flat:
                    continue
                idx = live.get(book.symbol)
                if idx is None:
                    continue
                bar = bars_by_symbol[book.symbol][idx]
                bars_in_market += 1
                leg = legs.get(leg_key)
                self._update_excursions(leg, book, bar)
                cur_atr = series[book.symbol].atr_at(idx)
                exit_order = self._check_protective_exits(book, bar, cur_atr)
                if exit_order is None:
                    continue
                px = exit_order.limit_price or Decimal(str(bar.close))
                qty = abs(book.qty)
                fee = self.costs.fee(qty * px, is_maker=False)
                fill = Fill(exit_order.client_order_id, bar.ts, book.symbol,
                            exit_order.side, qty, px, fee, is_maker=False,
                            strategy=book.strategy)
                realized = acct.apply_fill(fill)
                cor = leg.correlation_id if leg else new_correlation_id()
                self.bus.publish(order_filled(
                    fill.order_id, fill.symbol, fill.side.value, fill.qty,
                    fill.price, fill.fee, False, correlation_id=cor))
                if leg is not None:
                    leg.fees += fee
                    self._settle(legs, leg_key, px, realized,
                                 ExitReason(exit_order.meta["reason_enum"]),
                                 tick, bar.ts, trades, acct)

            # ── 3. mark the whole portfolio once ─────────────────────────
            equity = acct.mark(now, marks)
            equity_curve.append(float(equity))
            timestamps.append(now)

            snapshot = acct.snapshot(
                now, marks, correlation_clusters=self.correlation_clusters,
                open_order_count=len(pending))
            peak_gross = max(peak_gross, float(snapshot.gross_exposure()))
            peak_books = max(peak_books, snapshot.open_position_count())
            self.bus.publish(portfolio_updated(
                equity, acct.cash, float(snapshot.gross_exposure()),
                float(snapshot.drawdown()), snapshot.open_position_count()))

            if tick < max_warmup or tick >= len(clock) - 1:
                continue

            # ── 4. every strategy proposes on every symbol it owns ───────
            pending_signals: list[tuple[Any, ...]] = []
            for alloc in self.allocations:
                strat = alloc.strategy
                for symbol in alloc.symbols:
                    idx = live.get(symbol)
                    if idx is None or idx < strat.warmup:
                        continue
                    ser = series[symbol]
                    ctx = Context(
                        symbol=symbol, timeframe=self.timeframe, index=idx, ts=now,
                        open=ser.open[:idx + 1], high=ser.high[:idx + 1],
                        low=ser.low[:idx + 1], close=ser.close[:idx + 1],
                        volume=ser.volume[:idx + 1],
                        position=acct.position(strat.name, symbol),
                        f=ser.features.view(idx),
                        regime=(regimes[symbol][idx]
                                if regimes and symbol in regimes else Regime.UNKNOWN),
                        params=strat.params,
                    )
                    try:
                        proposals = strat.on_bar(ctx)
                    except Exception:
                        # Quarantine the strategy-symbol pair, not the run.
                        rejections["STRATEGY_ERROR"] = \
                            rejections.get("STRATEGY_ERROR", 0) + 1
                        log.exception("%s raised on %s", strat.name, symbol)
                        continue

                    for sig in proposals:
                        pending_signals.append((sig, alloc, ser, idx))

            # Arbitration: scarce capacity goes to the strongest signal in the
            # pass, not to whichever strategy happens to be listed first. Caps
            # bind on committed exposure, so without this the backtest result
            # depended on the ORDER OF `self.allocations` — an artefact of how
            # the list was typed rather than anything about the market.
            for sig, alloc, ser, idx in self.allocator.rank(
                    pending_signals, weights=self._weights,
                    key=lambda item: item[0]):
                        # Fresh snapshot per signal: reservations made by an
                        # earlier signal in this same pass must be visible to
                        # the next, or every cap is checked against a stale
                        # zero-exposure view and breached N-fold.
                        snap = acct.snapshot(
                            now, marks,
                            correlation_clusters=self.correlation_clusters)
                        self._consider(sig, alloc, acct, snap, ser, idx,
                                       legs, pending, signal_log, rejections)

        # ── close the books ──────────────────────────────────────────────
        # S2-3: metrics start at `max_warmup`, the first tick any strategy may
        # trade (line ~344 gates on exactly this). The flat warm-up prefix
        # otherwise depresses return mean/stdev and inflates n_periods and
        # exposure's denominator. Benchmark cut to the same window; the full
        # curve stays in `PortfolioResult` for plotting.
        metrics_start = min(max_warmup, len(equity_curve))
        report = build_report(
            equity_curve[metrics_start:], trades, self.timeframe,
            initial_equity=float(self.initial_equity),
            total_fees=float(acct.total_fees),
            bars_in_market=min(bars_in_market, len(clock)),
            n_trials=n_trials,
            benchmark_equity=self._benchmark(bars_by_symbol,
                                             clock[metrics_start:]),
        )
        return PortfolioResult(
            report, equity_curve, timestamps, trades, signal_log, rejections,
            per_strategy=self._attribute(trades, equity_curve),
            peak_gross_exposure=peak_gross,
            peak_committed_exposure=self._peak_committed,
            peak_open_books=peak_books)

    # ── helpers ──────────────────────────────────────────────────────────

    def _consider(self, sig: Signal, alloc: Allocation,
                  acct: AccountStateBuilder, snapshot: AccountState,
                  ser: _SymbolSeries, idx: int, legs: dict[str, _Leg],
                  pending: list[tuple[Order, int, str]],
                  signal_log: list[tuple[Signal, Decision]],
                  rejections: dict[str, int]) -> None:
        cor = (legs[f"{sig.strategy}|{sig.symbol}"].correlation_id
               if f"{sig.strategy}|{sig.symbol}" in legs else new_correlation_id())
        sig_event = self.bus.publish(signal_generated(
            sig.strategy, sig.symbol, sig.direction, sig.strength, cor,
            ref_price=sig.ref_price))

        decision = self.risk.evaluate(
            sig, snapshot,
            realized_ann_vol=self._realized_ann_vol(ser.close, idx),
            strategy_weight=alloc.weight, armed=self.armed)
        signal_log.append((sig, decision))

        if not decision.approved:
            code = decision.code.value if decision.code else "UNKNOWN"
            rejections[code] = rejections.get(code, 0) + 1
            self.bus.publish(signal_rejected(sig.strategy, sig.symbol, code,
                                             decision.detail, cor,
                                             causation_id=sig_event.seq))
            return

        order = decision.order
        assert order is not None
        if sig.direction == 0:
            order = replace(order, meta={**order.meta,
                                         "reason_enum": ExitReason.SIGNAL.value})
        approved = self.bus.publish(risk_approved(
            sig.strategy, sig.symbol, order.qty,
            decision.diagnostics.get("binding_sizer", "?"), cor,
            causation_id=sig_event.seq))
        self.bus.publish(order_submitted(
            order.client_order_id, order.symbol, order.side.value, order.qty,
            order.type.value, order.limit_price, cor, causation_id=approved.seq))

        acct.record_order(snapshot.ts)
        notional = order.notional(snapshot.marks.get(order.symbol, Decimal(0)))
        if snapshot.equity > 0:
            committed_after = float(
                snapshot.committed_gross_exposure() + notional / snapshot.equity)
            self._peak_committed = max(self._peak_committed, committed_after)
        acct.reserve(order.client_order_id, order.strategy, order.symbol, notional)
        pending.append((order, self._order_ttl(order), cor))

    def _settle(self, legs: dict[str, _Leg], leg_key: str,
                exit_price: Decimal, realized: Decimal, reason: ExitReason,
                tick: int, ts: datetime, trades: list[ClosedTrade],
                acct: AccountStateBuilder) -> None:
        leg = legs.pop(leg_key, None)
        if leg is None:
            return
        trade = ClosedTrade(
            strategy=leg.strategy, symbol=leg.symbol, direction=leg.direction,
            qty=leg.qty, entry_ts=leg.entry_ts, exit_ts=ts,
            entry_price=leg.entry_price, exit_price=exit_price,
            gross_pnl=realized + leg.fees, fees=leg.fees, exit_reason=reason,
            bars_held=tick - leg.entry_index, mae=leg.mae, mfe=leg.mfe)
        trades.append(trade)
        acct.record_trade_result(leg.strategy, trade.net_pnl, ts)
        self.bus.publish(position_closed(
            leg.symbol, leg.strategy, trade.net_pnl, trade.return_pct,
            reason.value, correlation_id=leg.correlation_id))

    def _validate(self, bars_by_symbol: dict[str, list[Bar]]) -> list[str]:
        wanted = {s for a in self.allocations for s in a.symbols}
        missing = wanted - set(bars_by_symbol)
        if missing:
            raise ValueError(f"no bars supplied for allocated symbols: "
                             f"{sorted(missing)}")
        for sym in sorted(wanted):
            need = max(a.strategy.warmup for a in self.allocations if sym in a.symbols)
            if len(bars_by_symbol[sym]) < need + 2:
                raise ValueError(
                    f"{sym}: {len(bars_by_symbol[sym])} bars is fewer than the "
                    f"deepest allocated strategy's warmup ({need}) + 2; the "
                    f"result would be an artefact")
        return sorted(wanted)

    def _order_ttl(self, order: Order) -> int:
        if order.type is OrderType.MARKET:
            return max(1, self.market_ttl_bars)
        if order.tif in (TimeInForce.IOC, TimeInForce.FOK):
            return 1
        return max(1, self.maker_ttl_bars)

    def _simulate_fill(self, order: Order, bar: Bar) -> Fill | None:
        cm = self.costs
        max_qty = Decimal(str(bar.dollar_volume * cm.max_bar_volume_pct)) / \
            Decimal(str(bar.open or bar.close or 1.0))
        qty = min(order.qty, max_qty) if max_qty > 0 else Decimal(0)
        if qty <= 0:
            return None

        if order.type is OrderType.MARKET:
            base = Decimal(str(bar.open))
            slip = Decimal(str(cm.slippage_bps(float(qty * base),
                                               bar.dollar_volume) / 10_000.0))
            price = base * (Decimal(1) + slip * Decimal(order.side.sign))
            return Fill(order.client_order_id, bar.ts, order.symbol, order.side,
                        qty, price, cm.fee(qty * price, is_maker=False),
                        is_maker=False, strategy=order.strategy)

        limit = order.limit_price
        if limit is None:
            return None
        touched = (order.side is Side.BUY and Decimal(str(bar.low)) <= limit) or \
                  (order.side is Side.SELL and Decimal(str(bar.high)) >= limit)
        if not touched:
            return None
        if order.tif is TimeInForce.POST_ONLY:
            crossed = (order.side is Side.BUY and Decimal(str(bar.open)) < limit) or \
                      (order.side is Side.SELL and Decimal(str(bar.open)) > limit)
            if crossed:
                return None
        return Fill(order.client_order_id, bar.ts, order.symbol, order.side, qty,
                    limit, cm.fee(qty * limit, is_maker=True), is_maker=True,
                    strategy=order.strategy)

    def _check_protective_exits(self, pos: Position, bar: Bar,
                                cur_atr: float) -> Order | None:
        long = pos.qty > 0
        if pos.trail_atr_mult and cur_atr > 0:
            offset = Decimal(str(pos.trail_atr_mult)) * Decimal(str(cur_atr))
            if long and pos.highest_close is not None:
                cand = Decimal(str(pos.highest_close)) - offset
                if pos.stop_price is None or cand > pos.stop_price:
                    pos.stop_price = cand
            elif not long and pos.lowest_close is not None:
                cand = Decimal(str(pos.lowest_close)) + offset
                if pos.stop_price is None or cand < pos.stop_price:
                    pos.stop_price = cand

        for level, reason in ((pos.stop_price,
                               ExitReason.TRAIL if pos.trail_atr_mult else ExitReason.STOP),
                              (pos.target_price, ExitReason.TARGET)):
            if level is None:
                continue
            if reason is ExitReason.TARGET:
                hit = ((long and Decimal(str(bar.high)) >= level)
                       or (not long and Decimal(str(bar.low)) <= level))
            else:
                hit = ((long and Decimal(str(bar.low)) <= level)
                       or (not long and Decimal(str(bar.high)) >= level))
            if hit:
                return Order(symbol=pos.symbol,
                             side=Side.SELL if long else Side.BUY,
                             type=OrderType.MARKET, qty=abs(pos.qty),
                             limit_price=level, reduce_only=True,
                             strategy=pos.strategy,
                             meta={"reason_enum": reason.value})
        return None

    @staticmethod
    def _update_excursions(leg: _Leg | None, pos: Position, bar: Bar) -> None:
        if leg is None or pos.avg_entry == 0:
            return
        entry = float(pos.avg_entry)
        if pos.qty > 0:
            pos.highest_close = max(pos.highest_close or bar.close, bar.close)
            leg.mfe = max(leg.mfe, (bar.high - entry) / entry)
            leg.mae = min(leg.mae, (bar.low - entry) / entry)
        else:
            pos.lowest_close = min(pos.lowest_close or bar.close, bar.close)
            leg.mfe = max(leg.mfe, (entry - bar.low) / entry)
            leg.mae = min(leg.mae, (entry - bar.high) / entry)

    def _realized_ann_vol(self, close: np.ndarray, i: int) -> float:
        lo = max(1, i - self.vol_lookback)
        window = close[lo:i + 1]
        if len(window) < 3:
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(np.log(np.maximum(window, 1e-12)))
        return annualize_vol(ewma_volatility(rets.tolist(),
                                             halflife=self.vol_lookback),
                             self.timeframe)

    def _benchmark(self, bars_by_symbol: dict[str, list[Bar]],
                   clock: list[datetime]) -> list[float]:
        """Equal-weight buy-and-hold of every traded symbol, net of one taker
        round trip. Beating a single asset is easy in its good year; beating the
        basket you chose to trade is the honest bar."""
        symbols = sorted(bars_by_symbol)
        if not symbols or not clock:
            return []
        fee = self.costs.taker_fee_bps / 10_000.0
        eq0 = float(self.initial_equity)
        per = eq0 * (1.0 - fee) / len(symbols)
        series = {s: _SymbolSeries(bars_by_symbol[s], self.atr_period)
                  for s in symbols}
        # Buy at the price prevailing at the WINDOW's first tick, not at bar 0
        # of the data: when the caller hands a post-warmup clock (S2-3), an
        # entry at the old price would smuggle the warm-up period's drift into
        # the benchmark while the strategy is forbidden from trading it.
        units = {}
        for s in symbols:
            i = series[s].at(clock[0])
            idx = i if i is not None else series[s].last_index_at(clock[0])
            first = series[s].close[idx] if idx is not None else series[s].close[0]
            units[s] = per / first if first > 0 else 0.0
        curve = [eq0]
        for now in clock[1:]:
            total = 0.0
            for s in symbols:
                i = series[s].at(now)
                idx = i if i is not None else series[s].last_index_at(now)
                total += units[s] * (series[s].close[idx] if idx is not None
                                     else series[s].close[0])
            curve.append(total)
        if len(curve) > 1:
            curve[-1] *= (1.0 - fee)
        return curve

    def _attribute(self, trades: list[ClosedTrade],
                   equity_curve: list[float]) -> dict[str, PerformanceReport]:
        """Per-strategy P&L from its own closed trades.

        Deliberately NOT a per-strategy equity curve: strategies share one cash
        balance, so splitting equity between them would require an allocation
        rule that does not exist yet. Trade-level attribution is exact; anything
        finer would be invented.
        """
        out: dict[str, PerformanceReport] = {}
        by_strategy: dict[str, list[ClosedTrade]] = {}
        for t in trades:
            by_strategy.setdefault(t.strategy, []).append(t)
        for name, ts in by_strategy.items():
            eq = [float(self.initial_equity)]
            for t in sorted(ts, key=lambda x: x.exit_ts):
                eq.append(eq[-1] + float(t.net_pnl))
            out[name] = build_report(
                eq, ts, self.timeframe,
                initial_equity=float(self.initial_equity),
                total_fees=float(sum(t.fees for t in ts)),
                bars_in_market=sum(t.bars_held for t in ts),
                n_trials=1)
        for alloc in self.allocations:
            out.setdefault(alloc.strategy.name, build_report(
                [float(self.initial_equity)], [], self.timeframe,
                initial_equity=float(self.initial_equity)))
        return out


class _SymbolSeries:
    """Numpy views of one symbol's bars, plus a timestamp index."""

    __slots__ = (
        "_atr",
        "_index",
        "_ts",
        "close",
        "features",
        "high",
        "low",
        "open",
        "volume",
    )

    def __init__(self, bars: list[Bar], atr_period: int) -> None:
        self.open = np.array([b.open for b in bars], dtype=np.float64)
        self.high = np.array([b.high for b in bars], dtype=np.float64)
        self.low = np.array([b.low for b in bars], dtype=np.float64)
        self.close = np.array([b.close for b in bars], dtype=np.float64)
        self.volume = np.array([b.volume for b in bars], dtype=np.float64)
        # Computed once over the whole series: every indicator here is causal,
        # so atr(full)[i] == atr(full[:i+1])[i].
        self._atr = atr(self.high, self.low, self.close, atr_period)
        # One indicator cache per symbol, shared by every strategy allocated to
        # it — so two strategies wanting adx(14) on BTC/USD compute it once
        # between them, not once each per bar.
        self.features = FeatureSet(self.open, self.high, self.low, self.close,
                                   self.volume)
        self._ts = [b.ts for b in bars]
        self._index = {b.ts: i for i, b in enumerate(bars)}

    def at(self, ts: datetime) -> int | None:
        """Index of the bar closing exactly at `ts`, or None. Never interpolates."""
        return self._index.get(ts)

    def last_index_at(self, ts: datetime) -> int | None:
        """Most recent bar at or before `ts` — for marking only, never signals."""
        import bisect
        i = bisect.bisect_right(self._ts, ts) - 1
        return i if i >= 0 else None

    def atr_at(self, i: int) -> float:
        v = self._atr[i]
        return float(v) if np.isfinite(v) else 0.0


__all__ = ["Allocation", "PortfolioBacktestEngine", "PortfolioResult"]
