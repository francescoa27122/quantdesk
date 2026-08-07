"""Event-driven backtest engine.

Deliberately NOT vectorised. A shift-and-multiply backtester is ~100× faster
and structurally invites look-ahead: one off-by-one on a `.shift()` silently
produces a 5-Sharpe equity curve and you will believe it. The N100 runs 8 years
of 4h bars across the whole strategy book in well under a minute here. Speed is
not the constraint; being wrong is.

The two rules that make this trustworthy:

  1. A strategy sees bar `t` only after it has closed, and receives array
     slices that physically end at `t`.
  2. Every order produced at bar `t` fills at bar `t+1`, never at `t`'s close.

Both are asserted by `tests/test_backtest.py::test_no_lookahead`.

The same RiskEngine instance that runs live sits between the strategy and the
broker here, so "the backtest was sized differently from production" cannot
happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import numpy as np

from ct.backtest.metrics import PerformanceReport, build_report
from ct.models import (
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
from ct.risk.engine import Decision, RiskEngine, RiskLimits
from ct.risk.sizing import annualize_vol, ewma_volatility
from ct.strategies.base import Context, Strategy
from ct.strategies.features import FeatureSet
from ct.strategies.indicators import atr


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fees and slippage. Defaults are Kraken's $0-tier — see ARCHITECTURE §0.1.

    `impact_k` drives a square-root market-impact term. It is intentionally
    conservative until calibrated from `book_snapshots`; an uncalibrated
    slippage model should over-charge, never under-charge.
    """
    maker_fee_bps: float = 25.0
    taker_fee_bps: float = 40.0
    half_spread_bps: float = 2.0
    impact_k: float = 0.10
    max_bar_volume_pct: float = 0.01   # cannot trade more than 1% of a bar

    def fee(self, notional: Decimal, is_maker: bool) -> Decimal:
        bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        return notional * Decimal(str(bps / 10_000.0))

    def slippage_bps(self, notional: float, bar_dollar_volume: float) -> float:
        """half-spread + k * sqrt(participation), in bps."""
        if bar_dollar_volume <= 0:
            return self.half_spread_bps + 50.0   # unknown liquidity: punish it
        participation = min(1.0, notional / bar_dollar_volume)
        # float(): np.sqrt returns np.floating, so the expression is Any to
        # mypy. The conversion is lossless and keeps numpy scalars out of the
        # cost model, which is compared and serialised as plain floats.
        return float(self.half_spread_bps
                     + self.impact_k * np.sqrt(participation) * 10_000.0)


@dataclass
class _OpenLeg:
    """Tracks an in-flight round trip so a ClosedTrade can be emitted."""
    strategy: str
    direction: int
    qty: Decimal
    entry_ts: datetime
    entry_price: Decimal
    entry_index: int
    fees: Decimal = Decimal(0)
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class BacktestResult:
    report: PerformanceReport
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    signals: list[tuple[Signal, Decision]] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)

    def rejection_summary(self) -> str:
        if not self.rejections:
            return "no signals were rejected"
        rows = sorted(self.rejections.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{code}×{n}" for code, n in rows)


class BacktestEngine:
    """Single-symbol, single-strategy. Portfolio-level runs compose these and
    combine the equity curves through the HRP allocator (ARCHITECTURE.md §8)."""

    def __init__(
        self,
        strategy: Strategy,
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
    ) -> None:
        self.strategy = strategy
        self.limits = limits or RiskLimits()
        self.costs = costs or CostModel()
        self.initial_equity = Decimal(str(initial_equity))
        self.timeframe = timeframe
        self.vol_lookback = vol_lookback
        # ATR used by the trailing stop. Kept separate from a strategy's own ATR
        # period on purpose: the trail is a risk construct, not a signal one.
        self.atr_period = atr_period
        # How many bars a resting limit order stays alive. PaperBroker's
        # equivalent is `maker_timeout` (15 min); on 4h bars 3 is the nearest
        # sane analogue. Market/IOC orders always get exactly 1.
        self.maker_ttl_bars = maker_ttl_bars
        # How many bars a market order keeps working its unfilled remainder.
        # See `_order_ttl` — abandoning it is the silent-truncation defect.
        self.market_ttl_bars = market_ttl_bars
        # Wired through rather than hardcoded so the arming gate sits on the
        # same code path in backtest, paper and live.
        self.armed = armed
        self.risk = RiskEngine(self.limits)

    # ── main loop ────────────────────────────────────────────────────────
    def run(
        self,
        bars: list[Bar],
        *,
        symbol: str | None = None,
        regimes: list[Regime] | None = None,
        n_trials: int = 1,
    ) -> BacktestResult:
        if len(bars) < self.strategy.warmup + 2:
            raise ValueError(
                f"{len(bars)} bars is fewer than the strategy's warmup "
                f"({self.strategy.warmup}) + 2; the result would be an artefact")

        sym = symbol or bars[0].symbol or "UNKNOWN"
        o = np.array([b.open for b in bars], dtype=np.float64)
        h = np.array([b.high for b in bars], dtype=np.float64)
        l = np.array([b.low for b in bars], dtype=np.float64)
        c = np.array([b.close for b in bars], dtype=np.float64)
        v = np.array([b.volume for b in bars], dtype=np.float64)

        # All account bookkeeping — cash, positions, HWM, day/week rollover,
        # loss streaks + breaker expiry, order-rate window — lives in the
        # builder, which the live engine will drive identically. Hand-rolling
        # it here is what produced five silent backtest/live divergences.
        acct = AccountStateBuilder(self.initial_equity, self.limits)

        # ATR computed ONCE over the whole series, O(n) rather than O(n²).
        # Safe because every indicator here is causal — asserted by
        # test_indicators_are_causal — so atr(full)[i] == atr(full[:i+1])[i].
        atr_series = atr(h, l, c, self.atr_period)

        # The same hoist, generalised to every indicator the strategy asks for
        # (B2). One cache for the whole run; the per-bar Context gets a view
        # truncated at that bar, so the strategy's reach is unchanged.
        features = FeatureSet(o, h, l, c, v)

        leg: _OpenLeg | None = None
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        trades: list[ClosedTrade] = []
        signal_log: list[tuple[Signal, Decision]] = []
        rejections: dict[str, int] = {}
        # (order, bars of life remaining). Unfilled resting orders used to be
        # dropped after exactly one bar, while PaperBroker rests them for 15
        # minutes — so the maker-first thesis, the largest quantified edge in
        # the project, was never simulated under its own fill model.
        pending: list[tuple[Order, int]] = []
        bars_in_market = 0

        def settle(exit_price: Decimal, realized: Decimal, reason: ExitReason,
                   index: int, ts: datetime) -> None:
            """Emit a ClosedTrade and update the loss streak. Clears `leg`."""
            nonlocal leg
            if leg is None:
                return
            trades.append(ClosedTrade(
                strategy=leg.strategy, symbol=sym, direction=leg.direction,
                qty=leg.qty, entry_ts=leg.entry_ts, exit_ts=ts,
                entry_price=leg.entry_price, exit_price=exit_price,
                gross_pnl=realized + leg.fees, fees=leg.fees,
                exit_reason=reason, bars_held=index - leg.entry_index,
                mae=leg.mae, mfe=leg.mfe,
            ))
            acct.record_trade_result(leg.strategy, realized - leg.fees, ts)
            leg = None

        for i, bar in enumerate(bars):
            mark = Decimal(str(bar.close))
            raw_atr = atr_series[i]
            cur_atr = float(raw_atr) if np.isfinite(raw_atr) else 0.0
            position = acct.position(self.strategy.name, sym)

            # ── 1. fill orders raised on a PREVIOUS bar ──────────────────
            still_pending: list[tuple[Order, int]] = []
            for order, ttl in pending:
                fill = self._simulate_fill(order, bar)
                if fill is None:
                    if ttl > 1:                       # still resting
                        still_pending.append((order, ttl - 1))
                    else:
                        acct.release(order.client_order_id)   # expired unfilled
                    continue

                acct.release(order.client_order_id, fill.qty * fill.price)
                realized = acct.apply_fill(fill)

                if leg is None and not position.is_flat:
                    leg = _OpenLeg(order.strategy, position.direction, abs(position.qty),
                                   bar.ts, position.avg_entry, i, fill.fee)
                    position.stop_price = order.attached_stop
                    position.target_price = order.attached_target
                    position.trail_atr_mult = order.trail_atr_mult
                    position.strategy = order.strategy
                    position.highest_close = position.lowest_close = bar.close
                elif leg is not None:
                    leg.fees += fill.fee
                    if position.is_flat:
                        reason = (ExitReason(order.meta["reason_enum"])
                                  if "reason_enum" in order.meta else ExitReason.SIGNAL)
                        settle(fill.price, realized, reason, i, bar.ts)

                # A partial fill's remainder used to evaporate — neither what an
                # exchange does nor what PaperBroker does. Carry it forward.
                remainder = order.qty - fill.qty
                if remainder > 0 and ttl > 1:
                    still_pending.append((replace(order, qty=remainder), ttl - 1))
                elif remainder > 0:
                    acct.release(order.client_order_id)
            pending = still_pending

            # ── 2. protective exits, checked intrabar against high/low ───
            if not position.is_flat:
                bars_in_market += 1
                self._update_excursions(leg, position, bar)
                exit_order = self._check_protective_exits(position, bar, cur_atr)
                if exit_order is not None:
                    # Protective exits fill on THIS bar at the stop price: the
                    # level was touched intrabar, so deferring to t+1 would
                    # understate losses — the optimistic direction.
                    px = exit_order.limit_price or mark
                    qty = abs(position.qty)
                    fee = self.costs.fee(qty * px, is_maker=False)
                    fill = Fill(exit_order.client_order_id, bar.ts, sym,
                                exit_order.side, qty, px, fee, is_maker=False,
                                strategy=self.strategy.name)
                    realized = acct.apply_fill(fill)
                    if leg is not None:
                        leg.fees += fee
                        settle(px, realized,
                               ExitReason(exit_order.meta["reason_enum"]), i, bar.ts)

            # ── 3. mark to market ───────────────────────────────────────
            # Equity, high-water mark, day/week rollover, consecutive-loss
            # breaker expiry and order-rate pruning all happen here, in the
            # same call the live engine makes.
            equity = acct.mark(bar.ts, {sym: mark})
            equity_curve.append(float(equity))
            timestamps.append(bar.ts)

            # ── 4. ask the strategy ─────────────────────────────────────
            if i < self.strategy.warmup or i >= len(bars) - 1:
                continue

            ctx = Context(
                symbol=sym, timeframe=self.timeframe, index=i, ts=bar.ts,
                # Slices, not views onto the full array: the strategy cannot
                # reach past `i` even by accident.
                open=o[:i + 1], high=h[:i + 1], low=l[:i + 1],
                close=c[:i + 1], volume=v[:i + 1],
                position=position,
                regime=regimes[i] if regimes else Regime.UNKNOWN,
                params=self.strategy.params,
                f=features.view(i),
            )
            try:
                signals = self.strategy.on_bar(ctx)
            except Exception as exc:  # quarantine, don't kill the run
                rejections["STRATEGY_ERROR"] = rejections.get("STRATEGY_ERROR", 0) + 1
                signals = []
                if len(rejections) and rejections["STRATEGY_ERROR"] > 50:
                    raise RuntimeError(
                        f"{self.strategy.name} raised >50 times; last: {exc}") from exc

            if not signals:
                continue

            account = acct.snapshot(bar.ts, {sym: mark},
                                    open_order_count=len(pending))
            realized_vol = self._realized_ann_vol(c, i)

            for sig in signals:
                decision = self.risk.evaluate(
                    sig, account, realized_ann_vol=realized_vol,
                    strategy_weight=1.0, armed=self.armed,
                )
                signal_log.append((sig, decision))
                if not decision.approved:
                    code = decision.code.value if decision.code else "UNKNOWN"
                    rejections[code] = rejections.get(code, 0) + 1
                    continue
                # Assert before binding, not after: `order` is already the
                # Order-typed loop variable from the pending-fill pass above,
                # so narrowing has to happen on `decision.order` itself.
                assert decision.order is not None
                order = decision.order
                if sig.direction == 0:
                    order = _with_meta(order, reason_enum=ExitReason.SIGNAL.value)
                # Counted against the rate limiter, which the backtest used to
                # hardcode to zero and therefore never exercise.
                acct.record_order(bar.ts)
                acct.reserve(order.client_order_id, order.strategy, order.symbol,
                             order.notional(mark))
                pending.append((order, self._order_ttl(order)))
                # Re-snapshot so a second signal on this bar sees the
                # reservation the first one just made.
                account = acct.snapshot(bar.ts, {sym: mark},
                                        open_order_count=len(pending))

        # S2-3: metrics start where trading can. The ~200 warm-up bars are a
        # flat prefix on the equity curve — trading is impossible there, so
        # counting them depressed the return mean and stdev, inflated
        # `n_periods` (and therefore `years` under CAGR), and diluted
        # `exposure = bars_in_market / n`. A systematic bias in the exact
        # number the promotion gate reads. The benchmark is cut at the same
        # bar, or the comparison would hand buy-and-hold a head start the
        # strategy is forbidden from using. The full curve stays in
        # `BacktestResult` for plotting.
        metrics_start = min(self.strategy.warmup, len(equity_curve))
        report = build_report(
            equity_curve[metrics_start:], trades, self.timeframe,
            initial_equity=float(self.initial_equity),
            total_fees=float(acct.total_fees),
            bars_in_market=bars_in_market,
            n_trials=n_trials,
            benchmark_equity=self._buy_and_hold(c[metrics_start:]),
        )
        return BacktestResult(report, equity_curve, timestamps, trades,
                              signal_log, rejections)

    # ── helpers ──────────────────────────────────────────────────────────

    def _order_ttl(self, order: Order) -> int:
        """Bars of life an order gets before its remainder is abandoned.

        Keyed on order TYPE, not time-in-force, and the distinction matters:

        * **MARKET → `market_ttl_bars`.** A market order is an instruction to
          acquire (or exit) a position. The participation cap that prevented a
          full fill is a *simulation artifact*, not an exchange rule, so
          abandoning the remainder would silently under-report both the
          position and the execution cost of getting it. It especially matters
          on exits: dropping an exit remainder leaves the backtest believing it
          is flat when it is not. Our risk engine tags market orders IOC to
          mean "do not rest as maker" — not "abandon the position."
        * **LIMIT + IOC/FOK → 1.** Here IOC is the actual venue semantic:
          immediate or cancel.
        * **LIMIT resting → `maker_ttl_bars`**, the backtest analogue of
          PaperBroker's `maker_timeout`.

        Extracted as a method so the rule is directly testable rather than
        buried in the run loop.
        """
        if order.type is OrderType.MARKET:
            return max(1, self.market_ttl_bars)
        if order.tif in (TimeInForce.IOC, TimeInForce.FOK):
            return 1
        return max(1, self.maker_ttl_bars)

    def _simulate_fill(self, order: Order, bar: Bar) -> Fill | None:
        """Fill `order` against `bar`. Returns None if it would not have filled."""
        cm = self.costs
        # Participation cap: you cannot backtest yourself into liquidity that
        # was not there.
        max_qty = Decimal(str(bar.dollar_volume * cm.max_bar_volume_pct)) / \
            Decimal(str(bar.open or bar.close or 1.0))
        qty = min(order.qty, max_qty) if max_qty > 0 else Decimal(0)
        if qty <= 0:
            return None

        if order.type is OrderType.MARKET or order.tif is TimeInForce.IOC:
            base = Decimal(str(bar.open))   # market orders fill at the OPEN of
                                            # the next bar, not its close
            slip_bps = cm.slippage_bps(float(qty * base), bar.dollar_volume)
            slip = Decimal(str(slip_bps / 10_000.0))
            price = base * (Decimal(1) + slip * Decimal(order.side.sign))
            fee = cm.fee(qty * price, is_maker=False)
            # `strategy=` is NOT optional. Omitting it books the fill under
            # the anonymous key "|SYMBOL": the strategy then sees itself flat
            # forever (re-enters, never exits), protective stops never engage
            # (they read the strategy-keyed book), and equity drifts via a
            # phantom book — the exact PaperBroker defect of 2026-07-28,
            # rediscovered here on 2026-07-30 when a cadence investigation
            # found every single-symbol backtest closing zero trades.
            return Fill(order.client_order_id, bar.ts, order.symbol, order.side,
                        qty, price, fee, is_maker=False,
                        strategy=order.strategy)

        # Limit / post-only: fills only if the bar actually traded through it.
        limit = order.limit_price
        if limit is None:
            return None
        touched = (order.side is Side.BUY and Decimal(str(bar.low)) <= limit) or \
                  (order.side is Side.SELL and Decimal(str(bar.high)) >= limit)
        if not touched:
            return None
        # A post-only that would have crossed at the open never rests as maker.
        if order.tif is TimeInForce.POST_ONLY:
            crossed = (order.side is Side.BUY and Decimal(str(bar.open)) < limit) or \
                      (order.side is Side.SELL and Decimal(str(bar.open)) > limit)
            if crossed:
                return None
        fee = cm.fee(qty * limit, is_maker=True)
        return Fill(order.client_order_id, bar.ts, order.symbol, order.side,
                    qty, limit, fee, is_maker=True, strategy=order.strategy)

    def _check_protective_exits(
        self, pos: Position, bar: Bar, cur_atr: float,
    ) -> Order | None:
        """Stop / trailing stop / target, evaluated against the bar's range.

        Where both the stop and the target were touched in the same bar we
        assume the STOP filled. Without tick data you cannot know the order,
        and assuming the favourable one is how backtests lie.

        **The trail is persisted onto `pos.stop_price` and only ever moves in
        the favourable direction.** The previous version recomputed it each bar
        from `original_stop` and the *current bar's range*, so a single
        wide-range bar moved the stop backwards — measured `[114, 114, 90, 114]`
        across bars of range 2/2/12/2. A stop that retreats is not a trailing
        stop; it gave back profit on exactly the volatile bars where protection
        matters. It now uses a real ATR series (computed once per run, O(n)) and
        ratchets monotonically.
        """
        long = pos.qty > 0
        if pos.trail_atr_mult and cur_atr > 0:
            offset = Decimal(str(pos.trail_atr_mult)) * Decimal(str(cur_atr))
            if long and pos.highest_close is not None:
                candidate = Decimal(str(pos.highest_close)) - offset
                if pos.stop_price is None or candidate > pos.stop_price:
                    pos.stop_price = candidate
            elif not long and pos.lowest_close is not None:
                candidate = Decimal(str(pos.lowest_close)) + offset
                if pos.stop_price is None or candidate < pos.stop_price:
                    pos.stop_price = candidate

        stop = pos.stop_price
        if stop is not None:
            hit = (long and Decimal(str(bar.low)) <= stop) or \
                  (not long and Decimal(str(bar.high)) >= stop)
            if hit:
                return Order(
                    symbol=pos.symbol, side=Side.SELL if long else Side.BUY,
                    type=OrderType.MARKET, qty=abs(pos.qty),
                    limit_price=stop, reduce_only=True, strategy=pos.strategy,
                    meta={"reason_enum": (ExitReason.TRAIL.value
                                          if pos.trail_atr_mult else ExitReason.STOP.value)},
                )
        if pos.target_price is not None:
            hit = (long and Decimal(str(bar.high)) >= pos.target_price) or \
                  (not long and Decimal(str(bar.low)) <= pos.target_price)
            if hit:
                return Order(
                    symbol=pos.symbol, side=Side.SELL if long else Side.BUY,
                    type=OrderType.MARKET, qty=abs(pos.qty),
                    limit_price=pos.target_price, reduce_only=True,
                    strategy=pos.strategy,
                    meta={"reason_enum": ExitReason.TARGET.value},
                )
        return None

    @staticmethod
    def _update_excursions(leg: _OpenLeg | None, pos: Position, bar: Bar) -> None:
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
        per = ewma_volatility(rets.tolist(), halflife=self.vol_lookback)
        return annualize_vol(per, self.timeframe)

    def _buy_and_hold(self, close: np.ndarray) -> list[float]:
        """Benchmark, net of one round trip of taker fees.

        The fee must be applied to the UNITS BOUGHT and again at the exit —
        NOT as a constant multiplier on the whole curve. A constant multiplier
        cancels out of the terminal ratio (b[-1]/b[0]) and silently produces a
        gross benchmark labelled as net, which flatters every strategy that is
        compared against it.
        """
        if len(close) == 0 or close[0] <= 0:
            return []
        fee = self.costs.taker_fee_bps / 10_000.0
        eq0 = float(self.initial_equity)
        units = eq0 * (1.0 - fee) / float(close[0])
        curve = [eq0] + [units * float(px) for px in close[1:]]
        if len(curve) > 1:
            curve[-1] *= (1.0 - fee)   # pay the exit
        return curve


def _with_meta(order: Order, **kw: object) -> Order:
    from dataclasses import replace
    return replace(order, meta={**order.meta, **kw})


def _atr_proxy(bar: Bar) -> Decimal:
    """Single-bar range as a stand-in for ATR in the trailing-stop ratchet.

    Approximation, and it is the conservative direction: a wider proxy trails
    further away, so the backtest gives back MORE than a true-ATR trail would.
    """
    return Decimal(str(max(bar.high - bar.low, 1e-12)))


__all__ = ["BacktestEngine", "BacktestResult", "CostModel"]
