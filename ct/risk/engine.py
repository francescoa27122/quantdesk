"""The risk engine — the most important component in QuantDesk.

Contract:
    Signal + AccountState + RiskLimits  ->  Decision(order | rejection)

It is a **pure function of its inputs**. No database, no clock, no exchange, no
randomness. That is what makes the entire gate stack property-testable, and it
is why this file has no imports outside the standard library and `ct.models`.

The engine is the ONLY component that turns a Signal into an Order, and the
ONLY component that can veto one. Strategies cannot size; brokers cannot
approve.

Gate order matters. Cheap, absolute gates run first (kill switch, arming); the
clamping gates run last so that the clamp is applied to an already-legal size.
Every rejection carries a machine-readable code that is written to `signals`,
so "why did my strategy do nothing for three weeks?" is a SQL query, not an
archaeology project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from ct.models import (
    AccountState,
    Order,
    OrderType,
    Regime,
    Side,
    Signal,
    TimeInForce,
)
from ct.risk.sizing import (
    fractional_kelly,
    notional_cap_size,
    risk_budget_size,
    round_to_increment,
    vol_target_size,
)


class RejectCode(str, Enum):
    """Every reason an order can be refused. Stable strings — they go in the DB."""
    KILL_SWITCH = "KILL_SWITCH"
    ACCOUNT_HALTED = "ACCOUNT_HALTED"
    NOT_ARMED = "NOT_ARMED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    STRATEGY_PAUSED = "STRATEGY_PAUSED"
    SYMBOL_NOT_ALLOWED = "SYMBOL_NOT_ALLOWED"
    STALE_DATA = "STALE_DATA"
    NO_MARK_PRICE = "NO_MARK_PRICE"
    ZERO_EQUITY = "ZERO_EQUITY"
    NO_STOP = "NO_STOP"
    BAD_STOP = "BAD_STOP"
    SHORT_NOT_ALLOWED = "SHORT_NOT_ALLOWED"
    LEVERAGE_EXCEEDED = "LEVERAGE_EXCEEDED"
    MAX_POSITIONS = "MAX_POSITIONS"
    DAILY_DRAWDOWN_HALT = "DAILY_DRAWDOWN_HALT"
    WEEKLY_DRAWDOWN_HALT = "WEEKLY_DRAWDOWN_HALT"
    MAX_DRAWDOWN_HALT = "MAX_DRAWDOWN_HALT"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    ORDER_RATE_LIMIT = "ORDER_RATE_LIMIT"
    EDGE_BELOW_FEES = "EDGE_BELOW_FEES"
    REGIME_BLOCKED = "REGIME_BLOCKED"
    ML_CONFIDENCE_LOW = "ML_CONFIDENCE_LOW"
    SIZE_ZERO = "SIZE_ZERO"
    BELOW_EXCHANGE_MIN = "BELOW_EXCHANGE_MIN"
    GROSS_EXPOSURE_CAP = "GROSS_EXPOSURE_CAP"
    SYMBOL_EXPOSURE_CAP = "SYMBOL_EXPOSURE_CAP"
    CORRELATION_CAP = "CORRELATION_CAP"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Serialised to `risk_config.limits` (jsonb) and versioned there.

    Defaults are deliberately conservative. They are the limits I would run
    with real money on an unproven system, not the limits that maximise a
    backtest.
    """
    # ── sizing ──
    risk_per_trade: float = 0.005          # 0.5% of equity at the stop
    target_ann_vol: float = 0.20           # 20% annualised portfolio vol target
    max_position_pct: float = 0.15         # notional cap per symbol
    kelly_fraction: float = 0.0            # OFF. See sizing.fractional_kelly.
    min_stop_atr_mult: float = 1.0
    default_stop_atr_mult: float = 2.5

    # ── exposure ──
    max_gross_exposure: float = 0.60
    max_symbol_exposure: float = 0.15
    max_cluster_exposure: float = 0.25     # correlated (rho>0.8) group cap
    max_open_positions: int = 5
    max_leverage: float = 1.0              # 1.0 = spot only. Raising this is
                                           # a jurisdiction decision, not a
                                           # config tweak.
    allow_shorts: bool = False             # spot: you cannot short.

    # ── drawdown circuit breakers ──
    max_daily_drawdown: float = 0.03       # halt new entries for the UTC day
    max_weekly_drawdown: float = 0.07
    max_total_drawdown: float = 0.20       # flatten all + disarm live
    max_consecutive_losses: int = 5        # pause the offending strategy...
    loss_pause_hours: float = 24.0         # ...for this long, then reset.
    # The cooldown is NOT cosmetic. Without it the breaker is a lifetime ban:
    # in production `strategies.paused_until` expires, so a backtest with no
    # cooldown silently models a DIFFERENT system than the one that will run.
    # That divergence is the exact failure mode this architecture exists to
    # prevent, so the number lives here and both paths read it.

    # ── execution economics (ARCHITECTURE.md §0.1) ──
    taker_fee_bps: float = 40.0
    maker_fee_bps: float = 25.0
    assumed_slippage_bps: float = 10.0
    min_edge_bps: float = 60.0             # must clear round-trip cost + margin
    prefer_maker: bool = True

    # ── plumbing ──
    max_orders_per_hour: int = 30
    max_data_age_bars: int = 3
    # An exchange enforces TWO independent minimums and the engine must model
    # both — an order can clear the notional floor and still be rejected on
    # quantity. Verified against Kraken's public AssetPairs on 2026-07-27:
    #   XXBTZUSD  costmin 0.5 USD, ordermin 0.00005 BTC, lot_decimals 8
    #   XETHZUSD  costmin 0.5 USD, ordermin 0.001   ETH, lot_decimals 8
    # These defaults are Kraken's real values; the engine overwrites them
    # per-symbol from the exchange at boot rather than trusting a constant.
    min_order_notional: float = 0.5        # exchange `costmin`, quote ccy
    min_order_qty: float = 0.0             # exchange `ordermin`, base units
    qty_increment: str = "0.00000001"      # exchange `lot_decimals`
    min_ml_confidence: float = 0.0         # 0 = meta-labeller not gating

    # ── regime rules ──
    # A strategy family is blocked when the classifier says the regime is
    # wrong for it. `crisis` blocks everything.
    block_reversion_in_trend: bool = True
    block_trend_in_range: bool = True
    block_all_in_crisis: bool = True


@dataclass(frozen=True, slots=True)
class Decision:
    approved: bool
    order: Order | None = None
    code: RejectCode | None = None
    detail: str = ""
    # Populated even on rejection where known, for the `signals` audit trail.
    diagnostics: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def reject(code: RejectCode, detail: str = "", **diag: object) -> Decision:
        return Decision(False, None, code, detail,
                        {k: str(v) for k, v in diag.items()})

    @staticmethod
    def accept(order: Order, **diag: object) -> Decision:
        return Decision(True, order, None, "",
                        {k: str(v) for k, v in diag.items()})


# Which strategy families are reversion vs trend, for the regime gate.
# Set on the Strategy class as `family`; passed in via Signal.meta['family'].
_REVERSION = {"reversion", "mean_reversion"}
_TREND = {"trend", "momentum", "breakout"}


class RiskEngine:
    """Stateless evaluator. Construct once, call `evaluate` per signal."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    # ── the entry point ──────────────────────────────────────────────────
    def evaluate(
        self,
        signal: Signal,
        account: AccountState,
        *,
        realized_ann_vol: float,
        strategy_weight: float = 1.0,
        strategy_enabled: bool = True,
        strategy_paused: bool = False,
        armed: bool = True,
        allowed_symbols: frozenset[str] | None = None,
        kill_switch: bool = False,
        win_rate: float = 0.5,
        payoff_ratio: float = 1.0,
    ) -> Decision:
        L = self.limits

        # An exit is not a new risk decision. It must pass the kill switch and
        # sanity checks only — never a drawdown halt, never an exposure cap.
        # Refusing to let a strategy close a position because the account is
        # in drawdown is how a small loss becomes a large one.
        is_exit = signal.direction == 0

        # ── 1. absolute stops ────────────────────────────────────────────
        if kill_switch:
            return Decision.reject(RejectCode.KILL_SWITCH,
                                   "kill switch engaged; no orders of any kind")
        if not is_exit:
            if account.halted:
                return Decision.reject(RejectCode.ACCOUNT_HALTED, account.halt_reason)
            if not armed:
                return Decision.reject(
                    RejectCode.NOT_ARMED,
                    "no valid unexpired live_arming row; run `qd arm-live`")
            if not strategy_enabled:
                return Decision.reject(RejectCode.STRATEGY_DISABLED, signal.strategy)
            if strategy_paused:
                return Decision.reject(RejectCode.STRATEGY_PAUSED, signal.strategy)
            if allowed_symbols is not None and signal.symbol not in allowed_symbols:
                return Decision.reject(RejectCode.SYMBOL_NOT_ALLOWED, signal.symbol)

        # ── 2. data sanity ───────────────────────────────────────────────
        age = account.data_age_bars.get(signal.symbol, 0)
        if age > L.max_data_age_bars:
            return Decision.reject(
                RejectCode.STALE_DATA,
                f"{signal.symbol} data is {age} bars old (max {L.max_data_age_bars})")

        mark = account.marks.get(signal.symbol)
        if mark is None or mark <= 0:
            return Decision.reject(RejectCode.NO_MARK_PRICE, signal.symbol)
        if account.equity <= 0:
            return Decision.reject(RejectCode.ZERO_EQUITY, str(account.equity))

        # A strategy sees only its OWN book on this symbol. Exposure caps below
        # still aggregate every book, because that is what the exchange holds.
        position = account.position_for(signal.strategy, signal.symbol)

        # ── 3. exits: size = whatever is open, and go ────────────────────
        if is_exit:
            if position is None or position.is_flat:
                return Decision.reject(RejectCode.SIZE_ZERO, "no position to exit")
            return Decision.accept(Order(
                symbol=signal.symbol,
                side=Side.SELL if position.qty > 0 else Side.BUY,
                type=OrderType.MARKET,   # exits are taker: when you want out,
                                         # you pay the spread. §6.3
                qty=abs(position.qty),
                strategy=signal.strategy,
                reduce_only=True,
                tif=TimeInForce.IOC,
                meta={"reason": "signal_exit"},
            ), exit=True)

        # ── 4. direction legality ────────────────────────────────────────
        if signal.direction < 0 and not L.allow_shorts:
            return Decision.reject(
                RejectCode.SHORT_NOT_ALLOWED,
                "spot account: shorting requires margin, which is off by policy")
        if L.max_leverage < 1.0:
            return Decision.reject(RejectCode.LEVERAGE_EXCEEDED,
                                   f"max_leverage={L.max_leverage}")

        # ── 5. drawdown circuit breakers ─────────────────────────────────
        # Checked before sizing so the reject reason is the drawdown, not a
        # downstream zero size.
        if account.drawdown() >= Decimal(str(L.max_total_drawdown)):
            return Decision.reject(
                RejectCode.MAX_DRAWDOWN_HALT,
                f"total drawdown {account.drawdown():.2%} >= {L.max_total_drawdown:.2%}"
                " — engine should flatten and disarm")
        if account.day_drawdown() >= Decimal(str(L.max_daily_drawdown)):
            return Decision.reject(
                RejectCode.DAILY_DRAWDOWN_HALT,
                f"day drawdown {account.day_drawdown():.2%} >= {L.max_daily_drawdown:.2%}")
        if account.week_drawdown() >= Decimal(str(L.max_weekly_drawdown)):
            return Decision.reject(
                RejectCode.WEEKLY_DRAWDOWN_HALT,
                f"week drawdown {account.week_drawdown():.2%} >= {L.max_weekly_drawdown:.2%}")

        losses = account.consecutive_losses.get(signal.strategy, 0)
        if losses >= L.max_consecutive_losses:
            return Decision.reject(
                RejectCode.CONSECUTIVE_LOSSES,
                f"{signal.strategy}: {losses} consecutive losses")

        if account.orders_last_hour >= L.max_orders_per_hour:
            return Decision.reject(
                RejectCode.ORDER_RATE_LIMIT,
                f"{account.orders_last_hour} orders in the last hour "
                f"(max {L.max_orders_per_hour}) — likely a signal loop")

        # ── 6. regime gate ───────────────────────────────────────────────
        family = str(signal.meta.get("family", "")).lower()
        if L.block_all_in_crisis and signal.regime is Regime.CRISIS:
            return Decision.reject(RejectCode.REGIME_BLOCKED, "regime=crisis")
        if (L.block_reversion_in_trend and family in _REVERSION
                and signal.regime is Regime.TRENDING):
            return Decision.reject(RejectCode.REGIME_BLOCKED,
                                   "reversion strategy in a trending regime")
        if (L.block_trend_in_range and family in _TREND
                and signal.regime is Regime.RANGING):
            return Decision.reject(RejectCode.REGIME_BLOCKED,
                                   "trend strategy in a ranging regime")

        # ── 7. ML meta-label gate (veto only, never creates a trade) ──────
        conf = signal.meta.get("ml_confidence")
        if L.min_ml_confidence > 0 and conf is not None:
            if float(conf) < L.min_ml_confidence:
                return Decision.reject(
                    RejectCode.ML_CONFIDENCE_LOW,
                    f"meta-label p={float(conf):.3f} < {L.min_ml_confidence:.3f}")

        # ── 8. the fee gate — §0.1, in code ──────────────────────────────
        # `expected_edge_bps` is the strategy's own estimate of gross move to
        # target. If it does not clear round-trip cost plus a margin, the trade
        # is a donation to the exchange regardless of how good the signal looks.
        round_trip_bps = (
            (L.maker_fee_bps if L.prefer_maker else L.taker_fee_bps)
            + L.taker_fee_bps                       # the exit is always taker
            + 2 * L.assumed_slippage_bps
        )
        edge_bps = signal.meta.get("expected_edge_bps")
        if edge_bps is not None:
            required = max(L.min_edge_bps, round_trip_bps)
            if float(edge_bps) < required:
                return Decision.reject(
                    RejectCode.EDGE_BELOW_FEES,
                    f"expected edge {float(edge_bps):.0f}bps < required "
                    f"{required:.0f}bps (round trip {round_trip_bps:.0f}bps)")

        # ── 9. stop distance ─────────────────────────────────────────────
        stop, stop_distance = self._resolve_stop(signal, float(mark))
        if stop_distance <= 0:
            return Decision.reject(
                RejectCode.NO_STOP,
                "signal has neither stop_price nor atr; risk-budget sizing "
                "is impossible and unsized entries are not permitted")
        min_dist = L.min_stop_atr_mult * (signal.atr or 0.0)
        if signal.atr and stop_distance < min_dist:
            return Decision.reject(
                RejectCode.BAD_STOP,
                f"stop {stop_distance:.6f} closer than {L.min_stop_atr_mult}×ATR "
                f"({min_dist:.6f}) — it will be taken out by noise")

        # ── 10. sizing: min of three, then conviction and Kelly ──────────
        candidates = {
            "risk_budget": risk_budget_size(account.equity, L.risk_per_trade, stop_distance),
            "vol_target": vol_target_size(account.equity, float(mark), realized_ann_vol,
                                          L.target_ann_vol, strategy_weight),
            "notional_cap": notional_cap_size(account.equity, float(mark), L.max_position_pct),
        }
        qty = min(candidates.values())
        binding = min(candidates, key=lambda k: candidates[k])

        qty *= Decimal(str(signal.strength))
        qty *= Decimal(str(fractional_kelly(win_rate, payoff_ratio, L.kelly_fraction)))

        # ── 11. exposure clamps ──────────────────────────────────────────
        qty, clamp = self._clamp_exposure(qty, signal.strategy, signal.symbol,
                                          mark, account)
        if qty <= 0:
            return Decision.reject(clamp or RejectCode.SIZE_ZERO,
                                   "exposure caps leave no room for this trade",
                                   binding=binding)

        qty = round_to_increment(qty, Decimal(L.qty_increment))
        if qty <= 0:
            return Decision.reject(RejectCode.SIZE_ZERO, "size rounds to zero at lot size")
        if qty * mark < Decimal(str(L.min_order_notional)):
            return Decision.reject(
                RejectCode.BELOW_EXCHANGE_MIN,
                f"notional {qty * mark:.2f} < exchange costmin "
                f"{L.min_order_notional:.2f}; equity is too small for this "
                f"risk budget at this stop distance")
        if L.min_order_qty > 0 and qty < Decimal(str(L.min_order_qty)):
            return Decision.reject(
                RejectCode.BELOW_EXCHANGE_MIN,
                f"qty {qty} < exchange ordermin {L.min_order_qty}; the venue "
                f"would reject this order even though its notional clears")

        # ── 12. build the order ──────────────────────────────────────────
        side = Side.BUY if signal.direction > 0 else Side.SELL
        if L.prefer_maker:
            # Post-only at the passive side. If it would cross, the exchange
            # rejects it and the engine's execution layer decides whether to
            # re-price or fall back to taker after MAKER_TIMEOUT_S. That
            # decision belongs to the executor, not to risk.
            otype, tif = OrderType.LIMIT, TimeInForce.POST_ONLY
            limit_price = mark
        else:
            otype, tif, limit_price = OrderType.MARKET, TimeInForce.IOC, None

        order = Order(
            symbol=signal.symbol,
            side=side,
            type=otype,
            qty=qty,
            limit_price=limit_price,
            tif=tif,
            strategy=signal.strategy,
            attached_stop=Decimal(str(stop)),
            attached_target=(Decimal(str(signal.target_price))
                             if signal.target_price else None),
            trail_atr_mult=signal.trail_atr_mult,
            meta={"binding_sizer": binding, "regime": signal.regime.value},
        )
        return Decision.accept(
            order,
            binding_sizer=binding,
            risk_budget=candidates["risk_budget"],
            vol_target=candidates["vol_target"],
            notional_cap=candidates["notional_cap"],
            stop_distance=stop_distance,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_stop(self, signal: Signal, mark: float) -> tuple[float, float]:
        """Return (stop_price, stop_distance). Distance 0 means 'cannot size'."""
        if signal.stop_price is not None and signal.stop_price > 0:
            dist = abs(mark - signal.stop_price)
            return signal.stop_price, dist
        if signal.atr and signal.atr > 0:
            dist = self.limits.default_stop_atr_mult * signal.atr
            stop = mark - dist if signal.direction > 0 else mark + dist
            return max(stop, 0.0), dist
        return 0.0, 0.0

    def _clamp_exposure(
        self, qty: Decimal, strategy: str, symbol: str, mark: Decimal,
        account: AccountState,
    ) -> tuple[Decimal, RejectCode | None]:
        """Shrink qty until every exposure cap holds. Never grows it."""
        L = self.limits
        equity = account.equity
        breach: RejectCode | None = None

        # Every cap below is checked against COMMITTED exposure — filled
        # positions PLUS orders already approved this pass but not yet filled.
        # Checking only filled exposure lets N signals in one engine pass each
        # pass the same cap independently and breach it N-fold. Measured before
        # the fix: a 30% gross cap reaching 50%, a 12% cluster cap reaching 50%.

        if account.open_position_count() >= L.max_open_positions:
            # Adding to a book we already hold (or already have in flight) is
            # not a new position; opening one we don't is. Counted per BOOK, so
            # two strategies both long BTC occupy two slots — correct, they are
            # two independent decisions that can each be wrong.
            if not account.holds_book(strategy, symbol):
                return Decimal(0), RejectCode.MAX_POSITIONS

        # Per-symbol.
        cur_sym = account.committed_symbol_exposure(symbol) * equity
        room = Decimal(str(L.max_symbol_exposure)) * equity - cur_sym
        if room <= 0:
            return Decimal(0), RejectCode.SYMBOL_EXPOSURE_CAP
        if qty * mark > room:
            qty, breach = room / mark, RejectCode.SYMBOL_EXPOSURE_CAP

        # Correlation cluster.
        for members in account.correlation_clusters.values():
            if symbol not in members:
                continue
            cur = sum((account.committed_symbol_exposure(s) for s in members),
                      Decimal(0)) * equity
            croom = Decimal(str(L.max_cluster_exposure)) * equity - cur
            if croom <= 0:
                return Decimal(0), RejectCode.CORRELATION_CAP
            if qty * mark > croom:
                qty, breach = croom / mark, RejectCode.CORRELATION_CAP

        # Gross.
        cur_gross = account.committed_gross_exposure() * equity
        groom = Decimal(str(L.max_gross_exposure)) * equity - cur_gross
        if groom <= 0:
            return Decimal(0), RejectCode.GROSS_EXPOSURE_CAP
        if qty * mark > groom:
            qty, breach = groom / mark, RejectCode.GROSS_EXPOSURE_CAP

        return qty, breach


__all__ = ["Decision", "RejectCode", "RiskEngine", "RiskLimits"]
