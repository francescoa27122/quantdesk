"""Walk-forward analysis, purged CV, and Monte Carlo.

This module exists because a single backtest over a single period tells you
almost nothing, and a parameter sweep over a single period tells you less than
nothing — it tells you a number that is confidently wrong.

Three techniques, each fixing a specific lie:

  * **Walk-forward**  — fixes "I fitted and tested on the same data."
  * **Purge + embargo** — fixes "my folds overlap, so test data leaked into
    training." (López de Prado, *Advances in Financial ML*, ch. 7.)
  * **Block bootstrap** — fixes "my one equity curve is one draw from a
    distribution I never looked at."

The number to act on is `mc_p95_max_drawdown`, not the mean anything.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ct.backtest.engine import BacktestEngine, CostModel
from ct.backtest.metrics import PerformanceReport, build_report
from ct.models import Bar, ClosedTrade
from ct.risk.engine import RiskLimits
from ct.strategies.base import Strategy

# ── purged / embargoed splits ────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Split:
    train: tuple[int, int]     # [start, end)
    test: tuple[int, int]

    def __str__(self) -> str:
        return (f"train[{self.train[0]}:{self.train[1]}] "
                f"test[{self.test[0]}:{self.test[1]}]")


def purged_walk_forward_splits(
    n: int,
    n_splits: int = 5,
    *,
    train_frac: float = 0.6,
    embargo_pct: float = 0.01,
    anchored: bool = True,
    label_horizon: int = 0,
) -> list[Split]:
    """Anchored (expanding) or rolling walk-forward splits with purge+embargo.

    `label_horizon` is how many bars into the future a training label looks.
    Those trailing bars of the train set overlap the test set and MUST be
    dropped (purged) or the model has literally seen the answer. Default 0 for
    rule-based strategies; set it to the triple-barrier horizon when fitting
    the meta-labeller.

    `embargo_pct` additionally drops a fraction of bars after each test fold,
    because serial correlation means bars just after the test window still
    carry information about it.
    """
    if n_splits < 1 or n < n_splits * 4:
        raise ValueError(f"cannot make {n_splits} splits from {n} bars")
    train_end0 = int(n * train_frac)
    remaining = n - train_end0
    fold = remaining // n_splits
    if fold < 2:
        raise ValueError("test folds would be shorter than 2 bars")

    embargo = max(0, int(n * embargo_pct))
    splits: list[Split] = []
    for k in range(n_splits):
        test_start = train_end0 + k * fold
        test_end = test_start + fold if k < n_splits - 1 else n
        train_start = 0 if anchored else max(0, test_start - train_end0)
        # Purge the label horizon AND the embargo off the end of train.
        train_stop = max(train_start + 1, test_start - label_horizon - embargo)
        splits.append(Split((train_start, train_stop), (test_start, test_end)))
    return splits


# ── parameter search ─────────────────────────────────────────────────────

def param_grid(space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


@dataclass
class WalkForwardResult:
    strategy_name: str
    splits: list[Split]
    chosen_params: list[dict[str, Any]]
    oos_report: PerformanceReport
    per_fold: list[PerformanceReport] = field(default_factory=list)
    n_trials: int = 1
    oos_equity: list[float] = field(default_factory=list)
    oos_trades: list[ClosedTrade] = field(default_factory=list)

    def summary(self) -> str:
        r = self.oos_report
        drift = [f"fold{i}: {p.sharpe:+.2f}" for i, p in enumerate(self.per_fold)]
        return (
            f"{self.strategy_name} — OUT-OF-SAMPLE over {len(self.splits)} folds\n"
            f"  trials evaluated : {self.n_trials}  (feeds the deflated Sharpe)\n"
            f"  return           : {r.total_return:+.1%}   CAGR {r.cagr:+.1%}\n"
            f"  Sharpe / DSR     : {r.sharpe:.2f} / {r.deflated_sharpe:.2f}\n"
            f"  max drawdown     : {r.max_drawdown:.1%} over {r.max_drawdown_duration} bars\n"
            f"  trades / win     : {r.n_trades} / {r.win_rate:.0%}\n"
            f"  fees             : {r.total_fees:,.0f} "
            f"({r.fees_pct_of_gross:.0%} of gross profit)\n"
            f"  benchmark        : {r.benchmark_return:+.1%} "
            f"(beats it: {r.beats_benchmark})\n"
            f"  per-fold Sharpe  : {', '.join(drift)}\n"
            f"  VERDICT          : {r.verdict()}"
        )


def walk_forward(
    strategy_cls: type[Strategy],
    bars: list[Bar],
    # Mapping, not dict: dict is invariant in its value type, so a caller's
    # dict[str, list[Any]] — which is what `parse_space` builds — is not a
    # dict[str, Sequence[Any]]. Both this and `param_grid` only read it.
    space: Mapping[str, Sequence[Any]],
    *,
    limits: RiskLimits | None = None,
    costs: CostModel | None = None,
    timeframe: str = "4h",
    initial_equity: float = 10_000.0,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    objective: Callable[[PerformanceReport], float] | None = None,
) -> WalkForwardResult:
    """Fit params on each train fold, evaluate on the following test fold.

    The out-of-sample equity curve is the concatenation of the test folds only.
    That curve — and nothing else in this module — is what a promotion decision
    may look at.

    Default objective is Calmar, not Sharpe: at retail size the thing that ends
    an account is the drawdown, and Calmar penalises it directly.
    """
    limits = limits or RiskLimits()
    costs = costs or CostModel()
    obj = objective or (lambda r: r.calmar if r.n_trades >= 5 else -1e9)

    grid = param_grid(space)
    if not grid:
        grid = [{}]
    splits = purged_walk_forward_splits(len(bars), n_splits, embargo_pct=embargo_pct)

    chosen: list[dict[str, Any]] = []
    per_fold: list[PerformanceReport] = []
    oos_equity: list[float] = []
    oos_trades: list[ClosedTrade] = []
    equity = initial_equity
    trials = 0
    total_fees = 0.0
    bars_in_market = 0

    for split in splits:
        train = bars[split.train[0]:split.train[1]]
        test = bars[split.test[0]:split.test[1]]

        best_params, best_score = grid[0], -math.inf
        for params in grid:
            try:
                strat = strategy_cls(**params)
                if len(train) < strat.warmup + 2:
                    continue
                res = BacktestEngine(strat, limits, costs,
                                     initial_equity=initial_equity,
                                     timeframe=timeframe).run(train)
                trials += 1
                score = obj(res.report)
                if score > best_score:
                    best_score, best_params = score, params
            except Exception:
                # A parameter set that crashes is a parameter set that loses.
                continue
        chosen.append(best_params)

        strat = strategy_cls(**best_params)
        # Prepend enough history for warm-up, but the equity curve we keep
        # starts at the fold boundary — otherwise warm-up bars would be counted
        # twice across folds.
        warm = max(0, split.test[0] - strat.warmup - 1)
        window = bars[warm:split.test[1]]
        if len(window) < strat.warmup + 2:
            continue
        res = BacktestEngine(strat, limits, costs, initial_equity=equity,
                             timeframe=timeframe).run(window)
        keep = len(window) - len(test)
        fold_equity = res.equity_curve[keep:] if keep < len(res.equity_curve) else []
        if fold_equity:
            oos_equity.extend(fold_equity)
            equity = fold_equity[-1]
        oos_trades.extend(t for t in res.trades if t.exit_ts >= test[0].ts)
        per_fold.append(res.report)
        total_fees += res.report.total_fees
        bars_in_market += int(res.report.exposure * res.report.n_periods)

    bench = [b.close for b in bars[splits[0].test[0]:]]
    bench_eq = ([initial_equity * (p / bench[0]) for p in bench]
                if bench and bench[0] > 0 else None)

    oos_report = build_report(
        oos_equity or [initial_equity], oos_trades, timeframe,
        initial_equity=initial_equity, total_fees=total_fees,
        bars_in_market=bars_in_market, n_trials=max(1, trials),
        benchmark_equity=bench_eq,
    )
    return WalkForwardResult(strategy_cls.name, splits, chosen, oos_report,
                             per_fold, trials, oos_equity, oos_trades)


# ── Monte Carlo ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    n_runs: int
    p05_final_equity: float
    p50_final_equity: float
    p95_final_equity: float
    p95_max_drawdown: float      # THE number to size against
    p50_max_drawdown: float
    prob_of_loss: float
    prob_of_ruin: float          # P(drawdown > ruin_threshold)

    def summary(self, ruin_threshold: float = 0.5) -> str:
        return (
            f"Monte Carlo ({self.n_runs} block-bootstrap paths)\n"
            f"  final equity  p05 {self.p05_final_equity:,.0f} | "
            f"p50 {self.p50_final_equity:,.0f} | p95 {self.p95_final_equity:,.0f}\n"
            f"  max drawdown  p50 {self.p50_max_drawdown:.1%} | "
            f"p95 {self.p95_max_drawdown:.1%}  <- size against this\n"
            f"  P(lose money) {self.prob_of_loss:.0%}   "
            f"P(drawdown > {ruin_threshold:.0%}) {self.prob_of_ruin:.0%}"
        )


def monte_carlo(
    trade_returns: Sequence[float],
    *,
    n_runs: int = 2000,
    block_size: int = 5,
    initial_equity: float = 10_000.0,
    ruin_threshold: float = 0.5,
    seed: int = 20260727,
) -> MonteCarloResult:
    """Stationary block bootstrap over TRADE returns.

    Blocks, not i.i.d. shuffling: trade outcomes are serially correlated (a
    trend strategy wins in clusters), and i.i.d. resampling destroys exactly
    the dependence that produces the deep drawdowns you care about. An i.i.d.
    bootstrap systematically UNDERSTATES max drawdown — the dangerous direction.

    Deterministic seed: a risk number that changes every time you look at it is
    not a risk number.
    """
    rets = [float(r) for r in trade_returns]
    if len(rets) < 5:
        raise ValueError(f"need >=5 trades to bootstrap, got {len(rets)}")

    rng = random.Random(seed)
    n = len(rets)
    finals, mdds = [], []

    for _ in range(n_runs):
        path, i = [], 0
        while i < n:
            start = rng.randrange(n)
            for j in range(min(block_size, n - i)):
                path.append(rets[(start + j) % n])
            i += block_size

        eq, peak, mdd = initial_equity, initial_equity, 0.0
        for r in path[:n]:
            eq *= (1.0 + r)
            if eq <= 0:
                eq, mdd = 1e-9, 1.0
                break
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        finals.append(eq)
        mdds.append(mdd)

    f = np.array(finals)
    d = np.array(mdds)
    return MonteCarloResult(
        n_runs=n_runs,
        p05_final_equity=float(np.percentile(f, 5)),
        p50_final_equity=float(np.percentile(f, 50)),
        p95_final_equity=float(np.percentile(f, 95)),
        p95_max_drawdown=float(np.percentile(d, 95)),
        p50_max_drawdown=float(np.percentile(d, 50)),
        prob_of_loss=float((f < initial_equity).mean()),
        prob_of_ruin=float((d > ruin_threshold).mean()),
    )


__all__ = [
    "MonteCarloResult",
    "Split",
    "WalkForwardResult",
    "monte_carlo",
    "param_grid",
    "purged_walk_forward_splits",
    "walk_forward",
]
