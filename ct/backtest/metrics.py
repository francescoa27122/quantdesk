"""Performance metrics.

Two things here are easy to get wrong and both are load-bearing:

1. **Annualisation uses 365, not 252.** Crypto trades every day. Using the
   equity convention inflates Sharpe by sqrt(365/252) = 1.20 — a 1.0 Sharpe
   becomes 1.2, which is the difference between "don't trade this" and "passes
   the promotion gate".

2. **A raw Sharpe from a search over many parameter sets is not a Sharpe.**
   If you try 200 variants, the best one has a high Sharpe *by construction*.
   `deflated_sharpe_ratio` corrects for exactly that, and it is the number the
   promotion gate should be reading. This is Bailey & López de Prado (2014),
   "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

import numpy as np

from ct.models import ClosedTrade
from ct.risk.sizing import periods_per_year

EULER_MASCHERONI = 0.5772156649015329


# ── normal distribution helpers (no scipy: keeps the image small) ────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF — Acklam's rational approximation.

    Accurate to ~1.15e-9 in the relative error, which is far more than enough
    for the DSR. Avoids a scipy dependency (~90 MB in the image) for one call.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── core metrics ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PerformanceReport:
    n_periods: int
    n_trades: int
    total_return: float
    cagr: float
    ann_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_bars_held: float
    exposure: float
    turnover: float
    total_fees: float
    net_profit: float
    fees_pct_of_gross: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    benchmark_return: float = 0.0
    benchmark_sharpe: float = 0.0
    beats_benchmark: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        """One honest line. This is what goes in the Telegram summary."""
        if self.n_trades < 30:
            return (f"INSUFFICIENT DATA — {self.n_trades} trades. No metric "
                    f"below is meaningful yet.")
        if self.deflated_sharpe < 0.95:
            return (f"LIKELY NOISE — deflated Sharpe {self.deflated_sharpe:.2f} "
                    f"(<0.95). The raw Sharpe of {self.sharpe:.2f} is not "
                    f"distinguishable from a lucky draw across the variants tried.")
        if not self.beats_benchmark:
            return (f"DOES NOT BEAT BUY-AND-HOLD. Strategy {self.total_return:+.1%} "
                    f"vs benchmark {self.benchmark_return:+.1%}. Do not trade this.")
        if self.fees_pct_of_gross > 0.5:
            return (f"FEE-DOMINATED — fees are {self.fees_pct_of_gross:.0%} of "
                    f"gross profit. Reduce trade frequency before going further.")
        return (f"PASSES SCREENING — DSR {self.deflated_sharpe:.2f}, "
                f"Sharpe {self.sharpe:.2f}, MaxDD {self.max_drawdown:.1%}. "
                f"Still requires the paper-trading gate.")


def _returns_from_equity(equity: Sequence[float]) -> np.ndarray:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return np.array([], dtype=np.float64)
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(prev > 0, eq[1:] / prev - 1.0, 0.0)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def sharpe_ratio(returns: np.ndarray, timeframe: str, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    ppy = periods_per_year(timeframe)
    excess = returns - rf / ppy
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * math.sqrt(ppy))


def sortino_ratio(returns: np.ndarray, timeframe: str, rf: float = 0.0) -> float:
    """Downside deviation uses the FULL sample in the denominator, not just the
    losing periods. Dividing by the count of negative periods only is a common
    error that inflates Sortino by ~sqrt(2) for a symmetric distribution."""
    if len(returns) < 2:
        return 0.0
    ppy = periods_per_year(timeframe)
    excess = returns - rf / ppy
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float((downside ** 2).sum()) / len(excess))
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * math.sqrt(ppy))


def max_drawdown(equity: Sequence[float]) -> tuple[float, int]:
    """Return (max drawdown as a positive fraction, duration in periods)."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
    mdd = float(np.nanmax(dd)) if len(dd) else 0.0

    longest = cur = 0
    for i in range(len(eq)):
        if eq[i] < peak[i]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return mdd, longest


def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_sr: float = 0.0
) -> float:
    """P(true Sharpe > benchmark_sr), correcting for skew and kurtosis.

    Note both Sharpes here are PER-PERIOD (non-annualised) — mixing an
    annualised observed SR with a per-period benchmark is the standard way to
    get a nonsense answer out of this formula.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = returns.mean() / sd
    z = (returns - returns.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())          # raw (non-excess) kurtosis
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return 0.0
    return _norm_cdf((sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] across `n_trials` independent trials with zero true skill.

    This is the bar a searched-over strategy must clear just to be interesting.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    sd = math.sqrt(sr_variance)
    g = EULER_MASCHERONI
    return sd * ((1 - g) * _norm_ppf(1 - 1.0 / n_trials)
                 + g * _norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(
    returns: np.ndarray, n_trials: int = 1, trial_sr_variance: float | None = None
) -> float:
    """PSR against the expected max Sharpe of `n_trials` noise strategies.

    Interpretation: the probability that the strategy's true Sharpe exceeds
    what you'd expect from the luckiest of `n_trials` skill-free attempts.
    Below ~0.95, you have found noise. Feed it the HONEST trial count — every
    parameter set you evaluated, not just the ones you kept.
    """
    if len(returns) < 3:
        return 0.0
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(returns, 0.0)
    if trial_sr_variance is None:
        # Under the null, SR_hat ~ N(0, 1/(n-1)) per period.
        trial_sr_variance = 1.0 / max(1, len(returns) - 1)
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials,
                                                                  trial_sr_variance))


# ── the report builder ───────────────────────────────────────────────────

def build_report(
    equity: Sequence[float],
    trades: Sequence[ClosedTrade],
    timeframe: str,
    *,
    initial_equity: float,
    total_fees: float = 0.0,
    bars_in_market: int = 0,
    n_trials: int = 1,
    benchmark_equity: Sequence[float] | None = None,
    rf: float = 0.0,
) -> PerformanceReport:
    eq = list(equity)
    rets = _returns_from_equity(eq)
    n = len(eq)
    ppy = periods_per_year(timeframe)

    final = eq[-1] if eq else initial_equity
    total_return = (final / initial_equity - 1.0) if initial_equity > 0 else 0.0
    years = n / ppy if ppy > 0 else 0.0
    cagr = ((final / initial_equity) ** (1 / years) - 1.0) \
        if years > 0 and initial_equity > 0 and final > 0 else 0.0

    mdd, mdd_dur = max_drawdown(eq)
    sharpe = sharpe_ratio(rets, timeframe, rf)
    ann_vol = float(rets.std(ddof=1) * math.sqrt(ppy)) if len(rets) > 1 else 0.0

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_win = float(sum(t.net_pnl for t in wins))
    gross_loss = abs(float(sum(t.net_pnl for t in losses)))
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
    # Profit factor of inf (zero losses) is not a strength signal, it is a
    # sample-size signal. Reported as 0 so it cannot rank first in a sweep.
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    net_profit = float(final - initial_equity)
    gross_profit = net_profit + total_fees
    fees_pct = (total_fees / gross_profit) if gross_profit > 0 else (
        1.0 if total_fees > 0 else 0.0)

    turnover = float(sum(abs(t.qty * t.entry_price) for t in trades)) / initial_equity \
        if initial_equity > 0 else 0.0

    bench_ret = bench_sharpe = 0.0
    beats = False
    if benchmark_equity is not None and len(benchmark_equity) > 1:
        b = list(benchmark_equity)
        bench_ret = b[-1] / b[0] - 1.0 if b[0] > 0 else 0.0
        bench_sharpe = sharpe_ratio(_returns_from_equity(b), timeframe, rf)
        b_mdd, _ = max_drawdown(b)
        strat_calmar = cagr / mdd if mdd > 0 else 0.0
        b_years = len(b) / ppy if ppy > 0 else 0.0
        b_cagr = ((b[-1] / b[0]) ** (1 / b_years) - 1.0) \
            if b_years > 0 and b[0] > 0 and b[-1] > 0 else 0.0
        bench_calmar = b_cagr / b_mdd if b_mdd > 0 else 0.0
        beats = strat_calmar > bench_calmar

    return PerformanceReport(
        n_periods=n,
        n_trades=len(trades),
        total_return=total_return,
        cagr=cagr,
        ann_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino_ratio(rets, timeframe, rf),
        calmar=(cagr / mdd) if mdd > 0 else 0.0,
        max_drawdown=mdd,
        max_drawdown_duration=mdd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        payoff_ratio=payoff,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=float(max((t.net_pnl for t in wins), default=Decimal(0))),
        largest_loss=float(min((t.net_pnl for t in losses), default=Decimal(0))),
        avg_bars_held=(sum(t.bars_held for t in trades) / len(trades)) if trades else 0.0,
        exposure=(bars_in_market / n) if n else 0.0,
        turnover=turnover,
        total_fees=total_fees,
        net_profit=net_profit,
        fees_pct_of_gross=fees_pct,
        probabilistic_sharpe=probabilistic_sharpe_ratio(rets, 0.0),
        deflated_sharpe=deflated_sharpe_ratio(rets, n_trials),
        benchmark_return=bench_ret,
        benchmark_sharpe=bench_sharpe,
        beats_benchmark=beats,
    )


__all__ = [
    "PerformanceReport",
    "build_report",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sortino_ratio",
]
