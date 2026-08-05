from ct.strategies.breakout import AtrSqueezeBreakoutStrategy, DonchianTrendStrategy
from ct.strategies.reversion import BollingerReversionStrategy, ZScoreReversionStrategy
from ct.strategies.trend import EmaCrossStrategy, MacdMomentumStrategy, SupertrendStrategy

REGISTRY = {
    s.name: s
    for s in (
        EmaCrossStrategy, SupertrendStrategy, MacdMomentumStrategy,
        BollingerReversionStrategy, ZScoreReversionStrategy,
        DonchianTrendStrategy, AtrSqueezeBreakoutStrategy,
    )
}

__all__ = ["REGISTRY", *REGISTRY]
