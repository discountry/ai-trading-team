"""Single-factor strategy implementations."""

from ai_trading_team.strategy.factors.funding_rate import FundingRateStrategy
from ai_trading_team.strategy.factors.long_short_ratio import LongShortRatioStrategy
from ai_trading_team.strategy.factors.ma_crossover import MACrossoverStrategy
from ai_trading_team.strategy.factors.ma_position import MAPositionStrategy
from ai_trading_team.strategy.factors.macd_cross import MACDCrossStrategy
from ai_trading_team.strategy.factors.price_level import PriceLevelStrategy
from ai_trading_team.strategy.factors.rsi_oversold import RSIOversoldStrategy
from ai_trading_team.strategy.factors.volatility import VolatilityStrategy

__all__ = [
    "FundingRateStrategy",
    "LongShortRatioStrategy",
    "MACrossoverStrategy",
    "MAPositionStrategy",
    "MACDCrossStrategy",
    "PriceLevelStrategy",
    "RSIOversoldStrategy",
    "VolatilityStrategy",
]
