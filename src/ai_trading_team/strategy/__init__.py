"""Strategy module - mechanical trading strategies and event-driven signals."""

# Re-export old types from signal_queue for backward compatibility
from ai_trading_team.core.signal_queue import SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy

# Export new event-driven signal system
from ai_trading_team.strategy.signals import (
    FundingRateSignal,
    LongShortRatioSignal,
    MACrossoverSignal,
    RSIExtremeSignal,
    Signal,
    SignalAggregator,
    SignalDirection,
    SignalSource,
    SignalStrength,
    SignalWindow,
    Timeframe,
)

__all__ = [
    # Base
    "Strategy",
    # Old signal types (backward compatibility)
    "SignalType",
    "StrategySignal",
    # New event-driven signal system
    "Signal",
    "SignalAggregator",
    "SignalDirection",
    "SignalSource",
    "SignalStrength",
    "SignalWindow",
    "Timeframe",
    "MACrossoverSignal",
    "RSIExtremeSignal",
    "FundingRateSignal",
    "LongShortRatioSignal",
]
