"""Event-driven signal system.

Signals are triggered by STATE CHANGES, not periodic evaluation.
Each signal source maintains its own state and only emits when state transitions.

Signal Sources (Technical Indicators):
- MA Crossover: Price crosses above/below SMA60
- RSI Extremes: RSI enters/exits overbought/oversold zones
- MACD Crossover: Golden cross / Death cross
- Bollinger Breakout: Price breaks above/below bands

Signal Sources (Market Data):
- Funding Rate: Significant funding rate changes
- Long/Short Ratio: L/S ratio changes >5% in 5 minutes
- Open Interest: OI changes >5% in 5 minutes
- Liquidation: Large liquidations >1M USD

Signal Sources (Account/Risk):
- P&L Change: User P&L changes by ±5%
- Risk Signal: Risk module triggers (stop loss, trailing stop, take profit)
- Order Status: Open orders filled/cancelled

Timeframes:
- 5m, 15m, 1h, 4h for different signal granularity
"""

from ai_trading_team.strategy.signals.aggregator import SignalAggregator, SignalWindow
from ai_trading_team.strategy.signals.base import SignalSource
from ai_trading_team.strategy.signals.bollinger_breakout import BollingerBreakoutSignal
from ai_trading_team.strategy.signals.funding_rate import FundingRateSignal
from ai_trading_team.strategy.signals.liquidation import LiquidationSignal
from ai_trading_team.strategy.signals.ls_ratio import LongShortRatioSignal
from ai_trading_team.strategy.signals.ma_crossover import MACrossoverSignal
from ai_trading_team.strategy.signals.macd_crossover import MACDCrossoverSignal
from ai_trading_team.strategy.signals.open_interest import OpenInterestSignal
from ai_trading_team.strategy.signals.order_status import OrderStatusSignal
from ai_trading_team.strategy.signals.pnl_change import PnLChangeSignal
from ai_trading_team.strategy.signals.risk_signal import RiskSignal
from ai_trading_team.strategy.signals.rsi_extreme import RSIExtremeSignal
from ai_trading_team.strategy.signals.types import (
    ALL_TIMEFRAMES,
    Signal,
    SignalDirection,
    SignalStrength,
    SignalType,
    Timeframe,
)

__all__ = [
    # Types
    "Signal",
    "SignalDirection",
    "SignalStrength",
    "SignalType",
    "Timeframe",
    "ALL_TIMEFRAMES",
    # Base
    "SignalSource",
    # Technical Indicator Sources
    "MACrossoverSignal",
    "RSIExtremeSignal",
    "MACDCrossoverSignal",
    "BollingerBreakoutSignal",
    # Market Data Sources
    "FundingRateSignal",
    "LongShortRatioSignal",
    "OpenInterestSignal",
    "LiquidationSignal",
    # Account/Risk Sources
    "PnLChangeSignal",
    "RiskSignal",
    "OrderStatusSignal",
    # Aggregator
    "SignalAggregator",
    "SignalWindow",
]
