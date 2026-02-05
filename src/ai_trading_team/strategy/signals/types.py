"""Signal type definitions for event-driven signal system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Timeframe(str, Enum):
    """Trading timeframes for signals."""

    M15 = "15m"
    H1 = "1h"
    H4 = "4h"

    @property
    def minutes(self) -> int:
        """Get timeframe duration in minutes."""
        mapping = {"15m": 15, "1h": 60, "4h": 240}
        return mapping[self.value]


# All timeframes for convenience
ALL_TIMEFRAMES = [Timeframe.M15, Timeframe.H1, Timeframe.H4]


class SignalType(str, Enum):
    """Types of trading signals based on specific events."""

    # MA Crossover signals (price crosses SMA60)
    MA_CROSS_UP = "ma_cross_up"  # Price crosses above MA
    MA_CROSS_DOWN = "ma_cross_down"  # Price crosses below MA

    # RSI signals (enters/exits overbought/oversold zones)
    RSI_ENTER_OVERSOLD = "rsi_enter_oversold"  # RSI enters oversold zone (<30)
    RSI_EXIT_OVERSOLD = "rsi_exit_oversold"  # RSI exits oversold zone (bullish)
    RSI_ENTER_OVERBOUGHT = "rsi_enter_overbought"  # RSI enters overbought zone (>70)
    RSI_EXIT_OVERBOUGHT = "rsi_exit_overbought"  # RSI exits overbought zone (bearish)

    # MACD signals (golden cross / death cross)
    MACD_GOLDEN_CROSS = "macd_golden_cross"  # MACD line crosses above signal line
    MACD_DEATH_CROSS = "macd_death_cross"  # MACD line crosses below signal line

    # Bollinger Bands signals (breakout)
    BB_BREAK_UPPER = "bb_break_upper"  # Price breaks above upper band
    BB_BREAK_LOWER = "bb_break_lower"  # Price breaks below lower band
    BB_RETURN_UPPER = "bb_return_upper"  # Price returns from above upper band
    BB_RETURN_LOWER = "bb_return_lower"  # Price returns from above lower band

    # Funding rate signals
    FUNDING_SPIKE_POSITIVE = "funding_spike_positive"  # High positive funding (bearish)
    FUNDING_SPIKE_NEGATIVE = "funding_spike_negative"  # High negative funding (bullish)
    FUNDING_NORMALIZE = "funding_normalize"  # Funding returns to normal

    # Long/Short ratio signals (5-minute change > 5%)
    LS_RATIO_SURGE = "ls_ratio_surge"  # L/S ratio increased >5% in 5min
    LS_RATIO_DROP = "ls_ratio_drop"  # L/S ratio decreased >5% in 5min

    # Open Interest signals (5-minute change > 5%)
    OI_SURGE = "oi_surge"  # OI increased >5% in 5min
    OI_DROP = "oi_drop"  # OI decreased >5% in 5min

    # Liquidation signals (>1M USD)
    LIQUIDATION_LONG = "liquidation_long"  # Large long liquidation
    LIQUIDATION_SHORT = "liquidation_short"  # Large short liquidation

    # P&L signals (user profit/loss change ±5%)
    PNL_PROFIT_INCREASE = "pnl_profit_increase"  # P&L increased by 5%
    PNL_PROFIT_DECREASE = "pnl_profit_decrease"  # P&L decreased by 5%

    # Risk signals
    RISK_FORCE_STOP_LOSS = "risk_force_stop_loss"  # Risk module triggered stop loss
    RISK_TRAILING_STOP = "risk_trailing_stop"  # Trailing stop triggered
    RISK_TAKE_PROFIT = "risk_take_profit"  # Take profit triggered

    # Order status signals
    ORDER_FILLED = "order_filled"  # Order was filled
    ORDER_CANCELLED = "order_cancelled"  # Order was cancelled
    ORDER_PARTIAL_FILL = "order_partial_fill"  # Order partially filled

    # Composite signals (from aggregator)
    BULLISH_CONFLUENCE = "bullish_confluence"  # Multiple bullish signals align
    BEARISH_CONFLUENCE = "bearish_confluence"  # Multiple bearish signals align


class SignalDirection(str, Enum):
    """Signal direction bias."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(str, Enum):
    """Signal strength classification."""

    STRONG = "strong"  # High confidence
    MODERATE = "moderate"  # Medium confidence
    WEAK = "weak"  # Low confidence, informational


@dataclass
class Signal:
    """Represents a trading signal triggered by a state change.

    Signals are immutable events - once created, they represent
    a point-in-time state transition.
    """

    signal_type: SignalType
    direction: SignalDirection
    strength: SignalStrength
    timeframe: Timeframe
    timestamp: datetime = field(default_factory=datetime.now)

    # Source identification
    source: str = ""  # e.g., "ma_crossover", "rsi_extreme"

    # Event-specific data
    data: dict[str, Any] = field(default_factory=dict)

    # Human-readable description
    description: str = ""

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction == SignalDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction == SignalDirection.BEARISH

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough for action."""
        return self.strength in (SignalStrength.STRONG, SignalStrength.MODERATE)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "description": self.description,
        }
