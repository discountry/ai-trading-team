"""Funding Rate Strategy.

According to the trading philosophy:
- Positive funding rate -> lean short (longs pay shorts)
- Negative funding rate -> lean long (shorts pay longs)
"""

import logging
from decimal import Decimal
from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy

logger = logging.getLogger(__name__)


class FundingRateStrategy(Strategy):
    """Funding Rate Strategy.

    Generates signals based on funding rate:
    - FUNDING_POSITIVE: Positive funding -> bearish bias (lean short)
    - FUNDING_NEGATIVE: Negative funding -> bullish bias (lean long)
    - FUNDING_EXTREME_*: Very high/low funding indicates strong sentiment
    """

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        normal_threshold: Decimal = Decimal("0.0001"),  # 0.01%
        extreme_threshold: Decimal = Decimal("0.001"),  # 0.1%
    ) -> None:
        """Initialize Funding Rate Strategy.

        Args:
            data_pool: Shared data pool
            signal_queue: Signal queue for emitting signals
            normal_threshold: Threshold for generating normal signals
            extreme_threshold: Threshold for extreme funding signals
        """
        super().__init__("funding_rate", data_pool, signal_queue)
        self._normal_threshold = normal_threshold
        self._extreme_threshold = extreme_threshold

        # Track previous state to detect changes
        self._prev_signal_type: SignalType | None = None

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check funding rate conditions.

        Args:
            snapshot: Current data snapshot

        Returns:
            Signal if funding rate is significant, None otherwise
        """
        if not snapshot.funding_rate:
            return None

        funding_rate = Decimal(str(snapshot.funding_rate.get("funding_rate", 0)))

        # Determine signal type
        current_signal_type: SignalType | None = None
        bias = "neutral"
        suggested_side = None

        if funding_rate >= self._extreme_threshold:
            current_signal_type = SignalType.FUNDING_EXTREME_POSITIVE
            bias = "strongly_bearish"
            suggested_side = "short"
        elif funding_rate >= self._normal_threshold:
            current_signal_type = SignalType.FUNDING_POSITIVE
            bias = "bearish"
            suggested_side = "short"
        elif funding_rate <= -self._extreme_threshold:
            current_signal_type = SignalType.FUNDING_EXTREME_NEGATIVE
            bias = "strongly_bullish"
            suggested_side = "long"
        elif funding_rate <= -self._normal_threshold:
            current_signal_type = SignalType.FUNDING_NEGATIVE
            bias = "bullish"
            suggested_side = "long"

        signal: StrategySignal | None = None

        # Only emit signal when state changes
        if current_signal_type and current_signal_type != self._prev_signal_type:
            logger.info(
                f"Funding rate signal: {current_signal_type.value}, "
                f"rate={float(funding_rate) * 100:.4f}%"
            )
            signal = StrategySignal(
                signal_type=current_signal_type,
                data={
                    "funding_rate": float(funding_rate),
                    "funding_rate_percent": float(funding_rate) * 100,
                    "bias": bias,
                    "suggested_side": suggested_side,
                    "next_funding_time": snapshot.funding_rate.get("funding_time"),
                },
                priority=1,  # Lower priority - supporting signal
            )

        self._prev_signal_type = current_signal_type

        # Update indicator
        self._data_pool.update_indicator(
            "funding_rate",
            {
                "rate": float(funding_rate),
                "rate_percent": float(funding_rate) * 100,
                "bias": bias,
                "is_extreme": abs(funding_rate) >= self._extreme_threshold,
            },
        )

        return signal

    def get_current_bias(self, snapshot: DataSnapshot) -> str:
        """Get current bias based on funding rate.

        Args:
            snapshot: Current data snapshot

        Returns:
            "bullish", "bearish", or "neutral"
        """
        if not snapshot.funding_rate:
            return "neutral"

        funding_rate = Decimal(str(snapshot.funding_rate.get("funding_rate", 0)))

        if funding_rate >= self._normal_threshold:
            return "bearish"
        elif funding_rate <= -self._normal_threshold:
            return "bullish"
        return "neutral"

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self._name,
            "normal_threshold": float(self._normal_threshold),
            "extreme_threshold": float(self._extreme_threshold),
            "enabled": self._enabled,
        }
