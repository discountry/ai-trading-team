"""Long/Short Ratio Strategy.

According to the trading philosophy:
- More longs -> lean short (contrarian)
- More shorts -> lean long (contrarian)
"""

import logging
from decimal import Decimal
from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy

logger = logging.getLogger(__name__)


class LongShortRatioStrategy(Strategy):
    """Long/Short Ratio Strategy.

    Contrarian strategy based on trader positioning:
    - LONGS_DOMINANT: More traders are long -> bearish (lean short)
    - SHORTS_DOMINANT: More traders are short -> bullish (lean long)
    """

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        long_dominant_threshold: Decimal = Decimal("0.55"),  # 55% longs
        short_dominant_threshold: Decimal = Decimal("0.45"),  # 45% longs (55% shorts)
    ) -> None:
        """Initialize Long/Short Ratio Strategy.

        Args:
            data_pool: Shared data pool
            signal_queue: Signal queue for emitting signals
            long_dominant_threshold: Threshold for longs dominant signal
            short_dominant_threshold: Threshold for shorts dominant signal
        """
        super().__init__("long_short_ratio", data_pool, signal_queue)
        self._long_threshold = long_dominant_threshold
        self._short_threshold = short_dominant_threshold

        self._prev_signal_type: SignalType | None = None

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check long/short ratio conditions.

        Args:
            snapshot: Current data snapshot

        Returns:
            Signal if ratio is significant, None otherwise
        """
        if not snapshot.long_short_ratio:
            return None

        long_ratio = Decimal(str(snapshot.long_short_ratio.get("long_ratio", "0.5")))
        short_ratio = Decimal(str(snapshot.long_short_ratio.get("short_ratio", "0.5")))

        current_signal_type: SignalType | None = None
        bias = "neutral"
        suggested_side = None

        if long_ratio >= self._long_threshold:
            # More longs than shorts -> contrarian bearish
            current_signal_type = SignalType.LONGS_DOMINANT
            bias = "bearish"
            suggested_side = "short"
        elif long_ratio <= self._short_threshold:
            # More shorts than longs -> contrarian bullish
            current_signal_type = SignalType.SHORTS_DOMINANT
            bias = "bullish"
            suggested_side = "long"

        signal: StrategySignal | None = None

        # Only emit signal when state changes
        if current_signal_type and current_signal_type != self._prev_signal_type:
            logger.info(
                f"Long/Short ratio signal: {current_signal_type.value}, "
                f"long={float(long_ratio) * 100:.1f}%, short={float(short_ratio) * 100:.1f}%"
            )
            signal = StrategySignal(
                signal_type=current_signal_type,
                data={
                    "long_ratio": float(long_ratio),
                    "short_ratio": float(short_ratio),
                    "long_short_ratio": float(long_ratio / short_ratio) if short_ratio > 0 else 0,
                    "bias": bias,
                    "suggested_side": suggested_side,
                },
                priority=1,
            )

        self._prev_signal_type = current_signal_type

        # Update indicator
        self._data_pool.update_indicator(
            "long_short_ratio",
            {
                "long_ratio": float(long_ratio),
                "short_ratio": float(short_ratio),
                "bias": bias,
                "is_extreme": (
                    long_ratio >= self._long_threshold + Decimal("0.1")
                    or long_ratio <= self._short_threshold - Decimal("0.1")
                ),
            },
        )

        return signal

    def get_current_bias(self, snapshot: DataSnapshot) -> str:
        """Get current bias based on long/short ratio."""
        if not snapshot.long_short_ratio:
            return "neutral"

        long_ratio = Decimal(str(snapshot.long_short_ratio.get("long_ratio", "0.5")))

        if long_ratio >= self._long_threshold:
            return "bearish"
        elif long_ratio <= self._short_threshold:
            return "bullish"
        return "neutral"

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self._name,
            "long_dominant_threshold": float(self._long_threshold),
            "short_dominant_threshold": float(self._short_threshold),
            "enabled": self._enabled,
        }
