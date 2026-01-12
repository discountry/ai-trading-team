"""MA Position Strategy - Price relative to Moving Average.

According to the trading philosophy:
- Price above 1H MA60 -> bullish (long bias)
- Price below 1H MA60 -> bearish (short bias)
"""

import logging
from typing import Any

from talipp.indicators import SMA

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy

logger = logging.getLogger(__name__)


class MAPositionStrategy(Strategy):
    """Moving Average Position Strategy.

    Generates signals based on price position relative to MA60 on 1H timeframe.
    - PRICE_ABOVE_MA: Price crossed above MA -> bullish
    - PRICE_BELOW_MA: Price crossed below MA -> bearish
    """

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        ma_period: int = 60,
        kline_interval: str = "1h",
    ) -> None:
        """Initialize MA Position Strategy.

        Args:
            data_pool: Shared data pool
            signal_queue: Signal queue for emitting signals
            ma_period: MA period (default 60 for MA60)
            kline_interval: Kline interval (default "1h" for 1-hour)
        """
        super().__init__("ma_position", data_pool, signal_queue)
        self._ma_period = ma_period
        self._kline_interval = kline_interval

        # Track previous position relative to MA
        self._prev_above_ma: bool | None = None

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check if price crossed above/below MA.

        Args:
            snapshot: Current data snapshot

        Returns:
            Signal if price crossed MA, None otherwise
        """
        klines = snapshot.klines.get(self._kline_interval, []) if snapshot.klines else []

        if len(klines) < self._ma_period + 1:
            logger.debug(
                f"Not enough klines for MA{self._ma_period}: {len(klines)} < {self._ma_period + 1}"
            )
            return None

        # Extract close prices
        closes = [float(k.get("close", 0)) for k in klines]
        current_close = closes[-1]

        # Calculate MA using talipp
        ma = SMA(self._ma_period, closes)
        if not ma or ma[-1] is None:
            return None

        ma_value = float(ma[-1])
        current_above_ma = current_close > ma_value

        signal: StrategySignal | None = None

        # Detect crossover
        if self._prev_above_ma is not None and self._prev_above_ma != current_above_ma:
            if current_above_ma:
                # Price crossed above MA -> bullish
                logger.info(
                    f"Price crossed above MA{self._ma_period}: "
                    f"close={current_close:.4f} > MA={ma_value:.4f}"
                )
                signal = StrategySignal(
                    signal_type=SignalType.PRICE_ABOVE_MA,
                    data={
                        "current_price": current_close,
                        "ma_value": ma_value,
                        "ma_period": self._ma_period,
                        "interval": self._kline_interval,
                        "bias": "bullish",
                        "suggested_side": "long",
                    },
                    priority=2,  # Higher priority as this is a key signal
                )
            else:
                # Price crossed below MA -> bearish
                logger.info(
                    f"Price crossed below MA{self._ma_period}: "
                    f"close={current_close:.4f} < MA={ma_value:.4f}"
                )
                signal = StrategySignal(
                    signal_type=SignalType.PRICE_BELOW_MA,
                    data={
                        "current_price": current_close,
                        "ma_value": ma_value,
                        "ma_period": self._ma_period,
                        "interval": self._kline_interval,
                        "bias": "bearish",
                        "suggested_side": "short",
                    },
                    priority=2,
                )

        # Update state
        self._prev_above_ma = current_above_ma

        # Always update indicator in data pool
        self._data_pool.update_indicator(
            f"MA{self._ma_period}_{self._kline_interval}",
            {
                "value": ma_value,
                "price_above": current_above_ma,
                "distance_percent": ((current_close - ma_value) / ma_value) * 100,
            },
        )

        return signal

    def get_current_bias(self, snapshot: DataSnapshot) -> str:
        """Get current market bias based on MA position.

        Args:
            snapshot: Current data snapshot

        Returns:
            "bullish", "bearish", or "neutral"
        """
        klines = snapshot.klines.get(self._kline_interval, []) if snapshot.klines else []

        if len(klines) < self._ma_period:
            return "neutral"

        closes = [float(k.get("close", 0)) for k in klines]
        current_close = closes[-1]

        ma = SMA(self._ma_period, closes)
        if not ma or ma[-1] is None:
            return "neutral"

        return "bullish" if current_close > float(ma[-1]) else "bearish"

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self._name,
            "ma_period": self._ma_period,
            "kline_interval": self._kline_interval,
            "enabled": self._enabled,
        }
