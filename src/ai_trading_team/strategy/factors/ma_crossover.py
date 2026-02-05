"""Moving Average Crossover Strategy.

A simple mechanical strategy that generates signals when
a short-term moving average crosses over a long-term moving average.
"""

import logging
from typing import Any

from talipp.indicators import EMA, SMA

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy

logger = logging.getLogger(__name__)


class MACrossoverStrategy(Strategy):
    """Moving Average Crossover Strategy.

    Generates:
    - MA_BULLISH_CROSS when short MA crosses above long MA (golden cross)
    - MA_BEARISH_CROSS when short MA crosses below long MA (death cross)
    """

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        short_period: int = 7,
        long_period: int = 25,
        ma_type: str = "ema",
        kline_interval: str = "1m",
    ) -> None:
        """Initialize MA Crossover Strategy.

        Args:
            data_pool: Shared data pool
            signal_queue: Signal queue for emitting signals
            short_period: Short MA period (default 7)
            long_period: Long MA period (default 25)
            ma_type: MA type - "sma" or "ema" (default "ema")
            kline_interval: Kline interval to use (default "1m")
        """
        super().__init__("ma_crossover", data_pool, signal_queue)
        self._short_period = short_period
        self._long_period = long_period
        self._ma_type = ma_type.lower()
        self._kline_interval = kline_interval

        # Previous MA values for crossover detection
        self._prev_short_ma: float | None = None
        self._prev_long_ma: float | None = None

    def _calculate_ma(self, closes: list[float], period: int) -> float | None:
        """Calculate moving average using talipp.

        Args:
            closes: List of close prices
            period: MA period

        Returns:
            MA value or None if not enough data
        """
        if len(closes) < period:
            return None

        ma = SMA(period, closes) if self._ma_type == "sma" else EMA(period, closes)

        return float(ma[-1]) if ma and ma[-1] is not None else None

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check for MA crossover conditions.

        Args:
            snapshot: Current data snapshot

        Returns:
            Signal if crossover detected, None otherwise
        """
        klines = snapshot.klines.get(self._kline_interval, []) if snapshot.klines else []

        if len(klines) < self._long_period + 1:
            logger.debug(f"Not enough klines: {len(klines)} < {self._long_period + 1}")
            return None

        # Extract close prices
        closes = [float(k.get("close", 0)) for k in klines]

        # Calculate current MAs
        short_ma = self._calculate_ma(closes, self._short_period)
        long_ma = self._calculate_ma(closes, self._long_period)

        if short_ma is None or long_ma is None:
            return None

        signal: StrategySignal | None = None

        # Detect crossover
        if self._prev_short_ma is not None and self._prev_long_ma is not None:
            # Golden Cross: short MA crosses above long MA
            if self._prev_short_ma <= self._prev_long_ma and short_ma > long_ma:
                logger.info(
                    f"MA Bullish Cross detected: "
                    f"short={short_ma:.2f} crossed above long={long_ma:.2f}"
                )
                signal = StrategySignal(
                    signal_type=SignalType.MA_BULLISH_CROSS,
                    data={
                        "short_ma": short_ma,
                        "long_ma": long_ma,
                        "short_period": self._short_period,
                        "long_period": self._long_period,
                        "ma_type": self._ma_type,
                        "current_price": closes[-1] if closes else None,
                    },
                    priority=1,
                )

            # Death Cross: short MA crosses below long MA
            elif self._prev_short_ma >= self._prev_long_ma and short_ma < long_ma:
                logger.info(
                    f"MA Bearish Cross detected: "
                    f"short={short_ma:.2f} crossed below long={long_ma:.2f}"
                )
                signal = StrategySignal(
                    signal_type=SignalType.MA_BEARISH_CROSS,
                    data={
                        "short_ma": short_ma,
                        "long_ma": long_ma,
                        "short_period": self._short_period,
                        "long_period": self._long_period,
                        "ma_type": self._ma_type,
                        "current_price": closes[-1] if closes else None,
                    },
                    priority=1,
                )

        # Update previous values
        self._prev_short_ma = short_ma
        self._prev_long_ma = long_ma

        # Update indicators in data pool for context
        self._data_pool.update_indicator(
            f"ma_short_{self._short_period}",
            short_ma,
        )
        self._data_pool.update_indicator(
            f"ma_long_{self._long_period}",
            long_ma,
        )

        return signal

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self._name,
            "short_period": self._short_period,
            "long_period": self._long_period,
            "ma_type": self._ma_type,
            "kline_interval": self._kline_interval,
            "enabled": self._enabled,
        }
