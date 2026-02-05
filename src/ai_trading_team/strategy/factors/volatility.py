"""Volatility Strategy using ATR and Bollinger Bands.

According to the trading philosophy:
- Low volatility -> avoid trading (narrow range can lead to repeated stop-losses)
- High volatility -> opportunities for trend trading
- Bollinger Band squeeze -> potential breakout coming
"""

import logging
from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy

logger = logging.getLogger(__name__)


class VolatilityStrategy(Strategy):
    """Volatility Strategy using ATR and Bollinger Bands.

    Generates signals based on market volatility:
    - VOLATILITY_LOW: Low ATR/BB squeeze -> avoid trading
    - VOLATILITY_EXPANDING: Volatility increasing -> opportunities
    - BOLLINGER_UPPER_TOUCH: Price at upper band -> potential reversal
    - BOLLINGER_LOWER_TOUCH: Price at lower band -> potential reversal
    """

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        low_volatility_percentile: float = 20.0,  # Bottom 20% is "low"
        kline_interval: str = "1h",
    ) -> None:
        """Initialize Volatility Strategy.

        Args:
            data_pool: Shared data pool
            signal_queue: Signal queue for emitting signals
            atr_period: ATR period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            low_volatility_percentile: Percentile threshold for low volatility
            kline_interval: Kline interval for calculations
        """
        super().__init__("volatility", data_pool, signal_queue)
        self._atr_period = atr_period
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._low_vol_percentile = low_volatility_percentile
        self._kline_interval = kline_interval

        # Historical ATR for percentile calculation
        self._atr_history: list[float] = []
        self._max_history = 100

        # Previous states
        self._prev_volatility_state: str | None = None
        self._prev_band_touch: str | None = None

    def _calculate_atr(self, klines: list[dict[str, Any]]) -> float | None:
        """Calculate ATR from klines."""
        if len(klines) < self._atr_period + 1:
            return None

        true_ranges: list[float] = []

        for i in range(1, len(klines)):
            high = float(klines[i].get("high", 0))
            low = float(klines[i].get("low", 0))
            prev_close = float(klines[i - 1].get("close", 0))

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        # Simple ATR (could use EMA for smoothing)
        if len(true_ranges) < self._atr_period:
            return None

        return sum(true_ranges[-self._atr_period :]) / self._atr_period

    def _calculate_bollinger_bands(self, closes: list[float]) -> tuple[float, float, float] | None:
        """Calculate Bollinger Bands.

        Returns:
            (upper, middle, lower) or None if not enough data
        """
        if len(closes) < self._bb_period:
            return None

        period_closes = closes[-self._bb_period :]
        middle = sum(period_closes) / len(period_closes)

        variance = sum((c - middle) ** 2 for c in period_closes) / len(period_closes)
        std_dev = variance**0.5

        upper = middle + (self._bb_std * std_dev)
        lower = middle - (self._bb_std * std_dev)

        return upper, middle, lower

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check volatility conditions.

        Args:
            snapshot: Current data snapshot

        Returns:
            Signal if volatility condition detected, None otherwise
        """
        klines = snapshot.klines.get(self._kline_interval, []) if snapshot.klines else []

        if len(klines) < max(self._atr_period, self._bb_period) + 1:
            return None

        # Calculate ATR
        atr = self._calculate_atr(klines)
        if atr is None:
            return None

        # Track ATR history
        self._atr_history.append(atr)
        if len(self._atr_history) > self._max_history:
            self._atr_history = self._atr_history[-self._max_history :]

        # Calculate ATR percentile
        sorted_atr = sorted(self._atr_history)
        atr_percentile = (
            (sorted_atr.index(atr) + 1) / len(sorted_atr) * 100 if len(sorted_atr) > 5 else 50
        )

        # Calculate Bollinger Bands
        closes = [float(k.get("close", 0)) for k in klines]
        current_price = closes[-1]
        bb = self._calculate_bollinger_bands(closes)

        signal: StrategySignal | None = None

        # Check volatility state
        current_vol_state = "normal"
        if atr_percentile <= self._low_vol_percentile:
            current_vol_state = "low"
        elif atr_percentile >= 80:
            current_vol_state = "high"

        # Volatility state change
        if current_vol_state != self._prev_volatility_state:
            if current_vol_state == "low":
                logger.info(f"Low volatility detected: ATR percentile={atr_percentile:.1f}%")
                signal = StrategySignal(
                    signal_type=SignalType.VOLATILITY_LOW,
                    data={
                        "atr": atr,
                        "atr_percentile": atr_percentile,
                        "recommendation": "avoid_trading",
                        "reason": "Low volatility may lead to choppy price action",
                    },
                    priority=3,  # High priority - affects all trading
                )
            elif current_vol_state == "high" and self._prev_volatility_state == "low":
                logger.info(f"Volatility expanding: ATR percentile={atr_percentile:.1f}%")
                signal = StrategySignal(
                    signal_type=SignalType.VOLATILITY_EXPANDING,
                    data={
                        "atr": atr,
                        "atr_percentile": atr_percentile,
                        "recommendation": "trend_opportunity",
                    },
                    priority=2,
                )

        self._prev_volatility_state = current_vol_state

        # Check Bollinger Band touches (only if not low volatility)
        if bb and current_vol_state != "low":
            upper, middle, lower = bb
            bb_width = (upper - lower) / middle * 100  # BB width as percentage

            current_band_touch = None
            if current_price >= upper * 0.99:  # Within 1% of upper band
                current_band_touch = "upper"
            elif current_price <= lower * 1.01:  # Within 1% of lower band
                current_band_touch = "lower"

            if current_band_touch and current_band_touch != self._prev_band_touch:
                if current_band_touch == "upper":
                    signal = StrategySignal(
                        signal_type=SignalType.BOLLINGER_UPPER_TOUCH,
                        data={
                            "current_price": current_price,
                            "upper_band": upper,
                            "middle_band": middle,
                            "lower_band": lower,
                            "bb_width_percent": bb_width,
                            "bias": "bearish",  # Price at upper band -> potential reversal
                        },
                        priority=1,
                    )
                else:
                    signal = StrategySignal(
                        signal_type=SignalType.BOLLINGER_LOWER_TOUCH,
                        data={
                            "current_price": current_price,
                            "upper_band": upper,
                            "middle_band": middle,
                            "lower_band": lower,
                            "bb_width_percent": bb_width,
                            "bias": "bullish",  # Price at lower band -> potential reversal
                        },
                        priority=1,
                    )

            self._prev_band_touch = current_band_touch

            # Update BB indicator
            self._data_pool.update_indicator(
                f"BB_{self._bb_period}",
                {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "width_percent": bb_width,
                    "position": (current_price - lower) / (upper - lower)
                    if upper != lower
                    else 0.5,
                },
            )

        # Update ATR indicator
        self._data_pool.update_indicator(
            f"ATR_{self._atr_period}",
            {
                "value": atr,
                "percentile": atr_percentile,
                "volatility_state": current_vol_state,
            },
        )

        return signal

    def is_low_volatility(self, snapshot: DataSnapshot) -> bool:
        """Check if current market is in low volatility state.

        Args:
            snapshot: Current data snapshot

        Returns:
            True if volatility is low
        """
        return self._prev_volatility_state == "low"

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self._name,
            "atr_period": self._atr_period,
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "low_volatility_percentile": self._low_vol_percentile,
            "kline_interval": self._kline_interval,
            "enabled": self._enabled,
        }
