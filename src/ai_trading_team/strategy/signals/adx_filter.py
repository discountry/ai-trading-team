"""ADX (Average Directional Index) trend strength filter.

Filters entry signals based on ADX value.
Only allows entry signals when ADX >= threshold (trending market).

ADX Interpretation:
- 0-25: Weak or no trend (avoid trading)
- 25-50: Strong trend (good for trend following)
- 50-75: Very strong trend
- 75-100: Extremely strong trend (rare)
"""

import logging
from dataclasses import dataclass

from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.strategy.signals.types import Timeframe

logger = logging.getLogger(__name__)


@dataclass
class ADXValues:
    """ADX indicator values."""

    adx: float
    plus_di: float  # +DI
    minus_di: float  # -DI


class ADXFilter:
    """Filter for checking trend strength before allowing entry signals.

    Uses ADX (Average Directional Index) to measure trend strength.
    Entry signals are only allowed when ADX >= threshold.

    This is NOT a signal source - it's a filter that gates other signals.
    """

    # Default threshold: 25 is commonly used as minimum for trending
    DEFAULT_THRESHOLD = 25.0

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        period: int = 14,
    ) -> None:
        """Initialize ADX filter.

        Args:
            threshold: Minimum ADX value to allow entry signals (default: 25)
            period: ADX calculation period (default: 14)
        """
        self._threshold = threshold
        self._period = period
        self._enabled = True

        # Cache last computed ADX per timeframe
        self._cached_adx: dict[Timeframe, ADXValues | None] = {}

    @property
    def enabled(self) -> bool:
        """Check if filter is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable filter."""
        self._enabled = value

    @property
    def threshold(self) -> float:
        """Current ADX threshold."""
        return self._threshold

    def compute_adx(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> ADXValues | None:
        """Compute ADX from kline data.

        ADX = 100 * EMA(|+DI - -DI| / (+DI + -DI), period)

        Where:
        - +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
        - -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
        - TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        - +DI = 100 * EMA(+DM, period) / EMA(TR, period)
        - -DI = 100 * EMA(-DM, period) / EMA(TR, period)

        Args:
            snapshot: Market data snapshot
            timeframe: Timeframe to compute for

        Returns:
            ADX values or None if insufficient data
        """
        if snapshot.klines is None:
            return None

        klines = snapshot.klines.get(timeframe.value, [])
        # Need period * 2 for proper smoothing
        min_required = self._period * 2 + 1
        if len(klines) < min_required:
            return None

        # Extract OHLC data
        highs = [float(k.get("high", 0)) for k in klines]
        lows = [float(k.get("low", 0)) for k in klines]
        closes = [float(k.get("close", 0)) for k in klines]

        if not all(highs) or not all(lows) or not all(closes):
            return None

        # Calculate +DM, -DM, and TR
        plus_dm = []
        minus_dm = []
        tr_values = []

        for i in range(1, len(klines)):
            high = highs[i]
            low = lows[i]
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            prev_close = closes[i - 1]

            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        # Calculate smoothed averages using Wilder's smoothing (RMA)
        def wilders_smooth(data: list[float], period: int) -> list[float]:
            """Wilder's smoothing (RMA)."""
            result = []
            alpha = 1.0 / period

            # First value is SMA
            if len(data) < period:
                return []

            first_avg = sum(data[:period]) / period
            result.append(first_avg)

            for i in range(period, len(data)):
                rma = result[-1] * (1 - alpha) + data[i] * alpha
                result.append(rma)

            return result

        if len(plus_dm) < self._period:
            return None

        # Smooth +DM, -DM, TR
        smooth_plus_dm = wilders_smooth(plus_dm, self._period)
        smooth_minus_dm = wilders_smooth(minus_dm, self._period)
        smooth_tr = wilders_smooth(tr_values, self._period)

        if not smooth_plus_dm or not smooth_minus_dm or not smooth_tr:
            return None

        # Calculate +DI and -DI series
        plus_di_series = []
        minus_di_series = []

        min_len = min(len(smooth_plus_dm), len(smooth_minus_dm), len(smooth_tr))
        for i in range(min_len):
            if smooth_tr[i] > 0:
                plus_di = 100 * smooth_plus_dm[i] / smooth_tr[i]
                minus_di = 100 * smooth_minus_dm[i] / smooth_tr[i]
            else:
                plus_di = 0
                minus_di = 0
            plus_di_series.append(plus_di)
            minus_di_series.append(minus_di)

        # Calculate DX series
        dx_series = []
        for i in range(len(plus_di_series)):
            di_sum = plus_di_series[i] + minus_di_series[i]
            if di_sum > 0:
                dx = 100 * abs(plus_di_series[i] - minus_di_series[i]) / di_sum
            else:
                dx = 0
            dx_series.append(dx)

        # Smooth DX to get ADX
        if len(dx_series) < self._period:
            return None

        adx_series = wilders_smooth(dx_series, self._period)
        if not adx_series:
            return None

        # Get latest values
        adx_value = adx_series[-1]
        plus_di_value = plus_di_series[-1] if plus_di_series else 0
        minus_di_value = minus_di_series[-1] if minus_di_series else 0

        result = ADXValues(
            adx=adx_value,
            plus_di=plus_di_value,
            minus_di=minus_di_value,
        )

        # Cache the result
        self._cached_adx[timeframe] = result

        return result

    def is_trending(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> bool:
        """Check if market is trending (ADX >= threshold).

        Args:
            snapshot: Market data snapshot
            timeframe: Timeframe to check

        Returns:
            True if ADX >= threshold (trending), False otherwise
        """
        if not self._enabled:
            return True  # Always allow if filter disabled

        adx_values = self.compute_adx(snapshot, timeframe)
        if adx_values is None:
            # If we can't compute ADX, default to allowing signals
            logger.warning(
                f"Could not compute ADX for {timeframe.value}, allowing signal"
            )
            return True

        is_trending = adx_values.adx >= self._threshold

        if not is_trending:
            logger.info(
                f"ADX filter blocked signal: ADX={adx_values.adx:.2f} < {self._threshold} "
                f"on {timeframe.value} (market not trending)"
            )

        return is_trending

    def should_allow_entry_signal(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> bool:
        """Decide if an entry signal should be allowed.

        This is the main method called by signal aggregator.

        Args:
            snapshot: Market data snapshot
            timeframe: Timeframe of the signal

        Returns:
            True if entry signal should be allowed
        """
        return self.is_trending(snapshot, timeframe)

    def get_cached_adx(self, timeframe: Timeframe) -> ADXValues | None:
        """Get cached ADX value for a timeframe.

        Args:
            timeframe: Timeframe to query

        Returns:
            Cached ADX values or None
        """
        return self._cached_adx.get(timeframe)

    def get_trend_info(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> dict:
        """Get detailed trend information for context.

        Args:
            snapshot: Market data snapshot
            timeframe: Timeframe to analyze

        Returns:
            Dictionary with trend information
        """
        adx_values = self.compute_adx(snapshot, timeframe)

        if adx_values is None:
            return {
                "adx": None,
                "plus_di": None,
                "minus_di": None,
                "trend_strength": "unknown",
                "trend_direction": "unknown",
                "is_trending": True,  # Default to allow
            }

        # Determine trend strength label
        if adx_values.adx < 20:
            strength = "very_weak"
        elif adx_values.adx < 25:
            strength = "weak"
        elif adx_values.adx < 40:
            strength = "moderate"
        elif adx_values.adx < 50:
            strength = "strong"
        else:
            strength = "very_strong"

        # Determine trend direction from DI
        if adx_values.plus_di > adx_values.minus_di:
            direction = "bullish"
        elif adx_values.minus_di > adx_values.plus_di:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "adx": round(adx_values.adx, 2),
            "plus_di": round(adx_values.plus_di, 2),
            "minus_di": round(adx_values.minus_di, 2),
            "trend_strength": strength,
            "trend_direction": direction,
            "is_trending": adx_values.adx >= self._threshold,
        }
