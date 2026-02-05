"""MACD Crossover signal source.

Detects MACD golden cross (bullish) and death cross (bearish).
State: BULLISH (MACD > Signal) or BEARISH (MACD < Signal)
Signal: Only emitted when state CHANGES (crossover occurs)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.strategy.signals.base import SignalSource
from ai_trading_team.strategy.signals.types import (
    ALL_TIMEFRAMES,
    Signal,
    SignalDirection,
    SignalStrength,
    SignalType,
    Timeframe,
)

logger = logging.getLogger(__name__)


class MACDPosition(str, Enum):
    """MACD position relative to signal line."""

    ABOVE = "above"  # MACD > Signal (bullish)
    BELOW = "below"  # MACD < Signal (bearish)


@dataclass
class MACDState:
    """State for MACD crossover detection."""

    position: MACDPosition
    macd_value: float
    signal_value: float
    histogram: float


class MACDCrossoverSignal(SignalSource):
    """Detects MACD golden cross and death cross.

    Golden Cross: MACD line crosses above signal line (bullish)
    Death Cross: MACD line crosses below signal line (bearish)

    Uses MACD(12, 26, 9) by default.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize MACD crossover signal source.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            timeframes: Timeframes to monitor (default: all)
        """
        super().__init__(
            name="macd_crossover",
            timeframes=timeframes or ALL_TIMEFRAMES,
            candle_gated=True,
        )
        self._fast = fast_period
        self._slow = slow_period
        self._signal = signal_period

    def _calculate_ema(self, values: list[float], period: int) -> list[float]:
        """Calculate EMA for a list of values.

        Args:
            values: List of values
            period: EMA period

        Returns:
            List of EMA values
        """
        if len(values) < period:
            return []

        multiplier = 2 / (period + 1)
        ema = [sum(values[:period]) / period]  # Initial SMA

        for value in values[period:]:
            ema.append((value - ema[-1]) * multiplier + ema[-1])

        return ema

    def _calculate_macd(self, closes: list[float]) -> tuple[float, float, float] | None:
        """Calculate MACD, Signal, and Histogram.

        Args:
            closes: List of close prices

        Returns:
            Tuple of (macd, signal, histogram) or None
        """
        if len(closes) < self._slow + self._signal:
            return None

        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(closes, self._fast)
        slow_ema = self._calculate_ema(closes, self._slow)

        if not fast_ema or not slow_ema:
            return None

        # Align EMAs (slow starts later)
        offset = self._slow - self._fast
        if len(fast_ema) <= offset:
            return None

        fast_aligned = fast_ema[offset:]
        if len(fast_aligned) != len(slow_ema):
            min_len = min(len(fast_aligned), len(slow_ema))
            fast_aligned = fast_aligned[-min_len:]
            slow_ema = slow_ema[-min_len:]

        # MACD line = Fast EMA - Slow EMA
        macd_line = [f - s for f, s in zip(fast_aligned, slow_ema, strict=False)]

        if len(macd_line) < self._signal:
            return None

        # Signal line = EMA of MACD line
        signal_line = self._calculate_ema(macd_line, self._signal)

        if not signal_line:
            return None

        macd = macd_line[-1]
        signal = signal_line[-1]
        histogram = macd - signal

        return macd, signal, histogram

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> MACDState | None:
        """Compute current MACD state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if insufficient data
        """
        # Try to get MACD from indicators first
        indicators = snapshot.indicators or {}
        macd_key = f"MACD_{self._fast}_{self._slow}_{self._signal}_{timeframe.value}"

        macd_data = indicators.get(macd_key)
        if macd_data and isinstance(macd_data, dict):
            macd_value = macd_data.get("macd")
            signal_value = macd_data.get("signal")
            histogram = macd_data.get("histogram")
            if all(v is not None for v in [macd_value, signal_value, histogram]):
                position = MACDPosition.ABOVE if macd_value > signal_value else MACDPosition.BELOW
                return MACDState(
                    position=position,
                    macd_value=macd_value,
                    signal_value=signal_value,
                    histogram=histogram,
                )

        # Calculate from klines if not in indicators
        if snapshot.klines is None:
            return None

        klines = snapshot.klines.get(timeframe.value, [])
        min_required = self._slow + self._signal + 10
        if len(klines) < min_required:
            return None

        closes = [float(k.get("close", 0)) for k in klines]
        result = self._calculate_macd(closes)

        if result is None:
            return None

        macd_value, signal_value, histogram = result

        position = MACDPosition.ABOVE if macd_value > signal_value else MACDPosition.BELOW

        return MACDState(
            position=position,
            macd_value=macd_value,
            signal_value=signal_value,
            histogram=histogram,
        )

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect MACD crossover.

        Args:
            prev_state: Previous MACDState
            new_state: Current MACDState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if crossover occurred
        """
        if not isinstance(new_state, MACDState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] {timeframe.value} initialized: "
                f"MACD={new_state.macd_value:.6f} Signal={new_state.signal_value:.6f} "
                f"({new_state.position.value})"
            )
            return None

        if not isinstance(prev_state, MACDState):
            return None

        # No position change
        if prev_state.position == new_state.position:
            return None

        signal: Signal | None = None

        # Golden Cross: MACD crosses above signal line
        if new_state.position == MACDPosition.ABOVE and prev_state.position == MACDPosition.BELOW:
            signal = Signal(
                signal_type=SignalType.MACD_GOLDEN_CROSS,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "macd": new_state.macd_value,
                    "signal": new_state.signal_value,
                    "histogram": new_state.histogram,
                    "prev_histogram": prev_state.histogram,
                },
                description=(
                    f"MACD Golden Cross on {timeframe.value}: "
                    f"MACD={new_state.macd_value:.6f} > Signal={new_state.signal_value:.6f}"
                ),
            )

        # Death Cross: MACD crosses below signal line
        elif new_state.position == MACDPosition.BELOW and prev_state.position == MACDPosition.ABOVE:
            signal = Signal(
                signal_type=SignalType.MACD_DEATH_CROSS,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "macd": new_state.macd_value,
                    "signal": new_state.signal_value,
                    "histogram": new_state.histogram,
                    "prev_histogram": prev_state.histogram,
                },
                description=(
                    f"MACD Death Cross on {timeframe.value}: "
                    f"MACD={new_state.macd_value:.6f} < Signal={new_state.signal_value:.6f}"
                ),
            )

        if signal:
            logger.info(f"🎯 {signal.description}")

        return signal
