"""MA Crossover signal source.

Detects when price crosses above or below a moving average.
State: ABOVE_MA or BELOW_MA
Signal: Only emitted when state CHANGES (crossover occurs)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.strategy.signals.base import SignalSource
from ai_trading_team.strategy.signals.types import (
    Signal,
    SignalDirection,
    SignalStrength,
    SignalType,
    Timeframe,
)

logger = logging.getLogger(__name__)


class MAPosition(str, Enum):
    """Price position relative to MA."""

    ABOVE = "above"
    BELOW = "below"
    AT = "at"  # Within threshold of MA


@dataclass
class MACrossoverState:
    """State for MA crossover detection."""

    position: MAPosition
    price: float
    ma_value: float
    distance_percent: float  # Distance from MA as percentage


class MACrossoverSignal(SignalSource):
    """Detects price crossing above/below moving average.

    Only emits signals when price CROSSES the MA, not continuously
    while price is above/below.

    Supports multiple MA periods (e.g., MA20, MA60, MA200).
    """

    def __init__(
        self,
        ma_period: int = 60,
        threshold_percent: float = 0.1,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize MA crossover signal source.

        Args:
            ma_period: MA period (e.g., 60 for MA60)
            threshold_percent: Percentage threshold for "at MA" zone
            timeframes: Timeframes to monitor
        """
        super().__init__(
            name=f"ma_crossover_{ma_period}",
            timeframes=timeframes or [Timeframe.H1],
            candle_gated=True,
        )
        self._ma_period = ma_period
        self._threshold_percent = threshold_percent
        self._last_non_at: dict[Timeframe, MAPosition] = {}

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> MACrossoverState | None:
        """Compute current price position relative to MA.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if insufficient data
        """
        # Get klines for the timeframe
        if snapshot.klines is None:
            return None
        klines = snapshot.klines.get(timeframe.value, [])
        if len(klines) < self._ma_period:
            return None

        # Calculate MA from klines
        closes = [float(k.get("close", 0)) for k in klines[-self._ma_period :]]
        if not closes or all(c == 0 for c in closes):
            return None

        # Filter out any zero values for MA calculation
        valid_closes = [c for c in closes if c > 0]
        if len(valid_closes) < self._ma_period // 2:  # Need at least half the period
            return None

        ma_value = sum(valid_closes) / len(valid_closes)

        # Use timeframe kline close to avoid intra-candle noise; fallback to ticker
        current_price = float(klines[-1].get("close", 0))
        if current_price == 0:
            ticker = snapshot.ticker
            if not ticker:
                return None
            current_price = float(ticker.get("last_price", 0))
            if current_price == 0:
                return None

        # Calculate distance from MA
        distance_percent = ((current_price - ma_value) / ma_value) * 100

        # Determine position
        if abs(distance_percent) <= self._threshold_percent:
            position = MAPosition.AT
        elif current_price > ma_value:
            position = MAPosition.ABOVE
        else:
            position = MAPosition.BELOW

        return MACrossoverState(
            position=position,
            price=current_price,
            ma_value=ma_value,
            distance_percent=distance_percent,
        )

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect MA crossover.

        Args:
            prev_state: Previous MACrossoverState
            new_state: Current MACrossoverState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if crossover occurred
        """
        if not isinstance(new_state, MACrossoverState):
            return None

        # First update - no transition
        if prev_state is None:
            if new_state.position != MAPosition.AT:
                self._last_non_at[timeframe] = new_state.position
            logger.debug(
                f"[{self._name}] {timeframe.value} initialized: "
                f"price {new_state.position.value} MA{self._ma_period} "
                f"({new_state.distance_percent:+.2f}%)"
            )
            return None

        if not isinstance(prev_state, MACrossoverState):
            return None

        # No position change
        if prev_state.position == new_state.position:
            return None

        if new_state.position == MAPosition.AT:
            return None

        last_non_at = self._last_non_at.get(timeframe)
        if last_non_at is None:
            self._last_non_at[timeframe] = new_state.position
            return None
        if last_non_at == new_state.position:
            return None

        # Detect crossover direction
        signal: Signal | None = None

        # Cross UP: was below/at, now above
        if new_state.position == MAPosition.ABOVE:
            signal = Signal(
                signal_type=SignalType.MA_CROSS_UP,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "ma_period": self._ma_period,
                    "price": new_state.price,
                    "ma_value": new_state.ma_value,
                    "prev_position": prev_state.position.value,
                },
                description=(
                    f"Price crossed ABOVE MA{self._ma_period} on {timeframe.value}: "
                    f"{new_state.price:.4f} > {new_state.ma_value:.4f}"
                ),
            )

        # Cross DOWN: was above/at, now below
        elif new_state.position == MAPosition.BELOW:
            signal = Signal(
                signal_type=SignalType.MA_CROSS_DOWN,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "ma_period": self._ma_period,
                    "price": new_state.price,
                    "ma_value": new_state.ma_value,
                    "prev_position": prev_state.position.value,
                },
                description=(
                    f"Price crossed BELOW MA{self._ma_period} on {timeframe.value}: "
                    f"{new_state.price:.4f} < {new_state.ma_value:.4f}"
                ),
            )

        if signal:
            self._last_non_at[timeframe] = new_state.position
            logger.info(f"🎯 {signal.description}")

        return signal

    def reset(self, timeframe: Timeframe | None = None) -> None:
        """Reset state for a timeframe or all timeframes."""
        super().reset(timeframe)
        if timeframe:
            self._last_non_at.pop(timeframe, None)
        else:
            self._last_non_at.clear()
