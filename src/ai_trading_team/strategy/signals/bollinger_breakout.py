"""Bollinger Bands breakout signal source.

Detects when price breaks above upper band or below lower band.
State: ABOVE_UPPER, INSIDE, BELOW_LOWER
Signal: Only emitted when state CHANGES (breakout occurs)
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


class BBPosition(str, Enum):
    """Price position relative to Bollinger Bands."""

    ABOVE_UPPER = "above_upper"  # Price > Upper Band
    INSIDE = "inside"  # Lower <= Price <= Upper
    BELOW_LOWER = "below_lower"  # Price < Lower Band


@dataclass
class BBState:
    """State for Bollinger Bands breakout detection."""

    position: BBPosition
    price: float
    upper_band: float
    middle_band: float
    lower_band: float
    band_width: float  # (Upper - Lower) / Middle as percentage


class BollingerBreakoutSignal(SignalSource):
    """Detects Bollinger Bands breakout.

    Break Upper: Price breaks above upper band
    Break Lower: Price breaks below lower band
    Return Upper: Price returns from above upper band
    Return Lower: Price returns from below lower band

    Uses BB(20, 2) by default.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize Bollinger Bands breakout signal source.

        Args:
            period: BB period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            timeframes: Timeframes to monitor (default: all)
        """
        super().__init__(
            name="bollinger_breakout",
            timeframes=timeframes or ALL_TIMEFRAMES,
            candle_gated=True,
        )
        self._period = period
        self._std_dev = std_dev

    def _calculate_bb(self, closes: list[float]) -> tuple[float, float, float] | None:
        """Calculate Bollinger Bands.

        Args:
            closes: List of close prices

        Returns:
            Tuple of (upper, middle, lower) or None
        """
        if len(closes) < self._period:
            return None

        recent = closes[-self._period :]
        middle = sum(recent) / len(recent)

        # Calculate standard deviation
        variance = sum((x - middle) ** 2 for x in recent) / len(recent)
        std = variance**0.5

        upper = middle + (self._std_dev * std)
        lower = middle - (self._std_dev * std)

        return upper, middle, lower

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> BBState | None:
        """Compute current BB state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if insufficient data
        """
        # Try to get BB from indicators first
        indicators = snapshot.indicators or {}
        bb_key = f"BB_{self._period}_{timeframe.value}"

        bb_data = indicators.get(bb_key)
        if bb_data and isinstance(bb_data, dict):
            upper = bb_data.get("upper")
            middle = bb_data.get("middle")
            lower = bb_data.get("lower")
            if all(v is not None for v in [upper, middle, lower]):
                # Get current price
                ticker = snapshot.ticker
                if ticker:
                    price = float(ticker.get("last_price", 0))
                    if price > 0:
                        return self._create_state(price, upper, middle, lower)

        # Calculate from klines if not in indicators
        if snapshot.klines is None:
            return None

        klines = snapshot.klines.get(timeframe.value, [])
        if len(klines) < self._period:
            return None

        closes = [float(k.get("close", 0)) for k in klines]
        result = self._calculate_bb(closes)

        if result is None:
            return None

        upper, middle, lower = result

        # Get current price from last kline or ticker
        ticker = snapshot.ticker
        price = float(ticker.get("last_price", 0)) if ticker else closes[-1]

        if price == 0:
            return None

        return self._create_state(price, upper, middle, lower)

    def _create_state(self, price: float, upper: float, middle: float, lower: float) -> BBState:
        """Create BBState from values."""
        # Determine position
        if price > upper:
            position = BBPosition.ABOVE_UPPER
        elif price < lower:
            position = BBPosition.BELOW_LOWER
        else:
            position = BBPosition.INSIDE

        # Band width as percentage
        band_width = ((upper - lower) / middle) * 100 if middle > 0 else 0

        return BBState(
            position=position,
            price=price,
            upper_band=upper,
            middle_band=middle,
            lower_band=lower,
            band_width=band_width,
        )

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect BB breakout.

        Args:
            prev_state: Previous BBState
            new_state: Current BBState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if breakout occurred
        """
        if not isinstance(new_state, BBState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] {timeframe.value} initialized: "
                f"price={new_state.price:.4f} ({new_state.position.value}) "
                f"BB=[{new_state.lower_band:.4f}, {new_state.upper_band:.4f}]"
            )
            return None

        if not isinstance(prev_state, BBState):
            return None

        # No position change
        if prev_state.position == new_state.position:
            return None

        signal: Signal | None = None

        # Break above upper band
        if (
            new_state.position == BBPosition.ABOVE_UPPER
            and prev_state.position != BBPosition.ABOVE_UPPER
        ):
            signal = Signal(
                signal_type=SignalType.BB_BREAK_UPPER,
                direction=SignalDirection.BULLISH,  # Momentum breakout
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "price": new_state.price,
                    "upper_band": new_state.upper_band,
                    "band_width": new_state.band_width,
                },
                description=(
                    f"BB Break Upper on {timeframe.value}: "
                    f"price={new_state.price:.4f} > upper={new_state.upper_band:.4f}"
                ),
            )

        # Return from above upper band
        elif (
            prev_state.position == BBPosition.ABOVE_UPPER
            and new_state.position == BBPosition.INSIDE
        ):
            signal = Signal(
                signal_type=SignalType.BB_RETURN_UPPER,
                direction=SignalDirection.BEARISH,  # Mean reversion
                strength=SignalStrength.WEAK,
                timeframe=timeframe,
                source=self._name,
                data={
                    "price": new_state.price,
                    "upper_band": new_state.upper_band,
                    "prev_price": prev_state.price,
                },
                description=(
                    f"BB Return from Upper on {timeframe.value}: "
                    f"price={new_state.price:.4f} back inside bands"
                ),
            )

        # Break below lower band
        elif (
            new_state.position == BBPosition.BELOW_LOWER
            and prev_state.position != BBPosition.BELOW_LOWER
        ):
            signal = Signal(
                signal_type=SignalType.BB_BREAK_LOWER,
                direction=SignalDirection.BEARISH,  # Momentum breakdown
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "price": new_state.price,
                    "lower_band": new_state.lower_band,
                    "band_width": new_state.band_width,
                },
                description=(
                    f"BB Break Lower on {timeframe.value}: "
                    f"price={new_state.price:.4f} < lower={new_state.lower_band:.4f}"
                ),
            )

        # Return from below lower band
        elif (
            prev_state.position == BBPosition.BELOW_LOWER
            and new_state.position == BBPosition.INSIDE
        ):
            signal = Signal(
                signal_type=SignalType.BB_RETURN_LOWER,
                direction=SignalDirection.BULLISH,  # Mean reversion
                strength=SignalStrength.WEAK,
                timeframe=timeframe,
                source=self._name,
                data={
                    "price": new_state.price,
                    "lower_band": new_state.lower_band,
                    "prev_price": prev_state.price,
                },
                description=(
                    f"BB Return from Lower on {timeframe.value}: "
                    f"price={new_state.price:.4f} back inside bands"
                ),
            )

        if signal:
            logger.info(f"🎯 {signal.description}")

        return signal
