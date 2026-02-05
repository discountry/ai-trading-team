"""RSI extreme signals.

Detects when RSI enters or exits overbought/oversold zones.
State: NORMAL, OVERSOLD, or OVERBOUGHT
Signal: Only emitted when state CHANGES (enters/exits extreme zone)
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


class RSIZone(str, Enum):
    """RSI zone classification."""

    OVERSOLD = "oversold"  # RSI < 30
    NORMAL = "normal"  # 30 <= RSI <= 70
    OVERBOUGHT = "overbought"  # RSI > 70


@dataclass
class RSIState:
    """State for RSI zone tracking."""

    zone: RSIZone
    rsi_value: float


class RSIExtremeSignal(SignalSource):
    """Detects RSI entering/exiting overbought and oversold zones.

    Signals:
    - RSI_ENTER_OVERSOLD: RSI drops into oversold zone (< 30)
    - RSI_EXIT_OVERSOLD: RSI rises out of oversold zone (bullish momentum)
    - RSI_ENTER_OVERBOUGHT: RSI rises into overbought zone (> 70)
    - RSI_EXIT_OVERBOUGHT: RSI drops out of overbought zone (bearish momentum)
    """

    def __init__(
        self,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        rsi_period: int = 14,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize RSI extreme signal source.

        Args:
            oversold_threshold: RSI level for oversold (default: 30)
            overbought_threshold: RSI level for overbought (default: 70)
            rsi_period: RSI calculation period
            timeframes: Timeframes to monitor
        """
        super().__init__(
            name="rsi_extreme",
            timeframes=timeframes or [Timeframe.H1],
            candle_gated=True,
        )
        self._oversold = oversold_threshold
        self._overbought = overbought_threshold
        self._rsi_period = rsi_period

    def _calculate_rsi(self, closes: list[float]) -> float | None:
        """Calculate RSI from close prices.

        Args:
            closes: List of close prices

        Returns:
            RSI value or None if insufficient data
        """
        if len(closes) < self._rsi_period + 1:
            return None

        # Calculate price changes
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # Separate gains and losses
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        # Calculate average gain/loss (Wilder's smoothing)
        avg_gain = sum(gains[-self._rsi_period :]) / self._rsi_period
        avg_loss = sum(losses[-self._rsi_period :]) / self._rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> RSIState | None:
        """Compute current RSI zone.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if insufficient data
        """
        # Try to get RSI from indicators first
        indicators = snapshot.indicators or {}
        rsi_key = f"RSI_{self._rsi_period}_{timeframe.value}"
        rsi_value = indicators.get(rsi_key)

        # If not in indicators, calculate from klines
        if rsi_value is None:
            if snapshot.klines is None:
                return None
            klines = snapshot.klines.get(timeframe.value, [])
            if len(klines) < self._rsi_period + 1:
                return None

            closes = [float(k.get("close", 0)) for k in klines]
            rsi_value = self._calculate_rsi(closes)

        if rsi_value is None:
            return None

        # Determine zone
        if rsi_value < self._oversold:
            zone = RSIZone.OVERSOLD
        elif rsi_value > self._overbought:
            zone = RSIZone.OVERBOUGHT
        else:
            zone = RSIZone.NORMAL

        return RSIState(zone=zone, rsi_value=rsi_value)

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect RSI zone transition.

        Args:
            prev_state: Previous RSIState
            new_state: Current RSIState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if zone changed
        """
        if not isinstance(new_state, RSIState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] {timeframe.value} initialized: "
                f"RSI={new_state.rsi_value:.1f} ({new_state.zone.value})"
            )
            return None

        if not isinstance(prev_state, RSIState):
            return None

        prev_rsi = prev_state.rsi_value
        current_rsi = new_state.rsi_value

        # No meaningful change around thresholds
        if prev_state.zone == new_state.zone:
            return None

        signal: Signal | None = None

        # Entering oversold
        if prev_rsi >= self._oversold and current_rsi < self._oversold:
            signal = Signal(
                signal_type=SignalType.RSI_ENTER_OVERSOLD,
                direction=SignalDirection.NEUTRAL,  # Just entering, not actionable yet
                strength=SignalStrength.WEAK,
                timeframe=timeframe,
                source=self._name,
                data={"rsi": current_rsi, "prev_rsi": prev_rsi, "threshold": self._oversold},
                description=(
                    f"RSI entered OVERSOLD zone on {timeframe.value}: "
                    f"RSI {prev_rsi:.1f} -> {current_rsi:.1f} < {self._oversold}"
                ),
            )

        # Exiting oversold (BULLISH signal)
        elif prev_rsi <= self._oversold and current_rsi > self._oversold:
            signal = Signal(
                signal_type=SignalType.RSI_EXIT_OVERSOLD,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={"rsi": current_rsi, "prev_rsi": prev_rsi},
                description=(
                    f"RSI exited OVERSOLD zone on {timeframe.value}: "
                    f"RSI {prev_rsi:.1f} -> {current_rsi:.1f} (bullish momentum)"
                ),
            )

        # Entering overbought
        elif prev_rsi <= self._overbought and current_rsi > self._overbought:
            signal = Signal(
                signal_type=SignalType.RSI_ENTER_OVERBOUGHT,
                direction=SignalDirection.NEUTRAL,
                strength=SignalStrength.WEAK,
                timeframe=timeframe,
                source=self._name,
                data={"rsi": current_rsi, "prev_rsi": prev_rsi, "threshold": self._overbought},
                description=(
                    f"RSI entered OVERBOUGHT zone on {timeframe.value}: "
                    f"RSI {prev_rsi:.1f} -> {current_rsi:.1f} > {self._overbought}"
                ),
            )

        # Exiting overbought (BEARISH signal)
        elif prev_rsi >= self._overbought and current_rsi < self._overbought:
            signal = Signal(
                signal_type=SignalType.RSI_EXIT_OVERBOUGHT,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={"rsi": current_rsi, "prev_rsi": prev_rsi},
                description=(
                    f"RSI exited OVERBOUGHT zone on {timeframe.value}: "
                    f"RSI {prev_rsi:.1f} -> {current_rsi:.1f} (bearish momentum)"
                ),
            )

        if signal:
            logger.info(f"🎯 {signal.description}")

        return signal
