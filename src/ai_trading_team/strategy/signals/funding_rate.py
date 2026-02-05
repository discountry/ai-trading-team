"""Funding rate shift signal.

Detects significant changes in perpetual futures funding rate.
State: NORMAL, HIGH_POSITIVE, or HIGH_NEGATIVE
Signal: Only emitted when state CHANGES
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


class FundingZone(str, Enum):
    """Funding rate zone classification."""

    HIGH_NEGATIVE = "high_negative"  # < -0.01% (shorts paying longs)
    NORMAL = "normal"  # -0.01% to 0.03%
    HIGH_POSITIVE = "high_positive"  # > 0.03% (longs paying shorts)


@dataclass
class FundingState:
    """State for funding rate tracking."""

    zone: FundingZone
    funding_rate: float  # As percentage


class FundingRateSignal(SignalSource):
    """Detects significant funding rate shifts.

    Funding rate signals are contrarian:
    - High positive funding (longs paying shorts) → Bearish (crowded long)
    - High negative funding (shorts paying longs) → Bullish (crowded short)

    This is a non-timeframe signal - funding rate is the same across timeframes.
    """

    def __init__(
        self,
        high_positive_threshold: float = 0.03,  # 0.03%
        high_negative_threshold: float = -0.01,  # -0.01%
    ) -> None:
        """Initialize funding rate signal source.

        Args:
            high_positive_threshold: Threshold for high positive funding (%)
            high_negative_threshold: Threshold for high negative funding (%)
        """
        super().__init__(
            name="funding_rate",
            timeframes=[Timeframe.H1],  # Funding checked hourly
        )
        self._high_positive = high_positive_threshold
        self._high_negative = high_negative_threshold

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> FundingState | None:
        """Compute current funding rate zone.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context (not used for funding)

        Returns:
            Current state or None if no funding data
        """
        _ = timeframe  # Funding rate is global, not timeframe-specific

        funding_data = snapshot.funding_rate
        if not funding_data:
            return None

        # Try different possible keys (Binance uses different formats)
        funding_rate = funding_data.get("lastFundingRate")
        if funding_rate is None:
            funding_rate = funding_data.get("fundingRate")
        if funding_rate is None:
            funding_rate = funding_data.get("funding_rate")
        if funding_rate is None:
            funding_rate = funding_data.get("r")  # Some WebSocket formats
        if funding_rate is None:
            return None

        funding_rate = float(funding_rate)

        # Convert to percentage if needed (some APIs return as decimal)
        if abs(funding_rate) < 0.001:  # Likely in decimal form (0.0001 = 0.01%)
            funding_rate = funding_rate * 100

        # Determine zone
        if funding_rate > self._high_positive:
            zone = FundingZone.HIGH_POSITIVE
        elif funding_rate < self._high_negative:
            zone = FundingZone.HIGH_NEGATIVE
        else:
            zone = FundingZone.NORMAL

        return FundingState(zone=zone, funding_rate=funding_rate)

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect funding rate zone transition.

        Args:
            prev_state: Previous FundingState
            new_state: Current FundingState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if zone changed
        """
        _ = snapshot  # Not needed for signal generation

        if not isinstance(new_state, FundingState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] initialized: "
                f"funding={new_state.funding_rate:.4f}% ({new_state.zone.value})"
            )
            return None

        if not isinstance(prev_state, FundingState):
            return None

        # No zone change
        if prev_state.zone == new_state.zone:
            return None

        signal: Signal | None = None

        # Entering high positive (bearish - crowded longs)
        if new_state.zone == FundingZone.HIGH_POSITIVE:
            signal = Signal(
                signal_type=SignalType.FUNDING_SPIKE_POSITIVE,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "funding_rate": new_state.funding_rate,
                    "prev_funding_rate": prev_state.funding_rate,
                    "threshold": self._high_positive,
                },
                description=(
                    f"Funding rate SPIKE POSITIVE: {new_state.funding_rate:.4f}% "
                    f"(crowded longs - bearish)"
                ),
            )

        # Entering high negative (bullish - crowded shorts)
        elif new_state.zone == FundingZone.HIGH_NEGATIVE:
            signal = Signal(
                signal_type=SignalType.FUNDING_SPIKE_NEGATIVE,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "funding_rate": new_state.funding_rate,
                    "prev_funding_rate": prev_state.funding_rate,
                    "threshold": self._high_negative,
                },
                description=(
                    f"Funding rate SPIKE NEGATIVE: {new_state.funding_rate:.4f}% "
                    f"(crowded shorts - bullish)"
                ),
            )

        # Returning to normal
        elif new_state.zone == FundingZone.NORMAL:
            signal = Signal(
                signal_type=SignalType.FUNDING_NORMALIZE,
                direction=SignalDirection.NEUTRAL,
                strength=SignalStrength.WEAK,
                timeframe=timeframe,
                source=self._name,
                data={
                    "funding_rate": new_state.funding_rate,
                    "prev_zone": prev_state.zone.value,
                },
                description=(
                    f"Funding rate NORMALIZED: {new_state.funding_rate:.4f}% "
                    f"(from {prev_state.zone.value})"
                ),
            )

        if signal:
            logger.info(f"🎯 {signal.description}")

        return signal
