"""Open Interest change signal source.

Detects when OI changes by more than 5% within 5 minutes.
This is a change-rate based signal for detecting significant position changes.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
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


@dataclass
class OIReading:
    """A single OI reading with timestamp."""

    timestamp: datetime
    open_interest: float


@dataclass
class OIState:
    """State for OI change tracking."""

    current_oi: float
    change_5min_percent: float  # Change in last 5 minutes
    triggered: bool  # Whether change threshold was triggered


class OpenInterestSignal(SignalSource):
    """Detects significant Open Interest changes within 5 minutes.

    Signal triggers when OI changes by more than 5% in 5 minutes.
    OI changes indicate:
    - OI surge: New positions entering market (increased conviction)
    - OI drop: Positions closing (reduced conviction)
    """

    def __init__(
        self,
        change_threshold_percent: float = 5.0,
        window_minutes: int = 5,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize OI signal source.

        Args:
            change_threshold_percent: Minimum change % to trigger (default: 5%)
            window_minutes: Time window for change calculation (default: 5 min)
            timeframes: Timeframes to monitor (default: 1h only for this metric)
        """
        super().__init__(
            name="open_interest",
            timeframes=timeframes or [Timeframe.H1],
        )
        self._change_threshold = change_threshold_percent
        self._window_minutes = window_minutes

        # Store historical readings for change calculation
        self._readings: dict[Timeframe, deque[OIReading]] = {}

        # Track last triggered direction to avoid duplicate signals
        self._last_triggered: dict[Timeframe, str | None] = {}
        self._last_sample_ts: dict[Timeframe, datetime] = {}

    def _get_readings(self, timeframe: Timeframe) -> deque[OIReading]:
        """Get or create readings deque for timeframe."""
        if timeframe not in self._readings:
            self._readings[timeframe] = deque(maxlen=100)
            self._last_triggered[timeframe] = None
        return self._readings[timeframe]

    def _parse_timestamp(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, int | float):
            return datetime.fromtimestamp(value / 1000 if value > 1e12 else value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                numeric = float(raw)
            except ValueError:
                numeric = None
            if numeric is not None:
                return datetime.fromtimestamp(numeric / 1000 if numeric > 1e12 else numeric)
            try:
                return datetime.fromisoformat(raw)
            except (TypeError, ValueError):
                return None
        return None

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> OIState | None:
        """Compute current OI change state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if insufficient data
        """
        # Get OI from snapshot
        oi_data = snapshot.open_interest
        if not oi_data:
            return None

        # Try different keys for OI value
        current_oi = oi_data.get("openInterest")
        if current_oi is None:
            current_oi = oi_data.get("sumOpenInterest")
        if current_oi is None:
            current_oi = oi_data.get("oi")
        if current_oi is None:
            current_oi = oi_data.get("open_interest")
        if current_oi is None:
            return None

        current_oi = float(current_oi)
        if current_oi <= 0:
            return None

        data_ts = self._parse_timestamp(oi_data.get("timestamp") or oi_data.get("time"))
        if data_ts is None:
            data_ts = datetime.now()
        last_ts = self._last_sample_ts.get(timeframe)
        if last_ts and data_ts <= last_ts:
            return None
        self._last_sample_ts[timeframe] = data_ts

        # Add current reading
        readings = self._get_readings(timeframe)
        readings.append(OIReading(timestamp=data_ts, open_interest=current_oi))

        # Calculate change over window
        cutoff = data_ts - timedelta(minutes=self._window_minutes)
        old_readings = [r for r in readings if r.timestamp <= cutoff]

        if not old_readings:
            # Not enough history
            return OIState(
                current_oi=current_oi,
                change_5min_percent=0.0,
                triggered=False,
            )

        # Use the oldest reading within window
        old_oi = old_readings[-1].open_interest if old_readings else readings[0].open_interest

        if old_oi == 0:
            return None

        # Calculate percentage change
        change_percent = ((current_oi - old_oi) / old_oi) * 100

        # Check if threshold triggered
        triggered = abs(change_percent) >= self._change_threshold

        return OIState(
            current_oi=current_oi,
            change_5min_percent=change_percent,
            triggered=triggered,
        )

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect OI surge or drop.

        Args:
            prev_state: Previous OIState
            new_state: Current OIState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if significant change detected
        """
        _ = snapshot  # Unused

        if not isinstance(new_state, OIState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] {timeframe.value} initialized: OI={new_state.current_oi:,.0f}"
            )
            return None

        if not isinstance(prev_state, OIState):
            return None

        # Only signal when newly triggered (was not triggered, now is)
        if not new_state.triggered:
            # Reset last triggered when back to normal
            self._last_triggered[timeframe] = None
            return None

        # Determine direction
        direction = "surge" if new_state.change_5min_percent > 0 else "drop"

        # Don't repeat same direction signal
        if self._last_triggered.get(timeframe) == direction:
            return None

        self._last_triggered[timeframe] = direction

        signal: Signal | None = None

        if new_state.change_5min_percent >= self._change_threshold:
            # OI surged - new positions entering (increased conviction)
            signal = Signal(
                signal_type=SignalType.OI_SURGE,
                direction=SignalDirection.NEUTRAL,  # OI alone doesn't indicate direction
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "current_oi": new_state.current_oi,
                    "change_5min_percent": new_state.change_5min_percent,
                    "threshold": self._change_threshold,
                },
                description=(
                    f"OI Surge on {timeframe.value}: "
                    f"{new_state.change_5min_percent:+.2f}% in {self._window_minutes}min "
                    f"(OI={new_state.current_oi:,.0f}) - New positions entering"
                ),
            )

        elif new_state.change_5min_percent <= -self._change_threshold:
            # OI dropped - positions closing (reduced conviction)
            signal = Signal(
                signal_type=SignalType.OI_DROP,
                direction=SignalDirection.NEUTRAL,  # OI alone doesn't indicate direction
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "current_oi": new_state.current_oi,
                    "change_5min_percent": new_state.change_5min_percent,
                    "threshold": self._change_threshold,
                },
                description=(
                    f"OI Drop on {timeframe.value}: "
                    f"{new_state.change_5min_percent:+.2f}% in {self._window_minutes}min "
                    f"(OI={new_state.current_oi:,.0f}) - Positions closing"
                ),
            )

        if signal:
            logger.info(f"🎯 {signal.description}")

        return signal

    def reset(self, timeframe: Timeframe | None = None) -> None:
        """Reset all states including readings history.

        Args:
            timeframe: Specific timeframe to reset, or None for all
        """
        super().reset(timeframe)
        if timeframe is None:
            # Full reset
            self._readings.clear()
            self._last_triggered.clear()
