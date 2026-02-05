"""Long/Short Ratio change signal source.

Detects when L/S ratio changes by more than 5% within 5 minutes.
This is a change-rate based signal, not a threshold-based one.
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
class LSRatioReading:
    """A single L/S ratio reading with timestamp."""

    timestamp: datetime
    ratio: float


@dataclass
class LSRatioState:
    """State for L/S ratio change tracking."""

    current_ratio: float
    change_5min_percent: float  # Change in last 5 minutes
    triggered: bool  # Whether change threshold was triggered


class LongShortRatioSignal(SignalSource):
    """Detects significant L/S ratio changes within 5 minutes.

    Signal triggers when ratio changes by more than 5% in 5 minutes.
    This is a contrarian indicator:
    - Rapid increase in L/S ratio (more longs) -> bearish signal
    - Rapid decrease in L/S ratio (more shorts) -> bullish signal
    """

    def __init__(
        self,
        change_threshold_percent: float = 5.0,
        window_minutes: int = 5,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize L/S ratio signal source.

        Args:
            change_threshold_percent: Minimum change % to trigger (default: 5%)
            window_minutes: Time window for change calculation (default: 5 min)
            timeframes: Timeframes to monitor (default: 1h only for this metric)
        """
        super().__init__(
            name="long_short_ratio",
            timeframes=timeframes or [Timeframe.H1],
        )
        self._change_threshold = change_threshold_percent
        self._window_minutes = window_minutes

        # Store historical readings for change calculation
        # Keyed by timeframe
        self._readings: dict[Timeframe, deque[LSRatioReading]] = {}

        # Track last triggered direction to avoid duplicate signals
        self._last_triggered: dict[Timeframe, str | None] = {}
        self._last_sample_ts: dict[Timeframe, datetime] = {}

    def _get_readings(self, timeframe: Timeframe) -> deque[LSRatioReading]:
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
    ) -> LSRatioState | None:
        """Compute current L/S ratio change state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if insufficient data
        """
        # Get L/S ratio from snapshot
        ls_data = snapshot.long_short_ratio
        if not ls_data:
            return None

        current_ratio = ls_data.get("longShortRatio")
        if current_ratio is None:
            current_ratio = ls_data.get("long_short_ratio")
        if current_ratio is None:
            long_ratio = ls_data.get("long_ratio")
            short_ratio = ls_data.get("short_ratio")
            if long_ratio is not None and short_ratio is not None:
                try:
                    long_ratio_f = float(long_ratio)
                    short_ratio_f = float(short_ratio)
                except (TypeError, ValueError):
                    long_ratio_f = 0.0
                    short_ratio_f = 0.0
                if short_ratio_f > 0:
                    current_ratio = long_ratio_f / short_ratio_f
        if current_ratio is None:
            # Try alternative key
            long_account = ls_data.get("longAccount", 0)
            short_account = ls_data.get("shortAccount", 0)
            if short_account > 0:
                current_ratio = float(long_account) / float(short_account)
            else:
                return None

        current_ratio = float(current_ratio)
        data_ts = self._parse_timestamp(ls_data.get("timestamp") or ls_data.get("time"))
        if data_ts is None:
            data_ts = datetime.now()
        last_ts = self._last_sample_ts.get(timeframe)
        if last_ts and data_ts <= last_ts:
            return None
        self._last_sample_ts[timeframe] = data_ts

        # Add current reading
        readings = self._get_readings(timeframe)
        readings.append(LSRatioReading(timestamp=data_ts, ratio=current_ratio))

        # Calculate change over window
        cutoff = data_ts - timedelta(minutes=self._window_minutes)
        old_readings = [r for r in readings if r.timestamp <= cutoff]

        if not old_readings:
            # Not enough history
            return LSRatioState(
                current_ratio=current_ratio,
                change_5min_percent=0.0,
                triggered=False,
            )

        # Use the oldest reading within window
        old_ratio = old_readings[-1].ratio if old_readings else readings[0].ratio

        if old_ratio == 0:
            return None

        # Calculate percentage change
        change_percent = ((current_ratio - old_ratio) / old_ratio) * 100

        # Check if threshold triggered
        triggered = abs(change_percent) >= self._change_threshold

        return LSRatioState(
            current_ratio=current_ratio,
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
        """Detect L/S ratio surge or drop.

        Args:
            prev_state: Previous LSRatioState
            new_state: Current LSRatioState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if significant change detected
        """
        _ = snapshot  # Unused

        if not isinstance(new_state, LSRatioState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] {timeframe.value} initialized: "
                f"L/S ratio={new_state.current_ratio:.4f}"
            )
            return None

        if not isinstance(prev_state, LSRatioState):
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
            # L/S ratio surged (more longs) -> bearish contrarian
            signal = Signal(
                signal_type=SignalType.LS_RATIO_SURGE,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "current_ratio": new_state.current_ratio,
                    "change_5min_percent": new_state.change_5min_percent,
                    "threshold": self._change_threshold,
                },
                description=(
                    f"L/S Ratio Surge on {timeframe.value}: "
                    f"{new_state.change_5min_percent:+.2f}% in {self._window_minutes}min "
                    f"(ratio={new_state.current_ratio:.4f}) - Bearish contrarian"
                ),
            )

        elif new_state.change_5min_percent <= -self._change_threshold:
            # L/S ratio dropped (more shorts) -> bullish contrarian
            signal = Signal(
                signal_type=SignalType.LS_RATIO_DROP,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "current_ratio": new_state.current_ratio,
                    "change_5min_percent": new_state.change_5min_percent,
                    "threshold": self._change_threshold,
                },
                description=(
                    f"L/S Ratio Drop on {timeframe.value}: "
                    f"{new_state.change_5min_percent:+.2f}% in {self._window_minutes}min "
                    f"(ratio={new_state.current_ratio:.4f}) - Bullish contrarian"
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
