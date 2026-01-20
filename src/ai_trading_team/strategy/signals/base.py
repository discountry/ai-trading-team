"""Base class for event-driven signal sources."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.strategy.signals.types import Signal, Timeframe

logger = logging.getLogger(__name__)


class SignalSource(ABC):
    """Base class for event-driven signal sources.

    Each signal source:
    1. Maintains its own internal state
    2. Only emits signals when state CHANGES
    3. Supports multiple timeframes
    4. Is responsible for a single type of signal logic

    Subclasses must implement:
    - _compute_state(): Compute current state from snapshot
    - _detect_transition(): Check if state changed and generate signal
    """

    def __init__(
        self,
        name: str,
        timeframes: list[Timeframe] | None = None,
        enabled: bool = True,
        candle_gated: bool = False,
    ) -> None:
        """Initialize signal source.

        Args:
            name: Unique name for this signal source
            timeframes: List of timeframes to monitor (default: all)
            enabled: Whether this source is active
        """
        self._name = name
        self._timeframes = timeframes or list(Timeframe)
        self._enabled = enabled
        self._candle_gated = candle_gated

        # State tracking per timeframe
        # Key: timeframe, Value: current state (implementation-specific)
        self._state: dict[Timeframe, Any] = {}
        self._last_update: dict[Timeframe, datetime] = {}
        self._last_signal_candle: dict[Timeframe, Any] = {}

    @property
    def name(self) -> str:
        """Signal source name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Whether source is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        self._enabled = value

    @property
    def timeframes(self) -> list[Timeframe]:
        """Supported timeframes."""
        return self._timeframes

    def update(self, snapshot: DataSnapshot, timeframe: Timeframe) -> Signal | None:
        """Update state and check for signal.

        This is the main entry point called by the signal aggregator.

        Args:
            snapshot: Current market data snapshot
            timeframe: Timeframe to update

        Returns:
            Signal if state changed, None otherwise
        """
        if not self._enabled:
            return None

        if timeframe not in self._timeframes:
            return None

        # Compute new state
        new_state = self._compute_state(snapshot, timeframe)
        if new_state is None:
            return None  # Not enough data

        # Get previous state
        prev_state = self._state.get(timeframe)

        # Detect state transition
        signal = self._detect_transition(prev_state, new_state, timeframe, snapshot)

        # Update stored state
        self._state[timeframe] = new_state
        self._last_update[timeframe] = datetime.now()

        if signal and self._candle_gated:
            candle_id = self._get_candle_id(snapshot, timeframe)
            if candle_id is not None:
                last_candle_id = self._last_signal_candle.get(timeframe)
                if last_candle_id == candle_id:
                    return None
                self._last_signal_candle[timeframe] = candle_id

        return signal

    @abstractmethod
    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> Any:
        """Compute current state from snapshot.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state (implementation-specific), or None if insufficient data
        """
        ...

    @abstractmethod
    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect state transition and generate signal if applicable.

        Args:
            prev_state: Previous state (may be None on first call)
            new_state: Current state
            timeframe: Timeframe context
            snapshot: Current market data for additional context

        Returns:
            Signal if state changed, None otherwise
        """
        ...

    def reset(self, timeframe: Timeframe | None = None) -> None:
        """Reset state for a timeframe or all timeframes.

        Args:
            timeframe: Specific timeframe to reset, or None for all
        """
        if timeframe:
            self._state.pop(timeframe, None)
            self._last_update.pop(timeframe, None)
            self._last_signal_candle.pop(timeframe, None)
        else:
            self._state.clear()
            self._last_update.clear()
            self._last_signal_candle.clear()

    def get_state(self, timeframe: Timeframe) -> Any:
        """Get current state for a timeframe.

        Args:
            timeframe: Timeframe to query

        Returns:
            Current state or None
        """
        return self._state.get(timeframe)

    def _get_candle_id(self, snapshot: DataSnapshot, timeframe: Timeframe) -> Any | None:
        if snapshot.klines is None:
            return None
        klines = snapshot.klines.get(timeframe.value, [])
        if not klines:
            return None
        last = klines[-1]
        if not isinstance(last, dict):
            return None
        return last.get("open_time") or last.get("close_time")
