"""Risk signal source.

Detects when risk module triggers stop loss, trailing stop, or take profit.
These are critical signals that indicate forced position changes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
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
class RiskEvent:
    """A risk event triggered by the risk module."""

    event_type: str  # "stop_loss", "trailing_stop", "take_profit"
    timestamp: datetime
    reason: str
    data: dict[str, Any]


@dataclass
class RiskState:
    """State for risk event tracking."""

    last_event: RiskEvent | None
    event_count: int


class RiskSignal(SignalSource):
    """Detects risk module triggered events.

    Signals:
    - RISK_FORCE_STOP_LOSS: Risk module triggered forced stop loss
    - RISK_TRAILING_STOP: Trailing stop was triggered
    - RISK_TAKE_PROFIT: Take profit was triggered

    These are critical signals that require immediate agent attention.
    """

    def __init__(
        self,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize risk signal source.

        Args:
            timeframes: Timeframes to monitor (default: 1h only)
        """
        super().__init__(
            name="risk_signal",
            timeframes=timeframes or [Timeframe.H1],
        )
        self._event_count = 0
        self._last_processed_event: RiskEvent | None = None

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> RiskState | None:
        """Compute current risk state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state
        """
        _ = timeframe  # Risk events are global

        # Get risk events from snapshot
        # These should be populated by the risk monitor
        risk_events = getattr(snapshot, "risk_events", None)
        if not risk_events:
            # Try getting from operations log
            recent_operations = snapshot.recent_operations or []
            risk_ops = [
                op
                for op in recent_operations
                if op.get("source") == "risk"
                or op.get("type") in ["stop_loss", "trailing_stop", "take_profit"]
            ]
            if risk_ops:
                latest = risk_ops[-1]
                event = RiskEvent(
                    event_type=latest.get("type", "unknown"),
                    timestamp=datetime.fromisoformat(
                        latest.get("timestamp", datetime.now().isoformat())
                    ),
                    reason=latest.get("reason", ""),
                    data=latest,
                )
                return RiskState(last_event=event, event_count=len(risk_ops))

            return RiskState(last_event=None, event_count=0)

        # Process risk_events if available
        if isinstance(risk_events, list) and risk_events:
            latest = risk_events[-1]
            event = RiskEvent(
                event_type=latest.get("type", "unknown"),
                timestamp=datetime.fromisoformat(
                    latest.get("timestamp", datetime.now().isoformat())
                ),
                reason=latest.get("reason", ""),
                data=latest,
            )
            return RiskState(last_event=event, event_count=len(risk_events))

        return RiskState(last_event=None, event_count=0)

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect risk event.

        Args:
            prev_state: Previous RiskState
            new_state: Current RiskState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if risk event occurred
        """
        _ = snapshot  # Unused

        if not isinstance(new_state, RiskState):
            return None

        # No event
        if new_state.last_event is None:
            return None

        # Check if this is a new event
        if prev_state is not None and isinstance(prev_state, RiskState):
            if prev_state.event_count >= new_state.event_count:
                return None
            if (
                self._last_processed_event
                and self._last_processed_event.timestamp == new_state.last_event.timestamp
                and self._last_processed_event.event_type == new_state.last_event.event_type
            ):
                return None

        self._last_processed_event = new_state.last_event
        event = new_state.last_event

        signal: Signal | None = None

        if event.event_type == "stop_loss" or event.event_type == "force_stop":
            signal = Signal(
                signal_type=SignalType.RISK_FORCE_STOP_LOSS,
                direction=SignalDirection.NEUTRAL,
                strength=SignalStrength.STRONG,
                timeframe=timeframe,
                source=self._name,
                data={
                    "event_type": event.event_type,
                    "reason": event.reason,
                    **event.data,
                },
                description=f"RISK: Force Stop Loss triggered - {event.reason}",
            )

        elif event.event_type == "trailing_stop":
            signal = Signal(
                signal_type=SignalType.RISK_TRAILING_STOP,
                direction=SignalDirection.NEUTRAL,
                strength=SignalStrength.STRONG,
                timeframe=timeframe,
                source=self._name,
                data={
                    "event_type": event.event_type,
                    "reason": event.reason,
                    **event.data,
                },
                description=f"RISK: Trailing Stop triggered - {event.reason}",
            )

        elif event.event_type == "take_profit":
            signal = Signal(
                signal_type=SignalType.RISK_TAKE_PROFIT,
                direction=SignalDirection.NEUTRAL,
                strength=SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "event_type": event.event_type,
                    "reason": event.reason,
                    **event.data,
                },
                description=f"RISK: Take Profit triggered - {event.reason}",
            )

        if signal:
            logger.info(f"🚨 {signal.description}")

        return signal
