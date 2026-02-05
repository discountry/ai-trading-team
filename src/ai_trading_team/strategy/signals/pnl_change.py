"""P&L change signal source.

Detects when user's profit/loss changes by ±5% relative to margin.
This helps the agent track significant portfolio changes.
"""

import logging
from dataclasses import dataclass
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
class PnLState:
    """State for P&L tracking."""

    pnl_percent: float  # P&L as percentage of margin
    unrealized_pnl: float  # USD
    margin: float  # USD
    last_signaled_threshold: int  # Last threshold that triggered signal (5, 10, 15, etc.)


class PnLChangeSignal(SignalSource):
    """Detects significant P&L changes (±5% of margin).

    Signal triggers when P&L crosses a 5% threshold:
    - +5%, +10%, +15%, etc. -> profit increase signals
    - -5%, -10%, -15%, etc. -> profit decrease signals

    The threshold is based on unrealized P&L / margin.
    """

    def __init__(
        self,
        threshold_percent: float = 5.0,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize P&L signal source.

        Args:
            threshold_percent: P&L change threshold (default: 5%)
            timeframes: Timeframes to monitor (default: 1h only)
        """
        super().__init__(
            name="pnl_change",
            timeframes=timeframes or [Timeframe.H1],
        )
        self._threshold = threshold_percent

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> PnLState | None:
        """Compute current P&L state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if no position data
        """
        _ = timeframe  # P&L is global

        # Get position data (single position or use position field)
        position = snapshot.position
        if not position:
            return None

        # Calculate unrealized P&L and margin
        total_pnl = float(
            position.get(
                "unrealized_pnl",
                position.get("unrealizedPnl", position.get("unrealisedPnl", 0)),
            )
        )
        total_margin = float(
            position.get(
                "margin",
                position.get("positionMargin", position.get("initialMargin", 0)),
            )
        )

        if total_margin <= 0:
            return None

        pnl_percent = (total_pnl / total_margin) * 100

        # Calculate which threshold we're at (floor to nearest threshold)
        threshold_level = int(pnl_percent / self._threshold) * int(self._threshold)

        return PnLState(
            pnl_percent=pnl_percent,
            unrealized_pnl=total_pnl,
            margin=total_margin,
            last_signaled_threshold=threshold_level,
        )

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect P&L threshold crossing.

        Args:
            prev_state: Previous PnLState
            new_state: Current PnLState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if P&L crossed a threshold
        """
        _ = snapshot  # Unused

        if not isinstance(new_state, PnLState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(
                f"[{self._name}] initialized: "
                f"P&L={new_state.pnl_percent:+.2f}% (${new_state.unrealized_pnl:+.2f})"
            )
            return None

        if not isinstance(prev_state, PnLState):
            return None

        # Check if threshold level changed
        if prev_state.last_signaled_threshold == new_state.last_signaled_threshold:
            return None

        signal: Signal | None = None
        threshold = new_state.last_signaled_threshold

        if threshold > prev_state.last_signaled_threshold:
            # P&L increased past a threshold
            signal = Signal(
                signal_type=SignalType.PNL_PROFIT_INCREASE,
                direction=SignalDirection.NEUTRAL,  # Informational
                strength=SignalStrength.MODERATE if threshold >= 10 else SignalStrength.WEAK,
                timeframe=timeframe,
                source=self._name,
                data={
                    "pnl_percent": new_state.pnl_percent,
                    "unrealized_pnl": new_state.unrealized_pnl,
                    "margin": new_state.margin,
                    "threshold": threshold,
                    "prev_threshold": prev_state.last_signaled_threshold,
                },
                description=(
                    f"P&L Increase: {new_state.pnl_percent:+.2f}% "
                    f"(${new_state.unrealized_pnl:+.2f}) - crossed +{threshold}%"
                ),
            )

        elif threshold < prev_state.last_signaled_threshold:
            # P&L decreased past a threshold
            signal = Signal(
                signal_type=SignalType.PNL_PROFIT_DECREASE,
                direction=SignalDirection.NEUTRAL,  # Informational
                strength=SignalStrength.STRONG if threshold <= -15 else SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "pnl_percent": new_state.pnl_percent,
                    "unrealized_pnl": new_state.unrealized_pnl,
                    "margin": new_state.margin,
                    "threshold": threshold,
                    "prev_threshold": prev_state.last_signaled_threshold,
                },
                description=(
                    f"P&L Decrease: {new_state.pnl_percent:+.2f}% "
                    f"(${new_state.unrealized_pnl:+.2f}) - crossed {threshold}%"
                ),
            )

        if signal:
            logger.info(f"🎯 {signal.description}")

        return signal
