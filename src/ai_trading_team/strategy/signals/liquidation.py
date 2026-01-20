"""Liquidation signal source.

Detects when large liquidations (>=100k / >=1M USD) occur.
This is an event-based signal for detecting forced position closures.
"""

import logging
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
class LiquidationState:
    """State for liquidation tracking."""

    last_liquidation_time: datetime | None
    last_liquidation_side: str | None  # "LONG" or "SHORT"
    last_liquidation_value: float  # USD value


class LiquidationSignal(SignalSource):
    """Detects large liquidation events (>=100k / >=1M USD).

    Large liquidations indicate:
    - Long liquidation: Forced selling pressure (can accelerate downtrend)
    - Short liquidation: Forced buying pressure (can accelerate uptrend)

    This is typically used as a momentum/trend confirmation signal.
    """

    def __init__(
        self,
        min_value_usd: float = 100_000.0,
        major_value_usd: float = 1_000_000.0,
        cooldown_seconds: int = 60,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize liquidation signal source.

        Args:
            min_value_usd: Minimum liquidation value in USD (default: 100k)
            major_value_usd: Major liquidation value in USD (default: 1M)
            cooldown_seconds: Minimum time between signals (default: 60s)
            timeframes: Timeframes to monitor (default: 1h only)
        """
        super().__init__(
            name="liquidation",
            timeframes=timeframes or [Timeframe.H1],
        )
        self._minor_value = min_value_usd
        self._major_value = max(major_value_usd, min_value_usd)
        self._cooldown = timedelta(seconds=cooldown_seconds)

        # Track last signal time to prevent spam
        self._last_signal_time: datetime | None = None
        self._last_signal_tier: str | None = None

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> LiquidationState | None:
        """Compute current liquidation state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state or None if no recent liquidation
        """
        _ = timeframe  # Liquidation is global

        # Get liquidation data from snapshot
        liq_data = snapshot.liquidations
        if not liq_data:
            return LiquidationState(
                last_liquidation_time=None,
                last_liquidation_side=None,
                last_liquidation_value=0.0,
            )

        # liq_data could be a single event or list of events
        if isinstance(liq_data, list):
            if not liq_data:
                return LiquidationState(
                    last_liquidation_time=None,
                    last_liquidation_side=None,
                    last_liquidation_value=0.0,
                )
            # Get most recent large liquidation
            for event in reversed(liq_data):
                value = self._get_liquidation_value(event)
                if value >= self._minor_value:
                    return LiquidationState(
                        last_liquidation_time=self._get_event_time(event),
                        last_liquidation_side=self._get_event_side(event),
                        last_liquidation_value=value,
                    )
            return LiquidationState(
                last_liquidation_time=None,
                last_liquidation_side=None,
                last_liquidation_value=0.0,
            )
        else:
            # Single event
            value = self._get_liquidation_value(liq_data)
            if value >= self._minor_value:
                return LiquidationState(
                    last_liquidation_time=self._get_event_time(liq_data),
                    last_liquidation_side=self._get_event_side(liq_data),
                    last_liquidation_value=value,
                )
            return LiquidationState(
                last_liquidation_time=None,
                last_liquidation_side=None,
                last_liquidation_value=0.0,
            )

    def _get_liquidation_value(self, event: dict[str, Any]) -> float:
        """Extract USD value from liquidation event."""
        # Try different possible keys
        # Binance format: {"o": {"q": quantity, "p": price}}
        order = event.get("o", event)
        quantity = float(order.get("q", order.get("quantity", order.get("origQty", 0))))
        price = float(order.get("p", order.get("price", order.get("lastFilledPrice", 0))))

        if quantity > 0 and price > 0:
            return quantity * price

        # Try direct value
        return float(event.get("value", event.get("usdValue", 0)))

    def _get_event_time(self, event: dict[str, Any]) -> datetime:
        """Extract timestamp from liquidation event."""
        # Try different possible keys
        ts = event.get("T", event.get("time", event.get("timestamp")))
        if ts is None:
            order = event.get("o")
            if isinstance(order, dict):
                ts = order.get("T", order.get("time", order.get("timestamp")))
        if ts is None:
            ts = event.get("E")
        if ts:
            if isinstance(ts, int | float):
                # Assume milliseconds
                return datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts)
            return datetime.fromisoformat(str(ts))
        return datetime.now()

    def _get_event_side(self, event: dict[str, Any]) -> str | None:
        side = event.get("side", event.get("S"))
        if side is not None:
            return str(side)
        order = event.get("o")
        if isinstance(order, dict):
            side = order.get("side", order.get("S"))
            if side is not None:
                return str(side)
        return None

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect large liquidation event.

        Args:
            prev_state: Previous LiquidationState
            new_state: Current LiquidationState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if large liquidation detected
        """
        _ = snapshot  # Unused

        if not isinstance(new_state, LiquidationState):
            return None

        # No liquidation above threshold
        if new_state.last_liquidation_time is None:
            return None

        value = new_state.last_liquidation_value
        is_major = value >= self._major_value
        tier = "major" if is_major else "minor"

        # Check cooldown
        now = datetime.now()
        if self._last_signal_time and now - self._last_signal_time < self._cooldown:
            if not is_major or self._last_signal_tier == "major":
                return None

        # Check if this is a new liquidation (not already signaled)
        if (
            prev_state is not None
            and isinstance(prev_state, LiquidationState)
            and prev_state.last_liquidation_time == new_state.last_liquidation_time
        ):
            return None

        signal: Signal | None = None
        side = (new_state.last_liquidation_side or "").upper()
        value_m = new_state.last_liquidation_value / 1_000_000

        if side == "BUY" or side == "LONG":
            # Long liquidation -> forced selling -> bearish momentum
            signal = Signal(
                signal_type=SignalType.LIQUIDATION_LONG,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.STRONG if is_major else SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "side": "LONG",
                    "value_usd": new_state.last_liquidation_value,
                    "value_millions": value_m,
                    "tier": tier,
                    "threshold_usd": self._major_value if is_major else self._minor_value,
                },
                description=(
                    f"LONG Liquidation: ${value_m:.2f}M - Forced selling pressure (bearish)"
                ),
            )

        elif side == "SELL" or side == "SHORT":
            # Short liquidation -> forced buying -> bullish momentum
            signal = Signal(
                signal_type=SignalType.LIQUIDATION_SHORT,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.STRONG if is_major else SignalStrength.MODERATE,
                timeframe=timeframe,
                source=self._name,
                data={
                    "side": "SHORT",
                    "value_usd": new_state.last_liquidation_value,
                    "value_millions": value_m,
                    "tier": tier,
                    "threshold_usd": self._major_value if is_major else self._minor_value,
                },
                description=(
                    f"SHORT Liquidation: ${value_m:.2f}M - Forced buying pressure (bullish)"
                ),
            )

        if signal:
            self._last_signal_time = now
            self._last_signal_tier = tier
            logger.info(f"🎯 {signal.description}")

        return signal
