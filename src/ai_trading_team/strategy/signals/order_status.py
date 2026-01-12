"""Order status change signal source.

Detects when open orders change status (filled, cancelled, partial fill).
This helps the agent track order execution events.
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
class OrderState:
    """State for order tracking."""

    order_ids: set[str]  # Current open order IDs
    order_count: int


class OrderStatusSignal(SignalSource):
    """Detects order status changes.

    Signals:
    - ORDER_FILLED: Order was completely filled
    - ORDER_CANCELLED: Order was cancelled
    - ORDER_PARTIAL_FILL: Order was partially filled

    These signals help the agent track order execution.
    """

    def __init__(
        self,
        timeframes: list[Timeframe] | None = None,
    ) -> None:
        """Initialize order status signal source.

        Args:
            timeframes: Timeframes to monitor (default: 1h only)
        """
        super().__init__(
            name="order_status",
            timeframes=timeframes or [Timeframe.H1],
        )
        self._known_orders: dict[str, dict[str, Any]] = {}

    def _compute_state(
        self,
        snapshot: DataSnapshot,
        timeframe: Timeframe,
    ) -> OrderState | None:
        """Compute current order state.

        Args:
            snapshot: Current market data
            timeframe: Timeframe context

        Returns:
            Current state
        """
        _ = timeframe  # Orders are global

        # Get open orders from snapshot
        open_orders = snapshot.orders or []

        order_ids = set()
        for order in open_orders:
            order_id = order.get("orderId", order.get("order_id", order.get("id")))
            if order_id:
                order_ids.add(str(order_id))
                # Track order details
                self._known_orders[str(order_id)] = order

        return OrderState(
            order_ids=order_ids,
            order_count=len(order_ids),
        )

    def _detect_transition(
        self,
        prev_state: Any,
        new_state: Any,
        timeframe: Timeframe,
        snapshot: DataSnapshot,
    ) -> Signal | None:
        """Detect order status change.

        Args:
            prev_state: Previous OrderState
            new_state: Current OrderState
            timeframe: Timeframe context
            snapshot: Current market data

        Returns:
            Signal if order status changed
        """
        if not isinstance(new_state, OrderState):
            return None

        # First update - no transition
        if prev_state is None:
            logger.debug(f"[{self._name}] initialized: {new_state.order_count} open orders")
            return None

        if not isinstance(prev_state, OrderState):
            return None

        # Find orders that disappeared (filled or cancelled)
        disappeared_orders = prev_state.order_ids - new_state.order_ids

        if not disappeared_orders:
            return None

        # Check recent operations to determine what happened
        recent_operations = snapshot.recent_operations or []
        recent_ops = recent_operations[-10:] if recent_operations else []

        for order_id in disappeared_orders:
            order_details = self._known_orders.get(order_id, {})

            # Try to find matching operation
            order_op = None
            for op in reversed(recent_ops):
                if str(op.get("order_id", op.get("orderId", ""))) == order_id:
                    order_op = op
                    break

            signal: Signal | None = None

            if order_op:
                status = order_op.get("status", order_op.get("action", "")).lower()

                if status in ["filled", "fill", "executed"]:
                    signal = Signal(
                        signal_type=SignalType.ORDER_FILLED,
                        direction=SignalDirection.NEUTRAL,
                        strength=SignalStrength.MODERATE,
                        timeframe=timeframe,
                        source=self._name,
                        data={
                            "order_id": order_id,
                            "order_details": order_details,
                            "operation": order_op,
                        },
                        description=(
                            f"Order FILLED: {order_id} - "
                            f"{order_details.get('side', 'N/A')} "
                            f"{order_details.get('quantity', order_details.get('origQty', 'N/A'))}"
                        ),
                    )

                elif status in ["cancelled", "cancel", "canceled"]:
                    signal = Signal(
                        signal_type=SignalType.ORDER_CANCELLED,
                        direction=SignalDirection.NEUTRAL,
                        strength=SignalStrength.WEAK,
                        timeframe=timeframe,
                        source=self._name,
                        data={
                            "order_id": order_id,
                            "order_details": order_details,
                            "operation": order_op,
                        },
                        description=(
                            f"Order CANCELLED: {order_id} - "
                            f"{order_details.get('side', 'N/A')} "
                            f"{order_details.get('quantity', order_details.get('origQty', 'N/A'))}"
                        ),
                    )

                elif status in ["partial", "partial_fill", "partially_filled"]:
                    signal = Signal(
                        signal_type=SignalType.ORDER_PARTIAL_FILL,
                        direction=SignalDirection.NEUTRAL,
                        strength=SignalStrength.WEAK,
                        timeframe=timeframe,
                        source=self._name,
                        data={
                            "order_id": order_id,
                            "order_details": order_details,
                            "operation": order_op,
                        },
                        description=(
                            f"Order PARTIAL FILL: {order_id} - {order_details.get('side', 'N/A')}"
                        ),
                    )

            else:
                # No operation found, assume filled
                signal = Signal(
                    signal_type=SignalType.ORDER_FILLED,
                    direction=SignalDirection.NEUTRAL,
                    strength=SignalStrength.MODERATE,
                    timeframe=timeframe,
                    source=self._name,
                    data={
                        "order_id": order_id,
                        "order_details": order_details,
                    },
                    description=(
                        f"Order completed: {order_id} - "
                        f"{order_details.get('side', 'N/A')} "
                        f"{order_details.get('quantity', order_details.get('origQty', 'N/A'))}"
                    ),
                )

            if signal:
                logger.info(f"📋 {signal.description}")
                # Clean up tracked order
                self._known_orders.pop(order_id, None)
                # Return first signal (one at a time)
                return signal

        return None

    def reset(self, timeframe: Timeframe | None = None) -> None:
        """Reset all states including known orders.

        Args:
            timeframe: Specific timeframe to reset, or None for all
        """
        super().reset(timeframe)
        if timeframe is None:
            # Full reset
            self._known_orders.clear()
