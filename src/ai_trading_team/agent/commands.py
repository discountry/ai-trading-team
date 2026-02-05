"""Agent command definitions."""

from dataclasses import dataclass
from enum import Enum

from ai_trading_team.core.types import OrderType, Side


class AgentAction(str, Enum):
    """Actions that agent can take."""

    OPEN = "open"  # Open a new position
    CLOSE = "close"  # Close existing position
    ADD = "add"  # Add to existing position
    REDUCE = "reduce"  # Reduce existing position
    CANCEL = "cancel"  # Cancel pending order
    OBSERVE = "observe"  # Do nothing, just observe
    MOVE_STOP_LOSS = "move_stop_loss"  # Move stop loss order


@dataclass
class AgentCommand:
    """Structured command from agent decision."""

    action: AgentAction
    symbol: str
    reason: str  # Agent's explanation (required for auditing)
    side: Side | None = None
    size: float | None = None
    price: float | None = None
    order_type: OrderType | None = None
    stop_loss_price: float | None = None  # Stop loss price for MOVE_STOP_LOSS action
    take_profit_price: float | None = None  # Take profit price for OPEN/ADD actions

    def is_actionable(self) -> bool:
        """Check if command requires execution."""
        return self.action != AgentAction.OBSERVE

    def validate(self) -> list[str]:
        """Validate command fields.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.action in (AgentAction.OPEN, AgentAction.ADD):
            if self.side is None:
                errors.append("side is required for open/add actions")
            if self.size is None or self.size <= 0:
                errors.append("positive size is required for open/add actions")

        if self.action == AgentAction.REDUCE and (self.size is None or self.size <= 0):
            errors.append("positive size is required for reduce action")

        if self.action == AgentAction.CLOSE and self.side is None:
            errors.append("side is required for close action")

        if self.action == AgentAction.MOVE_STOP_LOSS and (
            self.stop_loss_price is None or self.stop_loss_price <= 0
        ):
            errors.append("stop_loss_price is required for move_stop_loss action")

        if self.action in (AgentAction.OPEN, AgentAction.ADD) and (
            self.take_profit_price is None or self.take_profit_price <= 0
        ):
            errors.append("take_profit_price is required for open/add actions")

        if self.order_type == OrderType.LIMIT and self.price is None:
            errors.append("price is required for limit orders")

        if not self.reason:
            errors.append("reason is required for auditing")

        return errors
