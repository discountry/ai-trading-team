"""Risk control actions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class RiskAction:
    """Risk control action to be executed."""

    action_type: Literal["close", "close_all", "reduce", "cancel_orders", "move_stop_loss"]
    symbol: str
    reason: str
    priority: int = 0
    reduce_percent: float | None = None  # For "reduce" action
    data: dict[str, Any] | None = None  # Additional context data
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "symbol": self.symbol,
            "reason": self.reason,
            "priority": self.priority,
            "reduce_percent": self.reduce_percent,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
