"""Agent output schemas and decision models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_trading_team.agent.commands import AgentCommand


@dataclass
class AgentDecision:
    """Complete agent decision record for auditing."""

    # Input context
    signal_type: str
    signal_data: dict[str, Any]
    market_snapshot: dict[str, Any]

    # Output
    command: AgentCommand

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type,
            "signal_data": self.signal_data,
            "market_snapshot": self.market_snapshot,
            "command": {
                "action": self.command.action.value,
                "symbol": self.command.symbol,
                "side": self.command.side.value if self.command.side else None,
                "size": self.command.size,
                "price": self.command.price,
                "order_type": self.command.order_type.value if self.command.order_type else None,
                "reason": self.command.reason,
            },
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
        }
