"""Audit log models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentLog:
    """Agent decision log for auditing."""

    # Identification
    log_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Input context
    signal_type: str = ""
    signal_data: dict[str, Any] = field(default_factory=dict)
    market_data: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)
    position: dict[str, Any] | None = None
    orders: list[dict[str, Any]] = field(default_factory=list)

    # Agent decision
    action: str = ""
    command: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    # LLM metadata
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type,
            "signal_data": self.signal_data,
            "market_data": self.market_data,
            "indicators": self.indicators,
            "position": self.position,
            "orders": self.orders,
            "action": self.action,
            "command": self.command,
            "reason": self.reason,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class OrderLog:
    """Order execution log."""

    log_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Order details
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    size: float = 0.0
    price: float | None = None

    # Execution details
    status: str = ""
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    fee: float = 0.0

    # Timing
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    latency_ms: float = 0.0

    # Source
    source: str = ""  # "agent" or "risk"
    agent_log_id: str | None = None  # Link to agent decision

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "size": self.size,
            "price": self.price,
            "status": self.status,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "fee": self.fee,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "latency_ms": self.latency_ms,
            "source": self.source,
            "agent_log_id": self.agent_log_id,
        }
