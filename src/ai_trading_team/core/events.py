"""Event system for inter-module communication."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_trading_team.core.types import EventType


@dataclass
class Event:
    """Base event class for inter-module communication."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
