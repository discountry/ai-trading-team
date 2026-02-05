"""Grid level data model."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class GridState(Enum):
    """Grid level states."""

    EMPTY = "empty"
    OPENING = "opening"
    HOLDING = "holding"
    CLOSING = "closing"


@dataclass
class GridLevel:
    """Represents a single grid level."""

    price: float
    size: float
    direction: str
    state: GridState = GridState.EMPTY

    open_order_id: Optional[str] = None
    close_order_id: Optional[str] = None

    filled_price: Optional[float] = None
    filled_time: Optional[int] = None
    close_target_price: Optional[float] = None

    realized_pnl: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "price": self.price,
            "size": self.size,
            "direction": self.direction,
            "state": self.state.value,
            "open_order_id": self.open_order_id,
            "close_order_id": self.close_order_id,
            "filled_price": self.filled_price,
            "filled_time": self.filled_time,
            "close_target_price": self.close_target_price,
            "realized_pnl": self.realized_pnl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridLevel":
        """Create GridLevel from dictionary."""
        return cls(
            price=data["price"],
            size=data["size"],
            direction=data["direction"],
            state=GridState(data["state"]),
            open_order_id=data.get("open_order_id"),
            close_order_id=data.get("close_order_id"),
            filled_price=data.get("filled_price"),
            filled_time=data.get("filled_time"),
            close_target_price=data.get("close_target_price"),
            realized_pnl=data.get("realized_pnl", 0.0),
        )
