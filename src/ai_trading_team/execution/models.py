"""Execution layer data models."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from ai_trading_team.core.types import OrderStatus, OrderType, PositionStatus, Side, TimeInForce


@dataclass
class Position:
    """Trading position model."""

    symbol: str
    side: Side
    size: Decimal
    entry_price: Decimal
    leverage: int
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    liquidation_price: Decimal | None = None
    margin: Decimal = Decimal("0")
    status: PositionStatus = PositionStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    position_id: str = ""


@dataclass
class Order:
    """Trading order model."""

    symbol: str
    side: Side
    order_type: OrderType
    size: Decimal
    price: Decimal | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_size: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    stop_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    client_order_id: str = ""
    order_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED


@dataclass
class Account:
    """Trading account model."""

    total_equity: Decimal
    available_balance: Decimal
    used_margin: Decimal
    unrealized_pnl: Decimal
    positions: list[Position] = field(default_factory=list)
    open_orders: list[Order] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)
