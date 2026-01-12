"""Abstract exchange interface."""

from abc import ABC, abstractmethod
from decimal import Decimal

from ai_trading_team.core.types import OrderType, Side, TimeInForce
from ai_trading_team.execution.models import Account, Order, Position


class Exchange(ABC):
    """Abstract exchange interface.

    Defines the contract for exchange integrations.
    Implementations must handle REST API and WebSocket synchronization.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if exchange connection is active."""
        ...

    # Connection management

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to exchange."""
        ...

    # Account & Position

    @abstractmethod
    async def get_account(self) -> Account:
        """Get account information."""
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    # Order management

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        size: Decimal,
        price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> Order:
        """Place a new order."""
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders for a symbol.

        Returns:
            Number of orders cancelled
        """
        ...

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order | None:
        """Get order by ID."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Get all open orders for a symbol."""
        ...

    # Position management

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        side: Side,
        size: Decimal | None = None,
    ) -> Order | None:
        """Close a position (fully or partially).

        Args:
            symbol: Trading pair
            side: Position side to close
            size: Size to close (None = close all)

        Returns:
            Closing order, or None if no position to close
        """
        ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        ...
