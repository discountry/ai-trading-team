"""WebSocket management for grid strategy."""

from typing import Callable, Optional, Dict, Any
from weex_sdk import WeexWebSocket
from weex_sdk.logger import get_logger

logger = get_logger("grid_strategy.websocket_manager")


class WebSocketManager:
    """Manages WebSocket connections for grid strategy."""

    def __init__(self, ws: WeexWebSocket):
        """Initialize WebSocketManager.

        Args:
            ws: WeexWebSocket instance
        """
        self.ws = ws
        self._ticker_callback: Optional[Callable] = None
        self._order_callback: Optional[Callable] = None
        self._position_callback: Optional[Callable] = None
        logger.info("WebSocketManager initialized")

    def connect(self) -> None:
        """Connect WebSocket."""
        logger.info("Connecting WebSocket...")
        self.ws.connect()

    def subscribe_ticker(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to ticker updates.

        Args:
            symbol: Trading pair symbol
            callback: Callback function for ticker updates
        """
        self._ticker_callback = callback
        self.ws.subscribe_ticker(symbol, callback=callback)
        logger.info(f"Subscribed to ticker for {symbol}")

    def subscribe_order(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to order updates (private).

        Args:
            callback: Callback function for order updates
        """
        if not self.ws.is_private:
            logger.error("Order channel requires private WebSocket connection")
            raise ValueError("Private WebSocket connection required for order updates")

        self._order_callback = callback
        self.ws.subscribe_order(callback=callback)
        logger.info("Subscribed to order updates")

    def subscribe_position(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to position updates (private).

        Args:
            callback: Callback function for position updates
        """
        if not self.ws.is_private:
            logger.error("Position channel requires private WebSocket connection")
            raise ValueError("Private WebSocket connection required for position updates")

        self._position_callback = callback
        self.ws.subscribe_position(callback=callback)
        logger.info("Subscribed to position updates")

    def close(self) -> None:
        """Close WebSocket connection."""
        logger.info("Closing WebSocket connection...")
        self.ws.close()
