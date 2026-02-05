"""Order management for grid strategy."""

import time
from typing import List, Optional, Dict
from weex_sdk import WeexClient
from weex_sdk.logger import get_logger
from weex_sdk.exceptions import WeexAPIError
from .grid_level import GridLevel, GridState

logger = get_logger("grid_strategy.order_manager")


class OrderManager:
    """Manages orders for grid strategy."""

    ORDER_TYPE = {"long": {"open": "1", "close": "3"}, "short": {"open": "2", "close": "4"}}

    def __init__(self, client: WeexClient, symbol: str, margin_mode: int):
        """Initialize OrderManager.

        Args:
            client: WeexClient instance
            symbol: Trading pair symbol
            margin_mode: Margin mode (1: Cross, 3: Isolated)
        """
        self.client = client
        self.symbol = symbol
        self.margin_mode = margin_mode
        logger.info(f"OrderManager initialized for {symbol}")

    def place_open_order(self, grid: GridLevel) -> Optional[str]:
        """Place opening order for a grid level.

        Args:
            grid: GridLevel object

        Returns:
            Order ID if successful, None otherwise
        """
        if grid.state != GridState.EMPTY:
            logger.warning(f"Cannot place open order: grid at {grid.price} is not empty")
            return None

        try:
            client_oid = self._generate_client_oid("open", grid.price)

            result = self.client.trade.place_order(
                symbol=self.symbol,
                client_oid=client_oid,
                size=str(grid.size),
                order_type="0",
                match_price="0",
                price=str(grid.price),
                type=self.ORDER_TYPE[grid.direction]["open"],
                margin_mode=self.margin_mode,
            )

            order_id = result.get("order_id")
            if order_id:
                grid.open_order_id = order_id
                grid.state = GridState.OPENING
                logger.info(
                    f"Placed {grid.direction} open order at {grid.price}, order_id: {order_id}"
                )
                return order_id
            else:
                logger.error(f"Failed to place open order: no order_id in response")
                return None

        except WeexAPIError as e:
            logger.error(f"Failed to place open order at {grid.price}: {e}")
            return None

    def place_close_order(self, grid: GridLevel, target_price: float) -> Optional[str]:
        """Place closing order for a grid level.

        Args:
            grid: GridLevel object
            target_price: Target price for closing

        Returns:
            Order ID if successful, None otherwise
        """
        if grid.state != GridState.HOLDING:
            logger.warning(
                f"Cannot place close order: grid at {grid.price} is not holding position"
            )
            return None

        try:
            client_oid = self._generate_client_oid("close", grid.price)

            result = self.client.trade.place_order(
                symbol=self.symbol,
                client_oid=client_oid,
                size=str(grid.size),
                order_type="0",
                match_price="0",
                price=str(target_price),
                type=self.ORDER_TYPE[grid.direction]["close"],
                margin_mode=self.margin_mode,
            )

            order_id = result.get("order_id")
            if order_id:
                grid.close_order_id = order_id
                grid.close_target_price = target_price
                grid.state = GridState.CLOSING
                logger.info(
                    f"Placed {grid.direction} close order at {target_price}, "
                    f"original fill: {grid.filled_price}, order_id: {order_id}"
                )
                return order_id
            else:
                logger.error(f"Failed to place close order: no order_id in response")
                return None

        except WeexAPIError as e:
            logger.error(f"Failed to place close order at {target_price}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            result = self.client.trade.cancel_order(order_id=order_id)
            logger.info(f"Cancelled order {order_id}: {result}")
            return True
        except WeexAPIError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders for symbol.

        Returns:
            True if successful
        """
        try:
            result = self.client.trade.cancel_all_orders(
                cancel_order_type="normal", symbol=self.symbol
            )
            logger.info(f"Cancelled all orders for {self.symbol}: {result}")
            return True
        except WeexAPIError as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions for symbol using market orders.

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.client.trade.close_positions(symbol=self.symbol)
            logger.info(f"Closed all positions for {self.symbol}: {result}")
            return True
        except WeexAPIError as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order status string
        """
        try:
            order = self.client.trade.get_order_detail(order_id)
            return order.get("status")
        except WeexAPIError as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders for symbol.

        Returns:
            List of order dictionaries
        """
        try:
            orders = self.client.trade.get_current_orders(symbol=self.symbol)
            return orders if orders else []
        except WeexAPIError as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(self) -> List[Dict]:
        """Get current positions for symbol.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.client.account.get_single_position(self.symbol)
            return positions.get("position", []) if positions else []
        except WeexAPIError as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_floating_pnl(self) -> float:
        """Calculate total floating PnL from open positions.

        Returns:
            Total floating PnL
        """
        try:
            positions = self.get_positions()
            total_pnl = 0.0
            for pos in positions:
                unrealize_pnl = pos.get("unrealizePnl", "0")
                total_pnl += float(unrealize_pnl)
            return total_pnl
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to calculate floating PnL: {e}")
            return 0.0

    def _generate_client_oid(self, order_type: str, price: float) -> str:
        """Generate unique client order ID.

        Args:
            order_type: 'open' or 'close'
            price: Order price

        Returns:
            Client order ID string
        """
        timestamp = int(time.time() * 1000)
        return f"grid_{order_type}_{self.symbol}_{timestamp}_{int(price)}"
