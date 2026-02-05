"""Mock executor for DRY_RUN mode.

Simulates account, positions, and orders locally without connecting to any exchange.
Uses Binance market data for price simulation.
"""

import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.core.types import OrderStatus, OrderType, PositionStatus, Side
from ai_trading_team.execution.models import Account, Order, Position

logger = logging.getLogger(__name__)


class MockExecutor:
    """Mock exchange executor for simulation.

    Simulates trading without real exchange connection.
    Uses DataPool for market prices to calculate P&L.
    """

    def __init__(
        self,
        data_pool: DataPool,
        initial_balance: Decimal = Decimal("1000"),
        leverage: int = 75,
    ) -> None:
        """Initialize mock executor.

        Args:
            data_pool: Data pool for market prices
            initial_balance: Initial account balance in USDT
            leverage: Default leverage
        """
        self._data_pool = data_pool
        self._initial_balance = initial_balance
        self._leverage = leverage

        # Simulated state
        self._balance = initial_balance
        self._position: Position | None = None
        self._open_orders: list[Order] = []
        self._order_history: list[Order] = []
        self._realized_pnl = Decimal("0")

        self._connected = False

    @property
    def name(self) -> str:
        return "MOCK"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Simulate connection."""
        self._connected = True
        logger.info("[MOCK] Connected to mock executor")
        logger.info(f"[MOCK] Initial balance: ${self._initial_balance}")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.info("[MOCK] Disconnected from mock executor")

    def _get_current_price(self) -> Decimal:
        """Get current market price from data pool."""
        ticker = self._data_pool.ticker
        if ticker:
            return Decimal(str(ticker.get("last_price", 0)))
        return Decimal("0")

    def _calculate_unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L for current position."""
        if not self._position or self._position.size == 0:
            return Decimal("0")

        current_price = self._get_current_price()
        if current_price == 0:
            return Decimal("0")

        entry_price = self._position.entry_price
        size = self._position.size

        # P&L calculation depends on position side
        if self._position.side == Side.LONG:
            # Long: profit when price goes up
            price_diff = current_price - entry_price
        else:
            # Short: profit when price goes down
            price_diff = entry_price - current_price

        # P&L = size * price_diff (simplified for perpetual contracts)
        pnl = size * price_diff

        return pnl

    def _calculate_pnl_percent(self) -> Decimal:
        """Calculate P&L as percentage of margin."""
        if not self._position or self._position.margin == 0:
            return Decimal("0")

        unrealized_pnl = self._calculate_unrealized_pnl()
        return (unrealized_pnl / self._position.margin) * Decimal("100")

    async def get_account(self) -> Account:
        """Get simulated account information."""
        unrealized_pnl = self._calculate_unrealized_pnl()
        used_margin = self._position.margin if self._position else Decimal("0")

        return Account(
            total_equity=self._balance + unrealized_pnl,
            available_balance=self._balance - used_margin,
            used_margin=used_margin,
            unrealized_pnl=unrealized_pnl,
        )

    async def get_position(self, symbol: str) -> Position | None:
        """Get simulated position for a symbol."""
        if not self._position or self._position.symbol != symbol:
            return None

        if self._position.size == 0:
            return None

        # Update unrealized P&L
        self._position.unrealized_pnl = self._calculate_unrealized_pnl()

        return self._position

    async def get_positions(self) -> list[Position]:
        """Get all simulated positions."""
        if self._position and self._position.size > 0:
            self._position.unrealized_pnl = self._calculate_unrealized_pnl()
            return [self._position]
        return []

    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        size: float,
        price: float | None = None,
        action: str = "open",
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> Order:
        """Place a simulated order.

        For market orders, executes immediately.
        For limit orders, adds to pending orders.

        Args:
            symbol: Trading pair
            side: Order side
            order_type: Order type
            size: Order size
            price: Limit price
            action: "open" or "close"
            stop_loss_price: Preset stop loss price (logged in mock mode)
            take_profit_price: Preset take profit price (logged in mock mode)
        """
        # Log stop loss/take profit for debugging in mock mode
        if action == "open":
            if stop_loss_price is not None:
                logger.info(f"[MOCK] Preset stop loss: {stop_loss_price}")
            if take_profit_price is not None:
                logger.info(f"[MOCK] Preset take profit: {take_profit_price}")

        client_oid = f"mock_{uuid.uuid4().hex[:16]}"
        order_id = f"mock_order_{uuid.uuid4().hex[:8]}"

        current_price = self._get_current_price()
        exec_price = Decimal(str(price)) if price else current_price

        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=Decimal(str(size)),
            price=exec_price,
            status=OrderStatus.NEW,
            client_order_id=client_oid,
            order_id=order_id,
            created_at=datetime.now(),
        )

        # Market orders execute immediately
        if order_type == OrderType.MARKET:
            await self._execute_order(order, action, current_price)
        else:
            # Limit orders go to pending
            self._open_orders.append(order)
            logger.info(f"[MOCK] Limit order placed: {order_id} @ {exec_price}")

        return order

    async def _execute_order(self, order: Order, action: str, exec_price: Decimal) -> None:
        """Execute an order and update position."""
        size = order.size

        if action == "open":
            # Calculate margin required
            # margin = size * price / leverage (simplified)
            margin = (size * exec_price) / Decimal(str(self._leverage))

            if margin > self._balance:
                logger.warning(f"[MOCK] Insufficient margin: need {margin}, have {self._balance}")
                order.status = OrderStatus.REJECTED
                return

            if self._position and self._position.side == order.side:
                # Add to existing position
                old_size = self._position.size
                old_value = old_size * self._position.entry_price
                new_value = size * exec_price
                total_size = old_size + size
                avg_price = (old_value + new_value) / total_size

                self._position.size = total_size
                self._position.entry_price = avg_price
                self._position.margin += margin

                logger.info(
                    f"[MOCK] Added to position: {order.side.value} {size} @ {exec_price}, "
                    f"total size: {total_size}"
                )
            else:
                # Open new position (close existing if opposite side)
                if self._position and self._position.size > 0:
                    # Close existing position first
                    await self._close_position_internal()

                self._position = Position(
                    symbol=order.symbol,
                    side=order.side,
                    size=size,
                    entry_price=exec_price,
                    leverage=self._leverage,
                    margin=margin,
                    status=PositionStatus.OPEN,
                    created_at=datetime.now(),
                )

                logger.info(
                    f"[MOCK] Opened position: {order.side.value} {size} @ {exec_price}, "
                    f"margin: {margin:.2f}"
                )

            self._balance -= margin
            order.status = OrderStatus.FILLED
            order.filled_size = size
            order.avg_fill_price = exec_price

        else:  # close
            if self._position:
                await self._close_position_internal(size)
            order.status = OrderStatus.FILLED
            order.filled_size = size
            order.avg_fill_price = exec_price

        self._order_history.append(order)
        if len(self._order_history) > 1000:
            self._order_history = self._order_history[-1000:]

    async def _close_position_internal(self, size: Decimal | None = None) -> Decimal:
        """Close position and return realized P&L."""
        if not self._position:
            return Decimal("0")

        close_size = size if size else self._position.size
        if close_size > self._position.size:
            close_size = self._position.size

        current_price = self._get_current_price()

        # Calculate P&L for closed portion
        if self._position.side == Side.LONG:
            price_diff = current_price - self._position.entry_price
        else:
            price_diff = self._position.entry_price - current_price

        pnl = close_size * price_diff

        # Calculate margin to return
        margin_ratio = close_size / self._position.size
        returned_margin = self._position.margin * margin_ratio

        # Update balance
        self._balance += returned_margin + pnl
        self._realized_pnl += pnl

        logger.info(
            f"[MOCK] Closed position: {close_size} @ {current_price}, "
            f"P&L: {pnl:+.2f}, returned margin: {returned_margin:.2f}"
        )

        # Update or clear position
        if close_size >= self._position.size:
            self._position = None
        else:
            self._position.size -= close_size
            self._position.margin -= returned_margin

        return pnl

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self._open_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELED
                self._open_orders.remove(order)
                logger.info(f"[MOCK] Cancelled order: {order_id}")
                return True
        return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all pending orders for a symbol."""
        count = 0
        orders_to_remove = []
        for order in self._open_orders:
            if order.symbol == symbol:
                order.status = OrderStatus.CANCELED
                orders_to_remove.append(order)
                count += 1

        for order in orders_to_remove:
            self._open_orders.remove(order)

        logger.info(f"[MOCK] Cancelled {count} orders for {symbol}")
        return count

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Get pending orders for a symbol."""
        return [o for o in self._open_orders if o.symbol == symbol]

    async def close_position(
        self, symbol: str, side: Side, size: float | None = None
    ) -> Order | None:
        """Close a position."""
        if not self._position or self._position.symbol != symbol:
            logger.warning(f"[MOCK] No position to close for {symbol}")
            return None

        close_size = Decimal(str(size)) if size else self._position.size
        current_price = self._get_current_price()

        pnl = await self._close_position_internal(close_size)

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=close_size,
            price=current_price,
            status=OrderStatus.FILLED,
            filled_size=close_size,
            avg_fill_price=current_price,
            order_id=f"mock_close_{uuid.uuid4().hex[:8]}",
        )

        logger.info(f"[MOCK] Position closed with P&L: {pnl:+.2f}")
        return order

    async def reduce_position(self, symbol: str, side: Side, size: float) -> Order | None:
        """Partially close a position.

        Args:
            symbol: Trading pair
            side: Position side to reduce
            size: Size to close (must be positive)

        Returns:
            Order object if successful
        """
        if size <= 0:
            logger.error(f"[MOCK] Invalid reduce size: {size}")
            return None

        if not self._position or self._position.symbol != symbol:
            logger.warning(f"[MOCK] No position to reduce for {symbol}")
            return None

        logger.info(f"[MOCK] Reducing position by {size}")
        return await self.close_position(symbol, side, size)

    async def add_to_position(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_loss_price: float | None = None,
    ) -> Order | None:
        """Add to an existing position.

        Args:
            symbol: Trading pair
            side: Position side to add to
            size: Size to add
            stop_loss_price: Optional stop loss price

        Returns:
            Order object if successful
        """
        if size <= 0:
            logger.error(f"[MOCK] Invalid add size: {size}")
            return None

        try:
            order = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=size,
                action="open",
                stop_loss_price=stop_loss_price,
            )
            logger.info(f"[MOCK] Added {size} to position: {order.order_id}")
            return order
        except Exception as e:
            logger.error(f"[MOCK] Failed to add to position: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage (just store it)."""
        _ = symbol  # Symbol-specific leverage not implemented in mock
        self._leverage = leverage
        logger.info(f"[MOCK] Leverage set to {leverage}x")
        return True

    async def upload_ai_log(
        self,
        stage: str,
        model: str,
        input_data: dict[str, Any],
        output: dict[str, Any],
        explanation: str,
        order_id: int | None = None,
    ) -> bool:
        """Log AI decision (just log locally in mock mode)."""
        # Store for potential debugging
        _ = (input_data, explanation, order_id)
        logger.info(
            f"[MOCK] AI Log - Stage: {stage}, Model: {model}, "
            f"Output: {output.get('action', 'unknown')}"
        )
        return True

    def get_summary(self) -> dict[str, Any]:
        """Get summary of mock trading session."""
        current_equity = self._balance
        if self._position:
            current_equity += self._calculate_unrealized_pnl()

        return {
            "initial_balance": float(self._initial_balance),
            "current_balance": float(self._balance),
            "current_equity": float(current_equity),
            "realized_pnl": float(self._realized_pnl),
            "unrealized_pnl": float(self._calculate_unrealized_pnl()),
            "total_pnl": float(current_equity - self._initial_balance),
            "total_orders": len(self._order_history),
            "has_position": self._position is not None,
        }
