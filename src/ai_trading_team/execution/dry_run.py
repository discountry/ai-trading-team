"""Dry run executor - simulates trading without real execution."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from ai_trading_team.core.types import OrderStatus, OrderType, PositionStatus, Side
from ai_trading_team.execution.models import Account, Order, Position

logger = logging.getLogger(__name__)


@dataclass
class SimulatedPosition:
    """Internal position tracking for simulation."""

    symbol: str
    side: Side
    size: Decimal
    entry_price: Decimal
    leverage: int = 10
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)


class DryRunExecutor:
    """Dry run executor that simulates trading.

    Maintains virtual account, positions, and orders.
    All logic runs normally but no real orders are placed.
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        default_leverage: int = 10,
    ) -> None:
        """Initialize dry run executor.

        Args:
            initial_balance: Starting virtual balance in USDT
            default_leverage: Default leverage for positions
        """
        self._initial_balance = initial_balance
        self._default_leverage = default_leverage

        # Virtual state
        self._balance = initial_balance
        self._positions: dict[str, SimulatedPosition] = {}
        self._orders: dict[str, Order] = {}
        self._leverage_settings: dict[str, int] = {}
        self._connected = False

        # For simulating market prices (will be updated externally)
        self._current_prices: dict[str, Decimal] = {}

        logger.info(f"[DRY RUN] Initialized with balance: {initial_balance} USDT")

    @property
    def name(self) -> str:
        return "DRY_RUN"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_current_price(self, symbol: str, price: Decimal) -> None:
        """Update current market price for a symbol.

        Args:
            symbol: Trading pair
            price: Current market price
        """
        self._current_prices[symbol] = price
        self._update_unrealized_pnl(symbol)

    def _update_unrealized_pnl(self, symbol: str) -> None:
        """Update unrealized PnL for a position based on current price."""
        if symbol not in self._positions or symbol not in self._current_prices:
            return

        pos = self._positions[symbol]
        current_price = self._current_prices[symbol]

        if pos.side == Side.LONG:
            pnl = (current_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - current_price) * pos.size

        pos.unrealized_pnl = pnl

    async def connect(self) -> None:
        """Simulate connection."""
        self._connected = True
        logger.info("[DRY RUN] Connected (simulated)")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.info("[DRY RUN] Disconnected (simulated)")

    async def get_account(self) -> Account:
        """Get simulated account information."""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())
        used_margin = sum(pos.margin for pos in self._positions.values())

        return Account(
            total_equity=self._balance + total_unrealized_pnl,
            available_balance=self._balance - used_margin,
            used_margin=used_margin if used_margin else Decimal("0"),
            unrealized_pnl=total_unrealized_pnl if total_unrealized_pnl else Decimal("0"),
        )

    async def get_position(self, symbol: str) -> Position | None:
        """Get simulated position for a symbol."""
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        return Position(
            symbol=symbol,
            side=pos.side,
            size=pos.size,
            entry_price=pos.entry_price,
            leverage=pos.leverage,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl,
            margin=pos.margin,
            status=PositionStatus.OPEN,
            created_at=pos.created_at,
            position_id=f"dry_run_{symbol}",
        )

    async def get_positions(self) -> list[Position]:
        """Get all simulated positions."""
        positions = []
        for symbol, pos in self._positions.items():
            positions.append(
                Position(
                    symbol=symbol,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    leverage=pos.leverage,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    margin=pos.margin,
                    status=PositionStatus.OPEN,
                    created_at=pos.created_at,
                    position_id=f"dry_run_{symbol}",
                )
            )
        return positions

    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        size: float,
        price: float | None = None,
        action: str = "open",
    ) -> Order:
        """Simulate placing an order.

        For MARKET orders, immediately fills at current price.
        For LIMIT orders, stores as pending (simplified - doesn't simulate matching).
        """
        order_id = f"dry_{uuid.uuid4().hex[:12]}"
        client_oid = f"dry_client_{uuid.uuid4().hex[:8]}"

        # Get execution price
        exec_price = (
            Decimal(str(price)) if price else self._current_prices.get(symbol, Decimal("0"))
        )

        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=Decimal(str(size)),
            price=exec_price if price else None,
            status=OrderStatus.NEW,
            client_order_id=client_oid,
            order_id=order_id,
        )

        logger.info(
            f"[DRY RUN] Placing order: {action} {side.value} {size} {symbol} "
            f"@ {exec_price if exec_price else 'MARKET'}"
        )

        # Simulate immediate fill for MARKET orders
        if order_type == OrderType.MARKET:
            order = await self._fill_order(order, exec_price, action)
        else:
            # Store LIMIT orders as pending
            self._orders[order_id] = order
            logger.info(f"[DRY RUN] Limit order {order_id} pending")

        return order

    async def _fill_order(self, order: Order, fill_price: Decimal, action: str) -> Order:
        """Simulate order fill and update position."""
        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.avg_fill_price = fill_price

        symbol = order.symbol
        size = order.size
        leverage = self._leverage_settings.get(symbol, self._default_leverage)

        if action == "open":
            # Opening or adding to position
            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos.side == order.side:
                    # Adding to existing position - average entry price
                    total_size = pos.size + size
                    pos.entry_price = (
                        (pos.entry_price * pos.size) + (fill_price * size)
                    ) / total_size
                    pos.size = total_size
                    pos.margin = (total_size * pos.entry_price) / Decimal(leverage)
                else:
                    # Reducing opposite position
                    if size >= pos.size:
                        # Close and potentially open reverse
                        self._realize_pnl(symbol, pos.size, fill_price)
                        remaining = size - pos.size
                        if remaining > 0:
                            self._positions[symbol] = SimulatedPosition(
                                symbol=symbol,
                                side=order.side,
                                size=remaining,
                                entry_price=fill_price,
                                leverage=leverage,
                                margin=(remaining * fill_price) / Decimal(leverage),
                            )
                        else:
                            del self._positions[symbol]
                    else:
                        self._realize_pnl(symbol, size, fill_price)
                        pos.size -= size
                        pos.margin = (pos.size * pos.entry_price) / Decimal(leverage)
            else:
                # New position
                margin = (size * fill_price) / Decimal(leverage)
                self._positions[symbol] = SimulatedPosition(
                    symbol=symbol,
                    side=order.side,
                    size=size,
                    entry_price=fill_price,
                    leverage=leverage,
                    margin=margin,
                )

            logger.info(
                f"[DRY RUN] Opened {order.side.value} position: {size} {symbol} @ {fill_price}"
            )

        elif action == "close":
            # Closing position
            if symbol in self._positions:
                pos = self._positions[symbol]
                close_size = min(size, pos.size)
                self._realize_pnl(symbol, close_size, fill_price)

                if close_size >= pos.size:
                    del self._positions[symbol]
                    logger.info(f"[DRY RUN] Closed position: {symbol}")
                else:
                    pos.size -= close_size
                    pos.margin = (pos.size * pos.entry_price) / Decimal(pos.leverage)
                    logger.info(
                        f"[DRY RUN] Partially closed {close_size} {symbol}, remaining: {pos.size}"
                    )

        return order

    def _realize_pnl(self, symbol: str, size: Decimal, close_price: Decimal) -> None:
        """Realize PnL for a closed position."""
        if symbol not in self._positions:
            return

        pos = self._positions[symbol]
        if pos.side == Side.LONG:
            pnl = (close_price - pos.entry_price) * size
        else:
            pnl = (pos.entry_price - close_price) * size

        pos.realized_pnl += pnl
        self._balance += pnl
        logger.info(f"[DRY RUN] Realized PnL: {pnl:+.4f} USDT")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Simulate cancelling an order."""
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELED
            del self._orders[order_id]
            logger.info(f"[DRY RUN] Cancelled order: {order_id}")
            return True
        _ = symbol  # unused but kept for interface compatibility
        logger.warning(f"[DRY RUN] Order not found: {order_id}")
        return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Simulate cancelling all orders for a symbol."""
        cancelled = 0
        to_remove = []
        for order_id, order in self._orders.items():
            if order.symbol == symbol:
                order.status = OrderStatus.CANCELED
                to_remove.append(order_id)
                cancelled += 1

        for order_id in to_remove:
            del self._orders[order_id]

        logger.info(f"[DRY RUN] Cancelled {cancelled} orders for {symbol}")
        return cancelled

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Get simulated open orders for a symbol."""
        return [
            order for order in self._orders.values() if order.symbol == symbol and order.is_open
        ]

    async def close_position(
        self, symbol: str, side: Side, size: float | None = None
    ) -> Order | None:
        """Simulate closing a position."""
        if symbol not in self._positions:
            logger.warning(f"[DRY RUN] No position to close for {symbol}")
            return None

        pos = self._positions[symbol]
        close_size = Decimal(str(size)) if size else pos.size
        exec_price = self._current_prices.get(symbol, pos.entry_price)

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=close_size,
            status=OrderStatus.NEW,
            order_id=f"dry_close_{uuid.uuid4().hex[:8]}",
        )

        return await self._fill_order(order, exec_price, "close")

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        self._leverage_settings[symbol] = leverage
        logger.info(f"[DRY RUN] Set leverage to {leverage}x for {symbol}")
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
        """Simulate uploading AI log (just logs locally)."""
        logger.info(
            f"[DRY RUN] AI Log - Stage: {stage}, Model: {model}, "
            f"Explanation: {explanation[:100]}..."
        )
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get simulation statistics."""
        total_unrealized = sum(pos.unrealized_pnl for pos in self._positions.values())
        total_realized = sum(pos.realized_pnl for pos in self._positions.values())

        return {
            "initial_balance": float(self._initial_balance),
            "current_balance": float(self._balance),
            "unrealized_pnl": float(total_unrealized),
            "realized_pnl": float(total_realized),
            "total_equity": float(self._balance + total_unrealized),
            "open_positions": len(self._positions),
            "pending_orders": len(self._orders),
            "pnl_percent": float(
                ((self._balance - self._initial_balance) / self._initial_balance) * 100
            ),
        }

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._current_prices.clear()
        logger.info("[DRY RUN] Simulation reset")
