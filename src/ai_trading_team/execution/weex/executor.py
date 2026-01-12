"""WEEX exchange executor implementation."""

import logging
import uuid
from decimal import Decimal
from typing import Any

from weex_sdk import AsyncWeexClient

from ai_trading_team.core.types import OrderStatus, OrderType, PositionStatus, Side
from ai_trading_team.execution.models import Account, Order, Position

logger = logging.getLogger(__name__)


class WEEXExecutor:
    """WEEX exchange implementation.

    Combines REST API and WebSocket for order execution
    and real-time data synchronization.

    Note: This class implements the Exchange interface but does not formally
    inherit from it due to signature differences. The RiskMonitor accepts
    this class via duck typing.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
    ) -> None:
        """Initialize WEEX executor.

        Args:
            api_key: WEEX API key
            api_secret: WEEX API secret
            passphrase: WEEX API passphrase
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._client: AsyncWeexClient | None = None
        self._connected = False

    @property
    def name(self) -> str:
        return "WEEX"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establish connection to WEEX."""
        self._client = AsyncWeexClient(
            api_key=self._api_key,
            secret_key=self._api_secret,
            passphrase=self._passphrase,
        )
        # AsyncWeexClient uses context manager but we can start it manually
        await self._client.__aenter__()
        self._connected = True
        logger.info("Connected to WEEX")

    async def disconnect(self) -> None:
        """Close connection to WEEX."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
        self._connected = False
        logger.info("Disconnected from WEEX")

    def _ensure_connected(self) -> AsyncWeexClient:
        """Ensure client is connected and return it."""
        if not self._client or not self._connected:
            raise RuntimeError("Not connected to WEEX")
        return self._client

    async def get_account(self) -> Account:
        """Get account information."""
        client = self._ensure_connected()
        accounts = await client.account.get_accounts()

        # Find USDT account
        total_equity = Decimal("0")
        available_balance = Decimal("0")
        used_margin = Decimal("0")
        unrealized_pnl = Decimal("0")

        for acc in accounts if isinstance(accounts, list) else [accounts]:
            if acc.get("coin") == "USDT" or acc.get("marginCoin") == "USDT":
                total_equity = Decimal(str(acc.get("equity", 0)))
                available_balance = Decimal(str(acc.get("available", 0)))
                used_margin = Decimal(str(acc.get("frozen", 0)))
                unrealized_pnl = Decimal(str(acc.get("unrealisedPL", 0)))
                break

        return Account(
            total_equity=total_equity,
            available_balance=available_balance,
            used_margin=used_margin,
            unrealized_pnl=unrealized_pnl,
        )

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        client = self._ensure_connected()
        data = await client.account.get_single_position(symbol)

        if not data:
            return None

        # WEEX returns position data
        size = Decimal(str(data.get("total", 0)))
        if size == 0:
            return None

        side_str = data.get("holdSide", "long")
        side = Side.LONG if side_str.lower() == "long" else Side.SHORT

        return Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=Decimal(str(data.get("averageOpenPrice", 0))),
            leverage=int(data.get("leverage", 1)),
            unrealized_pnl=Decimal(str(data.get("unrealisedPL", 0))),
            realized_pnl=Decimal(str(data.get("realisedPL", 0))),
            liquidation_price=Decimal(str(data.get("liquidationPrice", 0)))
            if data.get("liquidationPrice")
            else None,
            margin=Decimal(str(data.get("margin", 0))),
            status=PositionStatus.OPEN,
        )

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        client = self._ensure_connected()
        data = await client.account.get_all_positions()

        positions = []
        for item in data if isinstance(data, list) else []:
            size = Decimal(str(item.get("total", 0)))
            if size > 0:
                side_str = item.get("holdSide", "long")
                side = Side.LONG if side_str.lower() == "long" else Side.SHORT

                positions.append(
                    Position(
                        symbol=item.get("symbol", ""),
                        side=side,
                        size=size,
                        entry_price=Decimal(str(item.get("averageOpenPrice", 0))),
                        leverage=int(item.get("leverage", 1)),
                        unrealized_pnl=Decimal(str(item.get("unrealisedPL", 0))),
                        status=PositionStatus.OPEN,
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
        action: str = "open",  # open/close
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> Order:
        """Place a new order.

        Args:
            symbol: Trading pair (e.g., "cmt_btcusdt")
            side: Order side (LONG/SHORT)
            order_type: Order type (MARKET/LIMIT)
            size: Order size
            price: Limit price (required for LIMIT orders)
            action: "open" for opening position, "close" for closing
            stop_loss_price: Preset stop loss price (for open orders)
            take_profit_price: Preset take profit price (for open orders)

        Returns:
            Created Order object
        """
        client = self._ensure_connected()

        # Generate unique client order ID
        client_oid = f"mvp_{uuid.uuid4().hex[:16]}"

        # Map to WEEX order type codes
        # Type codes: 1=open long, 2=open short, 3=close long, 4=close short
        if action == "open":
            type_code = "1" if side == Side.LONG else "2"
        else:
            type_code = "3" if side == Side.LONG else "4"

        # Order type: 0=normal, 1=post only, 2=FOK, 3=IOC
        weex_order_type = "0"

        # match_price: 0=limit, 1=market
        match_price = "1" if order_type == OrderType.MARKET else "0"

        order_params: dict[str, Any] = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": str(size),
            "order_type": weex_order_type,
            "match_price": match_price,
            "type": type_code,
        }

        if order_type == OrderType.LIMIT and price is not None:
            order_params["price"] = str(price)

        # Add preset stop loss if provided (for opening positions)
        if action == "open" and stop_loss_price is not None:
            order_params["preset_stop_loss_price"] = str(stop_loss_price)
            logger.info(f"Setting preset stop loss at {stop_loss_price}")

        # Add preset take profit if provided
        if action == "open" and take_profit_price is not None:
            order_params["preset_take_profit_price"] = str(take_profit_price)

        logger.info(f"Placing order: {order_params}")

        result = await client.trade.place_order(**order_params)

        return Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=Decimal(str(size)),
            price=Decimal(str(price)) if price else None,
            status=OrderStatus.NEW,
            client_order_id=client_oid,
            order_id=str(result.get("order_id", "")),
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        client = self._ensure_connected()

        try:
            await client.trade.cancel_order(order_id=order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders for a symbol."""
        client = self._ensure_connected()

        try:
            result = await client.trade.cancel_all_orders(
                cancel_order_type="normal",
                symbol=symbol,
            )
            count = result.get("count", 0) if isinstance(result, dict) else 0
            logger.info(f"Cancelled {count} orders for {symbol}")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Get all open orders for a symbol."""
        client = self._ensure_connected()

        data = await client.trade.get_current_orders(symbol=symbol)
        orders = []

        for item in data if isinstance(data, list) else []:
            side_str = item.get("side", "1")
            side = Side.LONG if side_str in ("1", "3") else Side.SHORT

            orders.append(
                Order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    size=Decimal(str(item.get("size", 0))),
                    price=Decimal(str(item.get("price", 0))),
                    status=OrderStatus.NEW,
                    order_id=str(item.get("orderId", "")),
                    filled_size=Decimal(str(item.get("filledSize", 0))),
                )
            )

        return orders

    async def close_position(
        self, symbol: str, side: Side, size: float | None = None
    ) -> Order | None:
        """Close a position.

        Args:
            symbol: Trading pair
            side: Position side to close
            size: Size to close (None = close all)

        Returns:
            Order object if successful
        """
        client = self._ensure_connected()

        try:
            # Use close_positions for market close
            result = await client.trade.close_positions(symbol=symbol)
            logger.info(f"Closed position for {symbol}: {result}")

            # Return a placeholder order
            return Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=Decimal(str(size)) if size else Decimal("0"),
                status=OrderStatus.FILLED,
            )
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None

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
            logger.error(f"Invalid reduce size: {size}")
            return None

        try:
            # Use place_order with action="close" for partial close
            order = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=size,
                action="close",
            )
            logger.info(f"Reduced position by {size}: {order.order_id}")
            return order
        except Exception as e:
            logger.error(f"Failed to reduce position: {e}")
            return None

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
            logger.error(f"Invalid add size: {size}")
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
            logger.info(f"Added {size} to position: {order.order_id}")
            return order
        except Exception as e:
            logger.error(f"Failed to add to position: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        client = self._ensure_connected()

        try:
            await client.account.set_leverage(
                symbol=symbol,
                margin_mode=1,  # Cross margin
                long_leverage=str(leverage),
                short_leverage=str(leverage),
            )
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    async def upload_ai_log(
        self,
        stage: str,
        model: str,
        input_data: dict[str, Any],
        output: dict[str, Any],
        explanation: str,
        order_id: int | None = None,
    ) -> bool:
        """Upload AI decision log to WEEX.

        Args:
            stage: Decision stage (e.g., "Decision Making")
            model: AI model name
            input_data: Input data to AI
            output: AI output
            explanation: Explanation of decision
            order_id: Associated order ID if any

        Returns:
            True if upload successful
        """
        client = self._ensure_connected()

        try:
            await client.ai.upload_ai_log(
                stage=stage,
                model=model,
                input_data=input_data,
                output=output,
                explanation=explanation,
                order_id=order_id,
            )
            logger.info("Uploaded AI log to WEEX")
            return True
        except Exception as e:
            logger.error(f"Failed to upload AI log: {e}")
            return False
