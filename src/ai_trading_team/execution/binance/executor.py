"""Binance USDT Futures executor implementation."""

import asyncio
import logging
import uuid
from decimal import Decimal
from typing import Any

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

from ai_trading_team.core.types import OrderStatus, OrderType, PositionStatus, Side, TimeInForce
from ai_trading_team.execution.models import Account, Order, Position

logger = logging.getLogger(__name__)


class BinanceExecutor:
    """Binance USDT-margined futures exchange implementation."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        """Initialize Binance executor.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._client: DerivativesTradingUsdsFutures | None = None
        self._connected = False

    @property
    def name(self) -> str:
        return "BINANCE"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establish connection to Binance."""
        if not self._api_key or not self._api_secret:
            raise ValueError("Binance API key/secret required for trading")

        config = ConfigurationRestAPI(
            api_key=self._api_key,
            api_secret=self._api_secret,
            base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
        )
        self._client = DerivativesTradingUsdsFutures(config_rest_api=config)
        self._connected = True
        logger.info("Connected to Binance USDT Futures")

    async def disconnect(self) -> None:
        """Close connection to Binance."""
        self._client = None
        self._connected = False
        logger.info("Disconnected from Binance USDT Futures")

    def _ensure_client(self) -> DerivativesTradingUsdsFutures:
        if not self._client or not self._connected:
            raise RuntimeError("Not connected to Binance")
        return self._client

    def _decimal_or_zero(self, value: Any) -> Decimal:
        if value is None or value == "":
            return Decimal("0")
        return Decimal(str(value))

    def _decimal_or_none(self, value: Any) -> Decimal | None:
        if value is None or value == "":
            return None
        return Decimal(str(value))

    def _normalize_data(self, data: Any) -> Any:
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        elif hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "dict"):
            data = data.dict()

        if isinstance(data, list):
            return [self._normalize_data(item) for item in data]
        if isinstance(data, dict):
            return {key: self._normalize_data(value) for key, value in data.items()}
        return data

    def _map_order_status(self, status: str | None) -> OrderStatus:
        mapping = {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "CANCELLED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        if not status:
            return OrderStatus.NEW
        return mapping.get(status.upper(), OrderStatus.NEW)

    def _map_order_type(self, order_type: str | None) -> OrderType:
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP_LIMIT,
            "STOP_MARKET": OrderType.STOP_MARKET,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT_LIMIT,
            "TAKE_PROFIT_MARKET": OrderType.TAKE_PROFIT_MARKET,
        }
        if not order_type:
            return OrderType.MARKET
        return mapping.get(order_type.upper(), OrderType.MARKET)

    def _order_side(self, side: Side, action: str) -> str:
        if action == "close":
            return "SELL" if side == Side.LONG else "BUY"
        return "BUY" if side == Side.LONG else "SELL"

    def _binance_order_type(self, order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LIMIT: "STOP",
            OrderType.STOP_MARKET: "STOP_MARKET",
            OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT",
            OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
        }
        return mapping.get(order_type, "MARKET")

    def _position_side(self, item: dict[str, Any], position_amt: Decimal) -> Side:
        position_side = str(item.get("positionSide", "")).upper()
        if position_side == "SHORT":
            return Side.SHORT
        if position_side == "LONG":
            return Side.LONG
        return Side.SHORT if position_amt < 0 else Side.LONG

    def _parse_standard_order(self, data: dict[str, Any], fallback_symbol: str) -> Order:
        symbol = data.get("symbol", fallback_symbol)
        side_raw = data.get("side", "")
        side = Side.LONG if side_raw.upper() == "BUY" else Side.SHORT
        raw_time_in_force = data.get("timeInForce")
        time_in_force = TimeInForce.GTC
        if raw_time_in_force:
            try:
                time_in_force = TimeInForce(raw_time_in_force)
            except ValueError:
                time_in_force = TimeInForce.GTC

        return Order(
            symbol=symbol,
            side=side,
            order_type=self._map_order_type(data.get("type") or data.get("origType")),
            size=self._decimal_or_zero(data.get("origQty")),
            price=self._decimal_or_none(data.get("price")),
            status=self._map_order_status(data.get("status")),
            filled_size=self._decimal_or_zero(data.get("executedQty")),
            avg_fill_price=self._decimal_or_none(data.get("avgPrice")),
            time_in_force=time_in_force,
            stop_price=self._decimal_or_none(data.get("stopPrice")),
            client_order_id=str(data.get("clientOrderId", "")),
            order_id=str(data.get("orderId", "")),
        )

    def _parse_algo_order(self, data: dict[str, Any], fallback_symbol: str) -> Order:
        symbol = data.get("symbol", fallback_symbol)
        side_raw = data.get("side", "")
        side = Side.LONG if side_raw.upper() == "BUY" else Side.SHORT
        algo_id = data.get("algoId")
        order_id = f"algo:{algo_id}" if algo_id is not None else "algo:unknown"

        return Order(
            symbol=symbol,
            side=side,
            order_type=self._map_order_type(data.get("orderType")),
            size=self._decimal_or_zero(data.get("quantity")),
            price=self._decimal_or_none(data.get("price")),
            status=OrderStatus.NEW,
            stop_price=self._decimal_or_none(data.get("triggerPrice")),
            client_order_id=str(data.get("clientAlgoId", "")),
            order_id=order_id,
        )

    async def get_account(self) -> Account:
        """Get account information."""
        client = self._ensure_client()

        def _fetch() -> Any:
            response = client.rest_api.account_information_v2()
            return response.data()

        data = await asyncio.to_thread(_fetch)
        data = self._normalize_data(data) or {}

        total_equity = self._decimal_or_zero(
            data.get("totalMarginBalance") or data.get("totalWalletBalance")
        )
        available_balance = self._decimal_or_zero(data.get("availableBalance"))
        used_margin = self._decimal_or_zero(
            data.get("totalPositionInitialMargin")
        ) + self._decimal_or_zero(data.get("totalOpenOrderInitialMargin"))
        unrealized_pnl = self._decimal_or_zero(data.get("totalUnrealizedProfit"))

        return Account(
            total_equity=total_equity,
            available_balance=available_balance,
            used_margin=used_margin,
            unrealized_pnl=unrealized_pnl,
        )

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        client = self._ensure_client()

        def _fetch() -> Any:
            response = client.rest_api.position_information_v2(symbol=symbol)
            return response.data()

        data = await asyncio.to_thread(_fetch)
        positions = self._normalize_data(data) or []

        if not isinstance(positions, list) or not positions:
            return None

        for item in positions:
            position_amt = self._decimal_or_zero(item.get("positionAmt"))
            if position_amt == 0:
                continue

            side = self._position_side(item, position_amt)
            size = abs(position_amt)
            leverage = int(self._decimal_or_zero(item.get("leverage"))) or 1
            margin = self._decimal_or_zero(item.get("isolatedMargin"))
            if margin == 0:
                notional = self._decimal_or_zero(item.get("notional"))
                if leverage:
                    margin = abs(notional) / Decimal(str(leverage))

            return Position(
                symbol=item.get("symbol", symbol),
                side=side,
                size=size,
                entry_price=self._decimal_or_zero(item.get("entryPrice")),
                leverage=leverage,
                unrealized_pnl=self._decimal_or_zero(item.get("unRealizedProfit")),
                liquidation_price=self._decimal_or_none(item.get("liquidationPrice")),
                margin=margin,
                status=PositionStatus.OPEN,
            )

        return None

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        client = self._ensure_client()

        def _fetch() -> Any:
            response = client.rest_api.position_information_v2()
            return response.data()

        data = await asyncio.to_thread(_fetch)
        positions_data = self._normalize_data(data) or []
        positions: list[Position] = []

        if not isinstance(positions_data, list):
            return positions

        for item in positions_data:
            position_amt = self._decimal_or_zero(item.get("positionAmt"))
            if position_amt == 0:
                continue

            side = self._position_side(item, position_amt)
            size = abs(position_amt)
            leverage = int(self._decimal_or_zero(item.get("leverage"))) or 1
            margin = self._decimal_or_zero(item.get("isolatedMargin"))
            if margin == 0:
                notional = self._decimal_or_zero(item.get("notional"))
                if leverage:
                    margin = abs(notional) / Decimal(str(leverage))

            positions.append(
                Position(
                    symbol=item.get("symbol", ""),
                    side=side,
                    size=size,
                    entry_price=self._decimal_or_zero(item.get("entryPrice")),
                    leverage=leverage,
                    unrealized_pnl=self._decimal_or_zero(item.get("unRealizedProfit")),
                    liquidation_price=self._decimal_or_none(item.get("liquidationPrice")),
                    margin=margin,
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
        time_in_force: TimeInForce = TimeInForce.GTC,
        action: str = "open",
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> Order:
        """Place a new order."""
        client = self._ensure_client()

        client_order_id = f"ai_{uuid.uuid4().hex[:16]}"
        order_side = self._order_side(side, action)
        binance_order_type = self._binance_order_type(order_type)
        reduce_only = "true" if action == "close" else "false"

        def _submit() -> Any:
            response = client.rest_api.new_order(
                symbol=symbol,
                side=order_side,
                type=binance_order_type,
                time_in_force=time_in_force.value if order_type == OrderType.LIMIT else None,
                quantity=float(size),
                reduce_only=reduce_only,
                price=float(price) if price is not None else None,
                new_client_order_id=client_order_id,
                new_order_resp_type="RESULT",
            )
            return response.data()

        data = await asyncio.to_thread(_submit)
        data = self._normalize_data(data) or {}

        order = self._parse_standard_order(
            data if isinstance(data, dict) else {},
            fallback_symbol=symbol,
        )
        order.side = side
        order.order_type = order_type
        order.client_order_id = client_order_id or order.client_order_id
        if order.size == 0:
            order.size = self._decimal_or_zero(size)
        if order.price is None and price is not None:
            order.price = self._decimal_or_zero(price)

        if action == "open" and stop_loss_price is not None:
            await self._place_conditional_order(
                symbol=symbol,
                side=side,
                trigger_price=stop_loss_price,
                order_type="STOP_MARKET",
                client_prefix="sl",
            )

        if action == "open" and take_profit_price is not None:
            await self._place_conditional_order(
                symbol=symbol,
                side=side,
                trigger_price=take_profit_price,
                order_type="TAKE_PROFIT_MARKET",
                client_prefix="tp",
            )

        return order

    async def _place_conditional_order(
        self,
        symbol: str,
        side: Side,
        trigger_price: float,
        order_type: str,
        client_prefix: str,
    ) -> None:
        """Place a conditional stop/take-profit order to close a position."""
        client = self._ensure_client()
        close_side = self._order_side(side, action="close")
        client_algo_id = f"{client_prefix}_{uuid.uuid4().hex[:16]}"

        def _submit() -> Any:
            response = client.rest_api.new_algo_order(
                algo_type="CONDITIONAL",
                symbol=symbol,
                side=close_side,
                type=order_type,
                trigger_price=float(trigger_price),
                close_position="true",
                working_type="MARK_PRICE",
                client_algo_id=client_algo_id,
            )
            return response.data()

        try:
            data = await asyncio.to_thread(_submit)
            data = self._normalize_data(data) or {}
            logger.info(
                "Placed conditional order: %s %s at %s (algoId=%s)",
                order_type,
                symbol,
                trigger_price,
                data.get("algoId"),
            )
        except Exception as e:
            logger.warning("Failed to place conditional order: %s", e)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        client = self._ensure_client()

        def _cancel_standard() -> Any:
            response = client.rest_api.cancel_order(symbol=symbol, order_id=int(order_id))
            return response.data()

        def _cancel_algo(algo_id: int) -> Any:
            response = client.rest_api.cancel_algo_order(algoid=algo_id)
            return response.data()

        try:
            if order_id.startswith("algo:"):
                algo_id = int(order_id.split(":", 1)[1])
                await asyncio.to_thread(_cancel_algo, algo_id)
                logger.info("Cancelled algo order %s", order_id)
            else:
                await asyncio.to_thread(_cancel_standard)
                logger.info("Cancelled order %s", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders for a symbol."""
        client = self._ensure_client()
        open_orders = await self.get_open_orders(symbol)
        count = len(open_orders)

        def _cancel_standard() -> Any:
            response = client.rest_api.cancel_all_open_orders(symbol=symbol)
            return response.data()

        def _cancel_algo() -> Any:
            response = client.rest_api.cancel_all_algo_open_orders(symbol=symbol)
            return response.data()

        try:
            await asyncio.to_thread(_cancel_standard)
            await asyncio.to_thread(_cancel_algo)
            logger.info("Cancelled %s open orders for %s", count, symbol)
            return count
        except Exception as e:
            logger.error("Failed to cancel all orders: %s", e)
            return 0

    async def get_order(self, symbol: str, order_id: str) -> Order | None:
        """Get order by ID."""
        client = self._ensure_client()

        if order_id.startswith("algo:"):
            return None

        def _fetch() -> Any:
            response = client.rest_api.query_order(symbol=symbol, order_id=int(order_id))
            return response.data()

        data = await asyncio.to_thread(_fetch)
        data = self._normalize_data(data)
        if not isinstance(data, dict):
            return None

        return self._parse_standard_order(data, fallback_symbol=symbol)

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Get all open orders for a symbol."""
        client = self._ensure_client()

        def _fetch_standard() -> Any:
            response = client.rest_api.current_all_open_orders(symbol=symbol)
            return response.data()

        def _fetch_algo() -> Any:
            response = client.rest_api.current_all_algo_open_orders(
                algo_type="CONDITIONAL",
                symbol=symbol,
            )
            return response.data()

        try:
            standard_data = await asyncio.to_thread(_fetch_standard)
        except Exception as e:
            logger.error("Failed to fetch open orders: %s", e)
            standard_data = []

        try:
            algo_data = await asyncio.to_thread(_fetch_algo)
        except Exception as e:
            logger.warning("Failed to fetch algo open orders: %s", e)
            algo_data = []

        standard_orders = self._normalize_data(standard_data) or []
        algo_orders = self._normalize_data(algo_data) or []

        orders: list[Order] = []
        if isinstance(standard_orders, list):
            for item in standard_orders:
                if isinstance(item, dict):
                    orders.append(self._parse_standard_order(item, fallback_symbol=symbol))

        if isinstance(algo_orders, list):
            for item in algo_orders:
                if isinstance(item, dict):
                    orders.append(self._parse_algo_order(item, fallback_symbol=symbol))

        return orders

    async def close_position(
        self,
        symbol: str,
        side: Side,
        size: float | None = None,
    ) -> Order | None:
        """Close a position (fully or partially)."""
        position = await self.get_position(symbol)
        if not position or position.size == 0:
            logger.warning("No position to close for %s", symbol)
            return None

        close_size = Decimal(str(size)) if size else position.size
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=float(close_size),
            action="close",
        )

    async def reduce_position(self, symbol: str, side: Side, size: float) -> Order | None:
        """Partially close a position."""
        if size <= 0:
            logger.error("Invalid reduce size: %s", size)
            return None

        return await self.close_position(symbol, side, size)

    async def add_to_position(
        self,
        symbol: str,
        side: Side,
        size: float,
        stop_loss_price: float | None = None,
    ) -> Order | None:
        """Add to an existing position."""
        if size <= 0:
            logger.error("Invalid add size: %s", size)
            return None

        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=size,
            action="open",
            stop_loss_price=stop_loss_price,
        )

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        client = self._ensure_client()

        def _submit() -> Any:
            response = client.rest_api.change_initial_leverage(symbol=symbol, leverage=leverage)
            return response.data()

        try:
            await asyncio.to_thread(_submit)
            logger.info("Set leverage to %sx for %s", leverage, symbol)
            return True
        except Exception as e:
            logger.error("Failed to set leverage: %s", e)
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
        """Log AI decision (Binance has no AI log endpoint)."""
        _ = (stage, model, input_data, output, explanation, order_id)
        logger.debug("AI log skipped for Binance")
        return False
