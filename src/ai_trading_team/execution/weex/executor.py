"""WEEX exchange executor implementation."""

import logging
import uuid
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation
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
        self._contract_cache: dict[str, dict[str, Any]] = {}

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

    def _json_safe(self, value: Any) -> Any:
        """Convert values to JSON-serializable forms for WEEX AI logs."""
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        if hasattr(value, "value"):
            return value.value
        return value

    def _truncate_explanation(self, explanation: str, limit: int = 1000) -> str:
        if len(explanation) <= limit:
            return explanation
        return explanation[:limit]

    def _normalize_order_id(self, order_id: int | str | None) -> int | None:
        if order_id is None:
            return None
        try:
            return int(order_id)
        except (TypeError, ValueError):
            return None

    def _decimal_or_zero(self, value: Any) -> Decimal:
        if value is None or value == "":
            return Decimal("0")
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0")

    def _unwrap_payload(self, response: Any) -> Any:
        if isinstance(response, dict):
            for key in ("data", "result"):
                if key in response and response[key] is not None:
                    return response[key]
        return response

    def _extract_assets(self, response: Any) -> list[dict[str, Any]]:
        payload = self._unwrap_payload(response)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("assets", "list"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _extract_collateral(self, response: Any) -> list[dict[str, Any]]:
        payload = self._unwrap_payload(response)
        if isinstance(payload, dict):
            collateral = payload.get("collateral")
            if isinstance(collateral, list):
                return collateral
        return []

    def _extract_list(self, response: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        payload = self._unwrap_payload(response)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _asset_coin(self, asset: dict[str, Any]) -> str:
        coin = asset.get("coinName") or asset.get("coin") or ""
        return str(coin).upper()

    def _format_decimal(self, value: Decimal) -> str:
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _quantize_step(self, value: Decimal, step: Decimal, rounding: str) -> Decimal:
        if step <= 0:
            return value
        return (value / step).to_integral_value(rounding=rounding) * step

    def _first_decimal(self, data: dict[str, Any], keys: tuple[str, ...]) -> Decimal:
        for key in keys:
            if key in data:
                value = self._decimal_or_zero(data.get(key))
                if value != 0:
                    return value
        return Decimal("0")

    def _position_size(self, data: dict[str, Any]) -> Decimal:
        size = self._first_decimal(
            data,
            (
                "total",
                "size",
                "holdVolume",
                "holdSize",
                "positionSize",
                "positionQty",
                "qty",
                "volume",
            ),
        )
        return abs(size)

    def _position_entry_price(self, data: dict[str, Any], size: Decimal) -> Decimal:
        entry = self._first_decimal(
            data,
            (
                "averageOpenPrice",
                "avgOpenPrice",
                "avgPrice",
                "entryPrice",
                "openPrice",
                "openAvgPrice",
            ),
        )
        if entry > 0:
            return entry

        open_value = self._first_decimal(data, ("openValue", "open_value", "openValue"))
        if open_value > 0 and size > 0:
            return open_value / size
        return Decimal("0")

    def _position_margin(self, data: dict[str, Any]) -> Decimal:
        return self._first_decimal(
            data,
            (
                "margin",
                "isolatedMargin",
                "marginSize",
                "positionMargin",
                "openMargin",
                "marginAmount",
            ),
        )

    def _position_leverage(self, data: dict[str, Any]) -> int:
        leverage = self._first_decimal(data, ("leverage", "lever", "marginLeverage"))
        return int(leverage) if leverage > 0 else 1

    def _position_side(self, data: dict[str, Any]) -> Side | None:
        side_raw = str(
            data.get("holdSide") or data.get("side") or data.get("positionSide") or ""
        ).lower()
        if side_raw in {"long", "buy"}:
            return Side.LONG
        if side_raw in {"short", "sell"}:
            return Side.SHORT
        return None

    def _position_unrealized(self, data: dict[str, Any]) -> Decimal:
        return self._first_decimal(
            data,
            (
                "unrealisedPL",
                "unrealisedPnl",
                "unrealizedPL",
                "unrealizedPnl",
                "unrealisePnl",
                "unrealizedPnL",
            ),
        )

    def _parse_position(self, symbol: str, data: dict[str, Any]) -> Position | None:
        size = self._position_size(data)
        if size <= 0:
            return None

        side = self._position_side(data) or Side.LONG
        entry_price = self._position_entry_price(data, size)
        leverage = self._position_leverage(data)
        margin = self._position_margin(data)
        if margin <= 0 and entry_price > 0 and leverage > 0:
            margin = (size * entry_price) / Decimal(leverage)

        unrealized = self._position_unrealized(data)
        realized_raw = data.get("realisedPL")
        if realized_raw is None:
            realized_raw = data.get("realizedPL")

        liquidation_price = data.get("liquidationPrice") or data.get("liquidation_price")

        return Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            leverage=leverage,
            unrealized_pnl=unrealized,
            realized_pnl=self._decimal_or_zero(realized_raw),
            liquidation_price=self._decimal_or_zero(liquidation_price)
            if liquidation_price
            else None,
            margin=margin,
            status=PositionStatus.OPEN,
        )

    async def _get_contract(self, symbol: str) -> dict[str, Any]:
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        client = self._ensure_connected()
        response = await client.get("/capi/v2/market/contracts", params={"symbol": symbol})
        contracts = self._extract_list(response, ("contracts", "list"))
        for item in contracts:
            if item.get("symbol") == symbol:
                self._contract_cache[symbol] = item
                return item
        return {}

    def _size_step_from_contract(self, contract: dict[str, Any]) -> Decimal:
        size_increment = contract.get("size_increment") or contract.get("sizeIncrement")
        min_order_size = contract.get("minOrderSize") or contract.get("min_order_size")

        step_from_increment = Decimal("0")
        if size_increment not in (None, ""):
            try:
                decimals = int(size_increment)
                step_from_increment = Decimal("1") / (Decimal("10") ** Decimal(decimals))
            except (ValueError, InvalidOperation):
                step_from_increment = Decimal("0")

        min_size = self._decimal_or_zero(min_order_size)
        if step_from_increment > 0 and min_size > 0:
            return max(step_from_increment, min_size)
        if min_size > 0:
            return min_size
        return step_from_increment

    def _price_step_from_contract(self, contract: dict[str, Any]) -> Decimal:
        tick_size = contract.get("tick_size") or contract.get("tickSize")
        price_end_step = contract.get("priceEndStep") or contract.get("price_end_step")

        if tick_size in (None, ""):
            return Decimal("0")

        try:
            decimals = int(tick_size)
        except (ValueError, InvalidOperation):
            return Decimal("0")

        step = self._decimal_or_zero(price_end_step) or Decimal("1")
        return step / (Decimal("10") ** Decimal(decimals))

    async def get_account(self) -> Account:
        """Get account information."""
        client = self._ensure_connected()
        assets_response = await client.get("/capi/v2/account/assets")
        assets = self._extract_assets(assets_response)

        # Find USDT asset balance (contract assets endpoint)
        total_equity = Decimal("0")
        available_balance = Decimal("0")
        used_margin = Decimal("0")
        unrealized_pnl = Decimal("0")
        found_asset = False

        for asset in assets:
            if self._asset_coin(asset) == "USDT":
                total_equity = self._decimal_or_zero(asset.get("equity"))
                available_balance = self._decimal_or_zero(asset.get("available"))
                used_margin = self._decimal_or_zero(asset.get("frozen"))
                equity_minus_available = total_equity - available_balance
                if equity_minus_available > used_margin:
                    used_margin = equity_minus_available
                if used_margin < 0:
                    used_margin = Decimal("0")
                unrealized_raw = asset.get("unrealizePnl")
                if unrealized_raw is None:
                    unrealized_raw = asset.get("unrealizedPnl")
                unrealized_pnl = self._decimal_or_zero(unrealized_raw)
                found_asset = True
                break

        if not found_asset:
            account = await client.account.get_account("USDT")
            collateral = self._extract_collateral(account)
            if not collateral:
                accounts = await client.account.get_accounts()
                collateral = self._extract_collateral(accounts)
            for item in collateral:
                if self._asset_coin(item) == "USDT":
                    amount = self._decimal_or_zero(item.get("amount"))
                    total_equity = amount
                    available_balance = amount
                    used_margin = Decimal("0")
                    unrealized_pnl = Decimal("0")
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
        response = await client.account.get_single_position(symbol)
        data = self._unwrap_payload(response)
        if isinstance(data, list):
            data = data[0] if data else None

        if isinstance(data, dict):
            position = self._parse_position(symbol, data)
            if position:
                return position

        # Fallback to all positions if single-position endpoint lacks data
        for position in await self.get_positions():
            if position.symbol == symbol:
                return position

        return None

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        client = self._ensure_connected()
        response = await client.get("/capi/v2/account/position/allPosition")
        data = self._extract_list(response, ("positions", "list"))

        positions = []
        for item in data:
            symbol = item.get("symbol", "")
            position = self._parse_position(symbol, item)
            if position:
                positions.append(position)

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
        contract = await self._get_contract(symbol)
        size_step = self._size_step_from_contract(contract)
        price_step = self._price_step_from_contract(contract)

        size_decimal = self._decimal_or_zero(size)
        if size_step > 0:
            min_size = self._decimal_or_zero(contract.get("minOrderSize"))
            if min_size > 0 and size_decimal < min_size:
                raise ValueError(
                    f"Order size {size_decimal} below minimum {min_size} (step={size_step})"
                )
        adjusted_size = self._quantize_step(size_decimal, size_step, ROUND_DOWN)
        if adjusted_size <= 0:
            raise ValueError(
                f"Order size {size_decimal} invalid after step adjustment (step={size_step})"
            )

        adjusted_price: Decimal | None = None
        if price is not None:
            adjusted_price = self._quantize_step(
                self._decimal_or_zero(price), price_step, ROUND_DOWN
            )

        adjusted_stop_loss: Decimal | None = None
        if stop_loss_price is not None:
            rounding = ROUND_DOWN if side == Side.LONG else ROUND_UP
            adjusted_stop_loss = self._quantize_step(
                self._decimal_or_zero(stop_loss_price), price_step, rounding
            )

        adjusted_take_profit: Decimal | None = None
        if take_profit_price is not None:
            adjusted_take_profit = self._quantize_step(
                self._decimal_or_zero(take_profit_price), price_step, ROUND_DOWN
            )

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
            "size": self._format_decimal(adjusted_size),
            "order_type": weex_order_type,
            "match_price": match_price,
            "type": type_code,
            "margin_mode": 3,
        }

        if order_type == OrderType.LIMIT and adjusted_price is not None:
            order_params["price"] = self._format_decimal(adjusted_price)
        elif order_type == OrderType.MARKET:
            order_params["price"] = (
                self._format_decimal(adjusted_price) if adjusted_price is not None else "0"
            )

        # Add preset stop loss if provided (for opening positions)
        if action == "open" and adjusted_stop_loss is not None:
            order_params["preset_stop_loss_price"] = self._format_decimal(adjusted_stop_loss)
            logger.info(f"Setting preset stop loss at {adjusted_stop_loss}")

        # Add preset take profit if provided
        if action == "open" and adjusted_take_profit is not None:
            order_params["preset_take_profit_price"] = self._format_decimal(adjusted_take_profit)

        logger.info(f"Placing order: {order_params}")

        result = await client.trade.place_order(**order_params)

        return Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=adjusted_size,
            price=adjusted_price,
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

    async def cancel_all_plan_orders(self, symbol: str) -> int:
        """Cancel all plan/trigger orders for a symbol."""
        client = self._ensure_connected()
        try:
            response = await client.post(
                "/capi/v2/order/cancelAllOrders",
                data={"cancelOrderType": "plan", "symbol": symbol},
            )
            if isinstance(response, list):
                return sum(1 for item in response if item.get("success"))
            return 0
        except Exception as e:
            logger.error(f"Failed to cancel plan orders: {e}")
            return 0

    async def get_current_plan_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Get current plan orders for a symbol."""
        client = self._ensure_connected()
        response = await client.get("/capi/v2/order/currentPlan", params={"symbol": symbol})
        data = self._extract_list(response, ("list", "orders"))
        if isinstance(response, list):
            return response
        return data

    async def cancel_plan_order(self, order_id: str) -> bool:
        """Cancel a specific plan order by order ID."""
        client = self._ensure_connected()
        try:
            await client.post("/capi/v2/order/cancel_plan", data={"orderId": str(order_id)})
            return True
        except Exception as e:
            logger.error(f"Failed to cancel plan order {order_id}: {e}")
            return False

    async def cancel_stop_loss_plans(self, symbol: str, side: Side | None = None) -> int:
        """Cancel existing stop-loss plan orders for a symbol/side."""
        position_side = None
        if side == Side.LONG:
            position_side = "long"
        elif side == Side.SHORT:
            position_side = "short"

        plans = await self.get_current_plan_orders(symbol)
        cancelled = 0
        for plan in plans:
            if plan.get("symbol") and plan.get("symbol") != symbol:
                continue
            plan_type = str(plan.get("planType") or plan.get("plan_type") or "").lower()
            if plan_type != "loss_plan":
                continue
            if position_side:
                plan_side = str(plan.get("positionSide") or plan.get("position_side") or "").lower()
                if plan_side and plan_side != position_side:
                    continue
            order_id = plan.get("orderId") or plan.get("order_id")
            if not order_id:
                continue
            if await self.cancel_plan_order(str(order_id)):
                cancelled += 1
        return cancelled

    async def get_stop_loss_plans(self, symbol: str, side: Side | None = None) -> list[str]:
        """List active stop-loss plan order IDs for a symbol/side."""
        position_side = None
        if side == Side.LONG:
            position_side = "long"
        elif side == Side.SHORT:
            position_side = "short"

        plans = await self.get_current_plan_orders(symbol)
        order_ids: list[str] = []
        for plan in plans:
            if plan.get("symbol") and plan.get("symbol") != symbol:
                continue
            plan_type = str(plan.get("planType") or plan.get("plan_type") or "").lower()
            if plan_type != "loss_plan":
                continue
            if position_side:
                plan_side = str(plan.get("positionSide") or plan.get("position_side") or "").lower()
                if plan_side and plan_side != position_side:
                    continue
            order_id = plan.get("orderId") or plan.get("order_id")
            if order_id:
                order_ids.append(str(order_id))
        return order_ids

    async def place_stop_loss_plan(
        self,
        symbol: str,
        side: Side,
        size: float,
        trigger_price: float,
    ) -> str | None:
        """Place a stop-loss plan order for an existing position."""
        client = self._ensure_connected()
        contract = await self._get_contract(symbol)
        size_step = self._size_step_from_contract(contract)
        price_step = self._price_step_from_contract(contract)

        size_decimal = self._decimal_or_zero(size)
        adjusted_size = self._quantize_step(size_decimal, size_step, ROUND_DOWN)
        if adjusted_size <= 0:
            raise ValueError(
                f"Stop loss size {size_decimal} invalid after step adjustment (step={size_step})"
            )

        rounding = ROUND_DOWN if side == Side.LONG else ROUND_UP
        adjusted_trigger = self._quantize_step(
            self._decimal_or_zero(trigger_price), price_step, rounding
        )
        if adjusted_trigger <= 0:
            raise ValueError("Stop loss trigger price must be > 0")

        client_order_id = f"sl_{uuid.uuid4().hex[:16]}"
        position_side = "long" if side == Side.LONG else "short"

        data = {
            "symbol": symbol,
            "clientOrderId": client_order_id,
            "planType": "loss_plan",
            "triggerPrice": self._format_decimal(adjusted_trigger),
            "executePrice": "0",
            "size": self._format_decimal(adjusted_size),
            "positionSide": position_side,
            "marginMode": 3,
        }

        response = await client.post("/capi/v2/order/placeTpSlOrder", data=data)
        if isinstance(response, list) and response:
            item = response[0]
            if item.get("success"):
                return str(item.get("orderId") or "")
        return None

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Get all open orders for a symbol."""
        client = self._ensure_connected()

        response = await client.get("/capi/v2/order/current", params={"symbol": symbol})
        data = self._extract_list(response, ("orders", "list"))
        orders = []

        for item in data:
            side_str = item.get("side", "1")
            side = Side.LONG if side_str in ("1", "3") else Side.SHORT

            orders.append(
                Order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    size=self._decimal_or_zero(item.get("size")),
                    price=self._decimal_or_zero(item.get("price")),
                    status=OrderStatus.NEW,
                    order_id=str(item.get("orderId", "")),
                    filled_size=self._decimal_or_zero(item.get("filledSize")),
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
            await client.account.change_hold_model(
                symbol=symbol,
                margin_mode=3,  # Isolated margin
                separated_mode=1,
            )
            await client.account.set_leverage(
                symbol=symbol,
                margin_mode=3,  # Isolated margin
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
                input_data=self._json_safe(input_data),
                output=self._json_safe(output),
                explanation=self._truncate_explanation(explanation),
                order_id=self._normalize_order_id(order_id),
            )
            logger.info("Uploaded AI log to WEEX")
            return True
        except Exception as e:
            logger.error(f"Failed to upload AI log: {e}")
            return False
