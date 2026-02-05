"""Trade API module for Weex SDK."""

from typing import Any, Dict, List, Optional

from weex_sdk.client import AsyncWeexClient, WeexClient
from weex_sdk.models import Order


class TradeAPI:
    """Trade API methods."""

    def __init__(self, client: WeexClient) -> None:
        """Initialize Trade API.

        Args:
            client: WeexClient instance
        """
        self.client = client

    def place_order(
        self,
        symbol: str,
        client_oid: str,
        size: str,
        order_type: str,
        match_price: str,
        price: str,
        type: str,
        preset_take_profit_price: Optional[str] = None,
        preset_stop_loss_price: Optional[str] = None,
        margin_mode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place an order.

        Args:
            symbol: Trading pair
            client_oid: Custom order ID (no more than 40 characters)
            size: Order quantity (cannot be zero or negative)
            order_type: Order type (0: Normal, 1: Post-Only, 2: Fill-Or-Kill,
                3: Immediate Or Cancel)
            match_price: Price type (0: Limit price, 1: Market price)
            price: Order price (required for limit orders)
            type: Order direction (1: Open long, 2: Open short, 3: Close long, 4: Close short)
            preset_take_profit_price: Preset take-profit price (optional)
            preset_stop_loss_price: Preset stop-loss price (optional)
            margin_mode: Margin mode (1: Cross Mode, 3: Isolated Mode, default: 1)

        Returns:
            Order placement response with order_id

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": str(size),
            "type": str(type),
            "order_type": str(order_type),
            "match_price": str(match_price),
            "price": str(price),
        }

        if preset_take_profit_price:
            data["presetTakeProfitPrice"] = str(preset_take_profit_price)
        if preset_stop_loss_price:
            data["presetStopLossPrice"] = str(preset_stop_loss_price)
        if margin_mode:
            data["marginMode"] = margin_mode

        return self.client.post("/capi/v2/order/placeOrder", data=data)

    def batch_orders(
        self,
        symbol: str,
        order_data_list: List[Dict[str, Any]],
        margin_mode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place batch orders (max 20 orders).

        Args:
            symbol: Trading pair
            order_data_list: List of order data (same structure as place_order)
            margin_mode: Margin mode (1: Cross Mode, 3: Isolated Mode, default: 1)

        Returns:
            Batch order placement response

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {
            "symbol": symbol,
            "orderDataList": order_data_list,
        }
        if margin_mode:
            data["marginMode"] = margin_mode

        return self.client.post("/capi/v2/order/batchOrders", data=data)

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: Order ID (either order_id or client_oid required)
            client_oid: Client order ID (either order_id or client_oid required)

        Returns:
            Cancellation response

        Raises:
            WeexAPIError: On API errors
            ValueError: If neither order_id nor client_oid provided
        """
        if not order_id and not client_oid:
            raise ValueError("Either order_id or client_oid must be provided")

        data: Dict[str, Any] = {}
        if order_id:
            data["orderId"] = str(order_id)
        if client_oid:
            data["clientOid"] = str(client_oid)

        return self.client.post("/capi/v2/order/cancel_order", data=data)

    def cancel_batch_orders(
        self,
        ids: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Cancel batch orders.

        Args:
            ids: List of order IDs (either ids or cids required)
            cids: List of client order IDs (either ids or cids required)

        Returns:
            Batch cancellation response

        Raises:
            WeexAPIError: On API errors
            ValueError: If neither ids nor cids provided
        """
        if not ids and not cids:
            raise ValueError("Either ids or cids must be provided")

        data: Dict[str, Any] = {}
        if ids:
            data["ids"] = [str(id) for id in ids]
        if cids:
            data["cids"] = [str(cid) for cid in cids]

        return self.client.post("/capi/v2/order/cancel_batch_orders", data=data)

    def get_order_detail(self, order_id: str) -> Order:
        """Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order details

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/order/detail", params={"orderId": order_id})

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        page_size: Optional[int] = None,
        create_date: Optional[int] = None,
    ) -> List[Order]:
        """Get order history.

        Args:
            symbol: Trading pair (optional)
            page_size: Items per page (optional)
            create_date: Creation time (Unix milliseconds, must be ≤ 90 and cannot be negative)

        Returns:
            List of historical orders

        Raises:
            WeexAPIError: On API errors
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if page_size:
            params["pageSize"] = page_size
        if create_date:
            params["createDate"] = create_date

        response = self.client.get("/capi/v2/order/history", params=params)
        if isinstance(response, list):
            return response
        return []

    def get_current_orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
    ) -> List[Order]:
        """Get current open orders.

        Args:
            symbol: Trading pair (optional)
            order_id: Order ID (optional)
            start_time: Query record start time (Unix milliseconds, optional)
            end_time: Query record end time (Unix milliseconds, optional)
            limit: Limit number (default: 100, max: 100)
            page: Page number (default: 0)

        Returns:
            List of current orders

        Raises:
            WeexAPIError: On API errors
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit
        if page is not None:
            params["page"] = page

        response = self.client.get("/capi/v2/order/current", params=params)
        if isinstance(response, list):
            return response
        return []

    def get_order_fills(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get order fill details.

        Args:
            symbol: Trading pair name (optional)
            order_id: Order ID (optional)
            start_time: Start timestamp (Unix milliseconds, optional)
            end_time: End timestamp (Unix milliseconds, optional)
            limit: Number of queries (max: 100, default: 100)

        Returns:
            Order fill details with list and pagination info

        Raises:
            WeexAPIError: On API errors
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit

        return self.client.get("/capi/v2/order/fills", params=params)

    def place_plan_order(
        self,
        symbol: str,
        client_oid: str,
        size: str,
        type: str,
        match_type: str,
        execute_price: str,
        trigger_price: str,
        margin_mode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place a plan/trigger order.

        Args:
            symbol: Trading pair
            client_oid: Custom order ID (≤40 chars, no special characters)
            size: Order quantity in lots
            type: Order direction (1: Open long, 2: Open short, 3: Close long, 4: Close short)
            match_type: Price type (0: Limit price, 1: Market price)
            execute_price: Execution price
            trigger_price: Trigger price
            margin_mode: Margin mode (1: Cross Mode, 3: Isolated Mode, default: 1)

        Returns:
            Plan order placement response

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": str(size),
            "type": str(type),
            "match_type": str(match_type),
            "execute_price": str(execute_price),
            "trigger_price": str(trigger_price),
        }
        if margin_mode:
            data["marginMode"] = margin_mode

        return self.client.post("/capi/v2/order/plan_order", data=data)

    def cancel_plan(self, order_id: str) -> Dict[str, Any]:
        """Cancel a plan order.

        Args:
            order_id: Plan order ID

        Returns:
            Cancellation response

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.post("/capi/v2/order/cancel_plan", data={"orderId": str(order_id)})

    def get_current_plan(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get current plan orders.

        Args:
            symbol: Trading pair (optional)
            order_id: Order ID (optional)
            start_time: Query record start time (Unix milliseconds, optional)
            end_time: Query record end time (Unix milliseconds, optional)
            limit: Limit number (default: 100, max: 100)
            page: Page number (default: 0)

        Returns:
            List of current plan orders

        Raises:
            WeexAPIError: On API errors
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit
        if page is not None:
            params["page"] = page

        response = self.client.get("/capi/v2/order/currentPlan", params=params)
        if isinstance(response, list):
            return response
        return []

    def get_history_plan(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        delegate_type: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get history plan orders.

        Args:
            symbol: Trading pair
            start_time: Start time (Unix milliseconds, optional)
            end_time: End time (Unix milliseconds, optional)
            delegate_type: Order type (1: Open long, 2: Open short, 3: Close long, 4: Close short)
            page_size: Items per page (1-100, default: 100)

        Returns:
            History plan orders with pagination info

        Raises:
            WeexAPIError: On API errors
        """
        params: Dict[str, Any] = {"symbol": symbol}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if delegate_type:
            params["delegateType"] = delegate_type
        if page_size:
            params["pageSize"] = page_size

        return self.client.get("/capi/v2/order/historyPlan", params=params)

    def close_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Close all positions (one-click close).

        Args:
            symbol: Trading pair (optional, closes all if not provided)

        Returns:
            List of close position results

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {}
        if symbol:
            data["symbol"] = symbol

        response = self.client.post("/capi/v2/order/closePositions", data=data)
        if isinstance(response, list):
            return response
        return []

    def cancel_all_orders(
        self,
        cancel_order_type: str,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cancel all orders.

        Args:
            cancel_order_type: Order type to cancel ('normal' or 'plan')
            symbol: Trading pair (optional, cancels all if not provided)

        Returns:
            List of cancellation results

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {"cancelOrderType": cancel_order_type}
        if symbol:
            data["symbol"] = symbol

        response = self.client.post("/capi/v2/order/cancelAllOrders", data=data)
        if isinstance(response, list):
            return response
        return []

    def place_tp_sl_order(
        self,
        symbol: str,
        client_order_id: str,
        plan_type: str,
        trigger_price: str,
        size: str,
        position_side: str,
        execute_price: Optional[str] = None,
        margin_mode: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Place take-profit/stop-loss order.

        Args:
            symbol: Trading pair
            client_order_id: Custom order ID (no more than 40 characters)
            plan_type: TP/SL type ('profit_plan' or 'loss_plan')
            trigger_price: Trigger price
            size: Order quantity
            position_side: Position direction ('long' or 'short')
            execute_price: Execution price (optional, market price if 0 or not provided)
            margin_mode: Margin mode (1: Cross Mode, 3: Isolated Mode, default: 1)

        Returns:
            TP/SL order placement response

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {
            "symbol": symbol,
            "clientOrderId": client_order_id,
            "planType": plan_type,
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "positionSide": position_side,
        }
        if execute_price:
            data["executePrice"] = str(execute_price)
        else:
            data["executePrice"] = "0"
        if margin_mode:
            data["marginMode"] = margin_mode

        response = self.client.post("/capi/v2/order/placeTpSlOrder", data=data)
        if isinstance(response, list):
            return response
        return []

    def modify_tp_sl_order(
        self,
        order_id: int,
        trigger_price: str,
        execute_price: Optional[str] = None,
        trigger_price_type: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Modify take-profit/stop-loss order.

        Args:
            order_id: Order ID of the TP/SL order to modify
            trigger_price: New trigger price
            execute_price: New execution price (optional, market price if 0 or not provided)
            trigger_price_type: Trigger price type (1: Last price, 3: Mark price, default: 1)

        Returns:
            Modification response

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {
            "orderId": order_id,
            "triggerPrice": str(trigger_price),
        }
        if execute_price:
            data["executePrice"] = str(execute_price)
        else:
            data["executePrice"] = "0"
        if trigger_price_type:
            data["triggerPriceType"] = trigger_price_type

        return self.client.post("/capi/v2/order/modifyTpSlOrder", data=data)


class AsyncTradeAPI:
    """Async Trade API methods."""

    def __init__(self, client: AsyncWeexClient) -> None:
        """Initialize Async Trade API.

        Args:
            client: AsyncWeexClient instance
        """
        self.client = client

    async def place_order(
        self,
        symbol: str,
        client_oid: str,
        size: str,
        order_type: str,
        match_price: str,
        price: str,
        type: str,
        preset_take_profit_price: Optional[str] = None,
        preset_stop_loss_price: Optional[str] = None,
        margin_mode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place an order (async)."""
        data: Dict[str, Any] = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": str(size),
            "type": str(type),
            "order_type": str(order_type),
            "match_price": str(match_price),
            "price": str(price),
        }
        if preset_take_profit_price:
            data["presetTakeProfitPrice"] = str(preset_take_profit_price)
        if preset_stop_loss_price:
            data["presetStopLossPrice"] = str(preset_stop_loss_price)
        if margin_mode:
            data["marginMode"] = margin_mode
        return await self.client.post("/capi/v2/order/placeOrder", data=data)

    async def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an order (async)."""
        if not order_id and not client_oid:
            raise ValueError("Either order_id or client_oid must be provided")
        data: Dict[str, Any] = {}
        if order_id:
            data["orderId"] = str(order_id)
        if client_oid:
            data["clientOid"] = str(client_oid)
        return await self.client.post("/capi/v2/order/cancel_order", data=data)

    async def get_order_detail(self, order_id: str) -> Order:
        """Get order details (async)."""
        return await self.client.get("/capi/v2/order/detail", params={"orderId": order_id})

    async def get_current_orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
    ) -> List[Order]:
        """Get current open orders (async)."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit
        if page is not None:
            params["page"] = page
        response = await self.client.get("/capi/v2/order/current", params=params)
        if isinstance(response, list):
            return response
        return []

    async def get_order_fills(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get order fill details (async)."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit
        return await self.client.get("/capi/v2/order/fills", params=params)

    async def close_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Close all positions (async)."""
        data: Dict[str, Any] = {}
        if symbol:
            data["symbol"] = symbol
        response = await self.client.post("/capi/v2/order/closePositions", data=data)
        if isinstance(response, list):
            return response
        return []

    async def cancel_all_orders(
        self,
        cancel_order_type: str,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cancel all orders (async)."""
        data: Dict[str, Any] = {"cancelOrderType": cancel_order_type}
        if symbol:
            data["symbol"] = symbol
        response = await self.client.post("/capi/v2/order/cancelAllOrders", data=data)
        if isinstance(response, list):
            return response
        return []
