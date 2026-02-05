"""Account API module for Weex SDK."""

from typing import Any, Dict, List, Optional

from weex_sdk.client import AsyncWeexClient, WeexClient
from weex_sdk.models import Asset, Position


class AccountAPI:
    """Account API methods."""

    def __init__(self, client: WeexClient) -> None:
        """Initialize Account API.

        Args:
            client: WeexClient instance
        """
        self.client = client

    def get_accounts(self) -> Dict[str, Any]:
        """Get all account information.

        Returns:
            Account information including account settings and collateral

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/account/getAccounts")

    def get_account(self, coin: str) -> Dict[str, Any]:
        """Get account information for a specific coin.

        Args:
            coin: Coin name (e.g., 'USDT')

        Returns:
            Account information for the specified coin

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/account/getAccount", params={"coin": coin})

    def get_assets(self) -> List[Asset]:
        """Get account assets.

        Returns:
            List of asset information

        Raises:
            WeexAPIError: On API errors
        """
        response = self.client.get("/capi/v2/account/assets")
        if isinstance(response, list):
            return response
        return []

    def get_bills(
        self,
        coin: Optional[str] = None,
        symbol: Optional[str] = None,
        business_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get contract account bill history.

        Args:
            coin: Currency name
            symbol: Trading pair
            business_type: Business type (deposit, withdraw, transfer_in, etc.)
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)
            limit: Return record limit (default: 20, min: 1, max: 100)

        Returns:
            Bill history with items and pagination info

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {}
        if coin:
            data["coin"] = coin
        if symbol:
            data["symbol"] = symbol
        if business_type:
            data["businessType"] = business_type
        if start_time:
            data["startTime"] = start_time
        if end_time:
            data["endTime"] = end_time
        if limit:
            data["limit"] = limit

        return self.client.post("/capi/v2/account/bills", data=data)

    def get_settings(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get account settings.

        Args:
            symbol: Trading pair (optional, returns all if not specified)

        Returns:
            Account settings including leverage settings

        Raises:
            WeexAPIError: On API errors
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.client.get("/capi/v2/account/settings", params=params)

    def set_leverage(
        self,
        symbol: str,
        margin_mode: int,
        long_leverage: str,
        short_leverage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set leverage for a trading pair.

        Args:
            symbol: Trading pair
            margin_mode: Margin mode (1: Cross Mode, 3: Isolated Mode)
            long_leverage: Long position leverage
            short_leverage: Short position leverage (required for isolated mode)

        Returns:
            Response with success status

        Raises:
            WeexAPIError: On API errors
        """
        if margin_mode == 3 and not short_leverage:
            short_leverage = long_leverage

        data = {
            "symbol": symbol,
            "marginMode": margin_mode,
            "longLeverage": str(long_leverage),
            "shortLeverage": str(short_leverage) if short_leverage else str(long_leverage),
        }
        return self.client.post("/capi/v2/account/leverage", data=data)

    def adjust_margin(
        self,
        isolated_position_id: int,
        collateral_amount: str,
        coin_id: int = 2,
    ) -> Dict[str, Any]:
        """Adjust margin for isolated position.

        Args:
            isolated_position_id: Isolated margin position ID
            collateral_amount: Collateral amount (positive to increase, negative to decrease)
            coin_id: Collateral ID (default: 2 for USDT)

        Returns:
            Response with success status

        Raises:
            WeexAPIError: On API errors
        """
        data = {
            "coinId": coin_id,
            "isolatedPositionId": isolated_position_id,
            "collateralAmount": str(collateral_amount),
        }
        return self.client.post("/capi/v2/account/adjustMargin", data=data)

    def modify_auto_append_margin(
        self,
        position_id: int,
        auto_append_margin: bool,
    ) -> Dict[str, Any]:
        """Modify auto-append margin setting.

        Args:
            position_id: Isolated margin position ID
            auto_append_margin: Whether to enable automatic margin call

        Returns:
            Response with success status

        Raises:
            WeexAPIError: On API errors
        """
        data = {
            "positionId": position_id,
            "autoAppendMargin": auto_append_margin,
        }
        return self.client.post("/capi/v2/account/modifyAutoAppendMargin", data=data)

    def get_all_positions(self) -> List[Position]:
        """Get all positions.

        Returns:
            List of position information

        Raises:
            WeexAPIError: On API errors
        """
        response = self.client.get("/capi/v2/account/position/allPosition")
        if isinstance(response, list):
            return response
        return []

    def get_single_position(self, symbol: str) -> Dict[str, Any]:
        """Get single position for a trading pair.

        Args:
            symbol: Trading pair

        Returns:
            Position information

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get(
            "/capi/v2/account/position/singlePosition", params={"symbol": symbol}
        )

    def change_hold_model(
        self,
        symbol: str,
        margin_mode: int,
        separated_mode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Change position holding model.

        Args:
            symbol: Trading pair
            margin_mode: Margin mode (1: Cross Mode, 3: Isolated Mode)
            separated_mode: Position segregation mode (1: Combined mode, default: 1)

        Returns:
            Response with success status

        Raises:
            WeexAPIError: On API errors
        """
        data = {
            "symbol": symbol,
            "marginMode": margin_mode,
        }
        if separated_mode:
            data["separatedMode"] = separated_mode
        return self.client.post("/capi/v2/account/position/changeHoldModel", data=data)


class AsyncAccountAPI:
    """Async Account API methods."""

    def __init__(self, client: AsyncWeexClient) -> None:
        """Initialize Async Account API.

        Args:
            client: AsyncWeexClient instance
        """
        self.client = client

    async def get_accounts(self) -> Dict[str, Any]:
        """Get all account information (async)."""
        return await self.client.get("/capi/v2/account/getAccounts")

    async def get_account(self, coin: str) -> Dict[str, Any]:
        """Get account information for a specific coin (async)."""
        return await self.client.get("/capi/v2/account/getAccount", params={"coin": coin})

    async def get_assets(self) -> List[Asset]:
        """Get account assets (async)."""
        response = await self.client.get("/capi/v2/account/assets")
        if isinstance(response, list):
            return response
        return []

    async def get_bills(
        self,
        coin: Optional[str] = None,
        symbol: Optional[str] = None,
        business_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get contract account bill history (async)."""
        data: Dict[str, Any] = {}
        if coin:
            data["coin"] = coin
        if symbol:
            data["symbol"] = symbol
        if business_type:
            data["businessType"] = business_type
        if start_time:
            data["startTime"] = start_time
        if end_time:
            data["endTime"] = end_time
        if limit:
            data["limit"] = limit
        return await self.client.post("/capi/v2/account/bills", data=data)

    async def get_settings(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get account settings (async)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self.client.get("/capi/v2/account/settings", params=params)

    async def set_leverage(
        self,
        symbol: str,
        margin_mode: int,
        long_leverage: str,
        short_leverage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set leverage for a trading pair (async)."""
        if margin_mode == 3 and not short_leverage:
            short_leverage = long_leverage
        data = {
            "symbol": symbol,
            "marginMode": margin_mode,
            "longLeverage": str(long_leverage),
            "shortLeverage": str(short_leverage) if short_leverage else str(long_leverage),
        }
        return await self.client.post("/capi/v2/account/leverage", data=data)

    async def adjust_margin(
        self,
        isolated_position_id: int,
        collateral_amount: str,
        coin_id: int = 2,
    ) -> Dict[str, Any]:
        """Adjust margin for isolated position (async)."""
        data = {
            "coinId": coin_id,
            "isolatedPositionId": isolated_position_id,
            "collateralAmount": str(collateral_amount),
        }
        return await self.client.post("/capi/v2/account/adjustMargin", data=data)

    async def modify_auto_append_margin(
        self,
        position_id: int,
        auto_append_margin: bool,
    ) -> Dict[str, Any]:
        """Modify auto-append margin setting (async)."""
        data = {
            "positionId": position_id,
            "autoAppendMargin": auto_append_margin,
        }
        return await self.client.post("/capi/v2/account/modifyAutoAppendMargin", data=data)

    async def get_all_positions(self) -> List[Position]:
        """Get all positions (async)."""
        response = await self.client.get("/capi/v2/account/position/allPosition")
        if isinstance(response, list):
            return response
        return []

    async def get_single_position(self, symbol: str) -> Dict[str, Any]:
        """Get single position for a trading pair (async)."""
        return await self.client.get(
            "/capi/v2/account/position/singlePosition", params={"symbol": symbol}
        )

    async def change_hold_model(
        self,
        symbol: str,
        margin_mode: int,
        separated_mode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Change position holding model (async)."""
        data = {
            "symbol": symbol,
            "marginMode": margin_mode,
        }
        if separated_mode:
            data["separatedMode"] = separated_mode
        return await self.client.post("/capi/v2/account/position/changeHoldModel", data=data)
