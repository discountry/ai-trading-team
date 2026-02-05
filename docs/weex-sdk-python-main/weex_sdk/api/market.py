"""Market API module for Weex SDK."""

from typing import Any, Dict, List, Optional

from weex_sdk.client import AsyncWeexClient, WeexClient
from weex_sdk.models import Contract, Depth, Ticker, Trade


class MarketAPI:
    """Market API methods."""

    def __init__(self, client: WeexClient) -> None:
        """Initialize Market API.

        Args:
            client: WeexClient instance
        """
        self.client = client

    def get_server_time(self) -> Dict[str, Any]:
        """Get server time.

        Returns:
            Server time information (epoch, iso, timestamp)

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/market/time")

    def get_contracts(self, symbol: Optional[str] = None) -> List[Contract]:
        """Get contract information.

        Args:
            symbol: Trading pair (optional, returns all if not specified)

        Returns:
            List of contract information

        Raises:
            WeexAPIError: On API errors
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        response = self.client.get("/capi/v2/market/contracts", params=params)
        if isinstance(response, list):
            return response
        return []

    def get_depth(self, symbol: str, limit: int = 15) -> Depth:
        """Get market depth.

        Args:
            symbol: Trading pair
            limit: Fixed gear enumeration value: 15/200 (default: 15)

        Returns:
            Market depth data (asks, bids, timestamp)

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/market/depth", params={"symbol": symbol, "limit": limit})

    def get_tickers(self) -> List[Ticker]:
        """Get all tickers.

        Returns:
            List of ticker information for all trading pairs

        Raises:
            WeexAPIError: On API errors
        """
        response = self.client.get("/capi/v2/market/tickers")
        if isinstance(response, list):
            return response
        return []

    def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for a specific trading pair.

        Args:
            symbol: Trading pair

        Returns:
            Ticker information

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/market/ticker", params={"symbol": symbol})

    def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades to return (default: 100, max: 1000)

        Returns:
            List of recent trades

        Raises:
            WeexAPIError: On API errors
        """
        response = self.client.get(
            "/capi/v2/market/trades", params={"symbol": symbol, "limit": limit}
        )
        if isinstance(response, list):
            return response
        return []

    def get_candles(
        self,
        symbol: str,
        granularity: str,
        limit: int = 100,
        price_type: str = "LAST",
    ) -> List[List[str]]:
        """Get K-line/candlestick data.

        Args:
            symbol: Trading pair
            granularity: Candlestick interval [1m,5m,15m,30m,1h,4h,12h,1d,1w]
            limit: Number of candles to return (default: 100, max: 1000)
            price_type: Price Type: LAST (latest market price), MARK (mark), INDEX (index)

        Returns:
            List of candlestick data arrays

        Raises:
            WeexAPIError: On API errors
        """
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": limit,
            "priceType": price_type,
        }
        response = self.client.get("/capi/v2/market/candles", params=params)
        if isinstance(response, list):
            return response
        return []

    def get_history_candles(
        self,
        symbol: str,
        granularity: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
        price_type: str = "LAST",
    ) -> List[List[str]]:
        """Get historical K-line/candlestick data.

        Args:
            symbol: Trading pair
            granularity: Candlestick interval [1m,5m,15m,30m,1h,4h,12h,1d,1w]
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)
            limit: Number of candles to return (default: 100, max: 100)
            price_type: Price Type: LAST, MARK, INDEX

        Returns:
            List of historical candlestick data arrays

        Raises:
            WeexAPIError: On API errors
        """
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": limit,
            "priceType": price_type,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        response = self.client.get("/capi/v2/market/historyCandles", params=params)
        if isinstance(response, list):
            return response
        return []

    def get_index(self, symbol: str, price_type: str = "INDEX") -> Dict[str, Any]:
        """Get index price.

        Args:
            symbol: Trading pair
            price_type: Price Type: MARK (mark), INDEX (index, default)

        Returns:
            Index price information

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get(
            "/capi/v2/market/index", params={"symbol": symbol, "priceType": price_type}
        )

    def get_open_interest(self, symbol: str) -> List[Dict[str, Any]]:
        """Get open interest.

        Args:
            symbol: Trading pair

        Returns:
            Open interest information

        Raises:
            WeexAPIError: On API errors
        """
        response = self.client.get("/capi/v2/market/open_interest", params={"symbol": symbol})
        if isinstance(response, list):
            return response
        return []

    def get_funding_time(self, symbol: str) -> Dict[str, Any]:
        """Get funding fee settlement time.

        Args:
            symbol: Trading pair

        Returns:
            Funding time information

        Raises:
            WeexAPIError: On API errors
        """
        return self.client.get("/capi/v2/market/funding_time", params={"symbol": symbol})

    def get_history_fund_rate(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical funding rates.

        Args:
            symbol: Trading pair
            limit: Number of records to return (default: 10, max: 100)

        Returns:
            List of historical funding rate information

        Raises:
            WeexAPIError: On API errors
        """
        response = self.client.get(
            "/capi/v2/market/getHistoryFundRate",
            params={"symbol": symbol, "limit": limit},
        )
        if isinstance(response, list):
            return response
        return []

    def get_current_fund_rate(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current funding rate.

        Args:
            symbol: Trading pair (optional, returns all if not specified)

        Returns:
            List of current funding rate information

        Raises:
            WeexAPIError: On API errors
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        response = self.client.get("/capi/v2/market/currentFundRate", params=params)
        if isinstance(response, list):
            return response
        return []


class AsyncMarketAPI:
    """Async Market API methods."""

    def __init__(self, client: AsyncWeexClient) -> None:
        """Initialize Async Market API.

        Args:
            client: AsyncWeexClient instance
        """
        self.client = client

    async def get_server_time(self) -> Dict[str, Any]:
        """Get server time (async)."""
        return await self.client.get("/capi/v2/market/time")

    async def get_contracts(self, symbol: Optional[str] = None) -> List[Contract]:
        """Get contract information (async)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        response = await self.client.get("/capi/v2/market/contracts", params=params)
        if isinstance(response, list):
            return response
        return []

    async def get_depth(self, symbol: str, limit: int = 15) -> Depth:
        """Get market depth (async)."""
        return await self.client.get(
            "/capi/v2/market/depth", params={"symbol": symbol, "limit": limit}
        )

    async def get_tickers(self) -> List[Ticker]:
        """Get all tickers (async)."""
        response = await self.client.get("/capi/v2/market/tickers")
        if isinstance(response, list):
            return response
        return []

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for a specific trading pair (async)."""
        return await self.client.get("/capi/v2/market/ticker", params={"symbol": symbol})

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades (async)."""
        response = await self.client.get(
            "/capi/v2/market/trades", params={"symbol": symbol, "limit": limit}
        )
        if isinstance(response, list):
            return response
        return []

    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        limit: int = 100,
        price_type: str = "LAST",
    ) -> List[List[str]]:
        """Get K-line/candlestick data (async)."""
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": limit,
            "priceType": price_type,
        }
        response = await self.client.get("/capi/v2/market/candles", params=params)
        if isinstance(response, list):
            return response
        return []

    async def get_history_candles(
        self,
        symbol: str,
        granularity: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
        price_type: str = "LAST",
    ) -> List[List[str]]:
        """Get historical K-line/candlestick data (async)."""
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": limit,
            "priceType": price_type,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        response = await self.client.get("/capi/v2/market/historyCandles", params=params)
        if isinstance(response, list):
            return response
        return []

    async def get_index(self, symbol: str, price_type: str = "INDEX") -> Dict[str, Any]:
        """Get index price (async)."""
        return await self.client.get(
            "/capi/v2/market/index", params={"symbol": symbol, "priceType": price_type}
        )

    async def get_open_interest(self, symbol: str) -> List[Dict[str, Any]]:
        """Get open interest (async)."""
        response = await self.client.get("/capi/v2/market/open_interest", params={"symbol": symbol})
        if isinstance(response, list):
            return response
        return []

    async def get_funding_time(self, symbol: str) -> Dict[str, Any]:
        """Get funding fee settlement time (async)."""
        return await self.client.get("/capi/v2/market/funding_time", params={"symbol": symbol})

    async def get_history_fund_rate(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical funding rates (async)."""
        response = await self.client.get(
            "/capi/v2/market/getHistoryFundRate",
            params={"symbol": symbol, "limit": limit},
        )
        if isinstance(response, list):
            return response
        return []

    async def get_current_fund_rate(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current funding rate (async)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        response = await self.client.get("/capi/v2/market/currentFundRate", params=params)
        if isinstance(response, list):
            return response
        return []
