"""Binance WebSocket stream client."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import websockets
from binance_common.configuration import ConfigurationWebSocketStreams
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

from ai_trading_team.data.models import Kline, Ticker

logger = logging.getLogger(__name__)


class BinanceStreamClient:
    """Binance Futures WebSocket stream client.

    Receives real-time market data from Binance USDS-M Futures.
    Features:
    - Automatic reconnection with exponential backoff
    - Connection state tracking
    - Callback-based event handling
    """

    # Reconnection settings
    MIN_RECONNECT_DELAY = 1.0  # Initial delay in seconds
    MAX_RECONNECT_DELAY = 60.0  # Maximum delay in seconds
    RECONNECT_MULTIPLIER = 2.0  # Exponential backoff multiplier

    def __init__(self) -> None:
        """Initialize Binance stream client."""
        self._client: DerivativesTradingUsdsFutures | None = None
        self._connection: Any = None
        self._running = False
        self._connected = False
        self._callbacks: dict[str, list[Callable[..., None]]] = {
            "ticker": [],
            "kline": [],
            "trade": [],
            "depth": [],
            "liquidation": [],
            "reconnect": [],  # Callback for reconnection events
        }
        # Reconnection state
        self._reconnect_delay = self.MIN_RECONNECT_DELAY
        self._reconnect_task: asyncio.Task | None = None
        self._symbol: str | None = None
        self._kline_intervals: list[str] = ["1m"]
        # Raw depth WebSocket (bypass buggy SDK parsing)
        self._depth_ws: Any = None
        self._depth_task: asyncio.Task | None = None

    def _get_client(self) -> DerivativesTradingUsdsFutures:
        """Get or create Binance client."""
        if self._client is None:
            config = ConfigurationWebSocketStreams(
                reconnect_delay=5000,
            )
            self._client = DerivativesTradingUsdsFutures(config_ws_streams=config)
        return self._client

    def on_ticker(self, callback: Callable[[Ticker], None]) -> None:
        """Register ticker callback.

        Args:
            callback: Function to call with Ticker updates
        """
        self._callbacks["ticker"].append(callback)

    def on_kline(self, callback: Callable[[str, Kline], None]) -> None:
        """Register kline callback.

        Args:
            callback: Function to call with (interval, Kline) updates
        """
        self._callbacks["kline"].append(callback)

    def on_depth(self, callback: Callable[[list, list], None]) -> None:
        """Register depth (orderbook) callback.

        Args:
            callback: Function to call with (bids, asks) updates
        """
        self._callbacks["depth"].append(callback)

    def on_liquidation(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register liquidation callback.

        Args:
            callback: Function to call with liquidation event data
        """
        self._callbacks["liquidation"].append(callback)

    def on_reconnect(self, callback: Callable[[int], None]) -> None:
        """Register reconnection callback.

        Args:
            callback: Function to call with reconnection attempt count
        """
        self._callbacks["reconnect"].append(callback)

    def _notify_reconnect(self, attempt: int) -> None:
        """Notify reconnection callbacks.

        Args:
            attempt: Current reconnection attempt number
        """
        for callback in self._callbacks["reconnect"]:
            try:
                callback(attempt)
            except Exception as e:
                logger.error(f"Error in reconnect callback: {e}")

    async def connect(self, symbol: str, kline_intervals: list[str] | str = "1m") -> None:
        """Connect to WebSocket streams.

        Args:
            symbol: Trading pair (e.g., "btcusdt")
            kline_intervals: K-line interval(s) (e.g., "1m", ["1m", "15m"])
        """
        # Store connection parameters for reconnection
        self._symbol = symbol
        if isinstance(kline_intervals, str):
            intervals = [kline_intervals]
        else:
            intervals = list(kline_intervals)
        if not intervals:
            intervals = ["1m"]
        self._kline_intervals = list(dict.fromkeys(intervals))

        client = self._get_client()
        self._running = True

        try:
            self._connection = await client.websocket_streams.create_connection()
            self._connected = True
            self._reconnect_delay = self.MIN_RECONNECT_DELAY  # Reset on successful connect
            logger.info("WebSocket connection established")

            # Subscribe to ticker stream
            symbol_lower = symbol.lower()
            ticker_stream = await self._connection.individual_symbol_ticker_streams(
                symbol=symbol_lower
            )
            ticker_stream.on("message", self._handle_ticker)
            logger.info(f"Subscribed to ticker stream: {symbol_lower}")

            # Subscribe to kline streams
            for interval in self._kline_intervals:
                kline_stream = await self._connection.kline_candlestick_streams(
                    symbol=symbol_lower,
                    interval=interval,
                )
                kline_stream.on("message", self._handle_kline)
                logger.info(f"Subscribed to kline stream: {symbol_lower}@{interval}")

            # Subscribe to liquidation stream
            try:
                liquidation_stream = await self._connection.liquidation_order_streams(
                    symbol=symbol_lower,
                )
                liquidation_stream.on("message", self._handle_liquidation)
                logger.info(f"Subscribed to liquidation stream: {symbol_lower}@forceOrder")
            except Exception as e:
                logger.warning(f"Failed to subscribe to liquidation stream: {e}")

            # Start raw depth WebSocket (bypass buggy SDK parsing)
            self._depth_task = asyncio.create_task(self._run_depth_stream(symbol_lower))
            logger.info(f"Started raw depth stream: {symbol_lower}@depth10")

        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect WebSocket: {e}")
            raise

    async def _run_depth_stream(self, symbol: str) -> None:
        """Run raw WebSocket for depth data.

        The Binance SDK has a bug where it doesn't parse depth messages correctly.
        We use raw websockets to bypass the SDK and get proper orderbook data.

        Args:
            symbol: Trading pair in lowercase
        """
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@depth10@100ms"

        while self._running:
            try:
                async with websockets.connect(uri) as ws:
                    self._depth_ws = ws
                    logger.info(f"Raw depth WebSocket connected: {uri}")

                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(message)
                            self._handle_raw_depth(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse depth message: {e}")

            except websockets.exceptions.ConnectionClosed as e:
                if self._running:
                    logger.warning(f"Depth WebSocket closed: {e}, reconnecting...")
                    await asyncio.sleep(1)
            except Exception as e:
                if self._running:
                    logger.error(f"Depth WebSocket error: {e}, reconnecting...")
                    await asyncio.sleep(1)

        self._depth_ws = None

    def _handle_raw_depth(self, data: dict) -> None:
        """Handle raw depth data from WebSocket.

        Args:
            data: Raw JSON data with 'bids' and 'asks' keys
        """
        try:
            bids_data = data.get("bids", [])
            asks_data = data.get("asks", [])

            bids = [[str(b[0]), str(b[1])] for b in bids_data if len(b) >= 2]
            asks = [[str(a[0]), str(a[1])] for a in asks_data if len(a) >= 2]

            if not bids and not asks:
                return

            # Validate orderbook integrity
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                if best_bid >= best_ask:
                    logger.warning(
                        f"Invalid orderbook: best_bid({best_bid}) >= best_ask({best_ask})"
                    )
                    return

            # Notify callbacks
            for callback in self._callbacks["depth"]:
                callback(bids, asks)

        except Exception as e:
            logger.error(f"Error handling raw depth: {e}")

    def _handle_ticker(self, message: Any) -> None:
        """Handle incoming ticker data."""
        try:
            # Convert Pydantic model to dict
            data = message.to_dict() if hasattr(message, "to_dict") else message

            ticker = Ticker(
                symbol=data.get("s", ""),
                last_price=Decimal(str(data.get("c", 0))),
                bid_price=Decimal(str(data.get("b", 0))),
                ask_price=Decimal(str(data.get("a", 0))),
                high_24h=Decimal(str(data.get("h", 0))),
                low_24h=Decimal(str(data.get("l", 0))),
                volume_24h=Decimal(str(data.get("v", 0))),
                price_change_percent=Decimal(str(data.get("P", 0))),
                timestamp=datetime.fromtimestamp(data.get("E", 0) / 1000)
                if data.get("E")
                else datetime.now(),
            )

            for callback in self._callbacks["ticker"]:
                callback(ticker)

        except Exception as e:
            logger.error(f"Error handling ticker: {e}")

    def _handle_kline(self, message: Any) -> None:
        """Handle incoming kline data."""
        try:
            # Convert Pydantic model to dict
            data = message.to_dict() if hasattr(message, "to_dict") else message

            kline_data = data.get("k", {})
            interval = kline_data.get("i", "")

            kline = Kline(
                open_time=datetime.fromtimestamp(kline_data.get("t", 0) / 1000),
                open=Decimal(str(kline_data.get("o", 0))),
                high=Decimal(str(kline_data.get("h", 0))),
                low=Decimal(str(kline_data.get("l", 0))),
                close=Decimal(str(kline_data.get("c", 0))),
                volume=Decimal(str(kline_data.get("v", 0))),
                close_time=datetime.fromtimestamp(kline_data.get("T", 0) / 1000),
                quote_volume=Decimal(str(kline_data.get("q", 0))),
                trades=int(kline_data.get("n", 0)),
            )

            for callback in self._callbacks["kline"]:
                callback(interval, kline)

        except Exception as e:
            logger.error(f"Error handling kline: {e}")

    def _handle_liquidation(self, message: Any) -> None:
        """Handle incoming liquidation data."""
        try:
            data = message.to_dict() if hasattr(message, "to_dict") else message
            event = data
            if isinstance(data, dict) and isinstance(data.get("data"), dict):
                event = data.get("data")
            if not isinstance(event, dict):
                return
            for callback in self._callbacks["liquidation"]:
                callback(event)
        except Exception as e:
            logger.error(f"Error handling liquidation: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        self._connected = False

        # Cancel depth stream task
        if self._depth_task:
            self._depth_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._depth_task
            self._depth_task = None

        # Close raw depth WebSocket
        if self._depth_ws:
            with contextlib.suppress(Exception):
                await self._depth_ws.close()
            self._depth_ws = None

        if self._connection:
            try:
                await self._connection.close_connection(close_session=True)
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self._connection = None

    async def run_forever(self) -> None:
        """Keep the connection running with automatic reconnection."""
        reconnect_attempt = 0

        while self._running:
            if not self._connected and self._symbol:
                reconnect_attempt += 1
                logger.warning(
                    f"WebSocket disconnected. Reconnecting in {self._reconnect_delay:.1f}s "
                    f"(attempt {reconnect_attempt})..."
                )
                self._notify_reconnect(reconnect_attempt)

                await asyncio.sleep(self._reconnect_delay)

                # Exponential backoff
                self._reconnect_delay = min(
                    self._reconnect_delay * self.RECONNECT_MULTIPLIER,
                    self.MAX_RECONNECT_DELAY,
                )

                try:
                    await self.connect(self._symbol, self._kline_intervals)
                    reconnect_attempt = 0  # Reset on success
                    logger.info("WebSocket reconnected successfully")
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
                    self._connected = False
            else:
                await asyncio.sleep(1)

    async def _reconnect(self) -> None:
        """Internal reconnection handler.

        Called when connection is lost. Implements exponential backoff.
        """
        if not self._symbol:
            logger.warning("Cannot reconnect: no symbol configured")
            return

        self._connected = False
        reconnect_attempt = 0

        while self._running and not self._connected:
            reconnect_attempt += 1
            logger.info(
                f"Attempting reconnection {reconnect_attempt} in {self._reconnect_delay:.1f}s..."
            )
            self._notify_reconnect(reconnect_attempt)

            await asyncio.sleep(self._reconnect_delay)

            # Increase delay with exponential backoff
            self._reconnect_delay = min(
                self._reconnect_delay * self.RECONNECT_MULTIPLIER,
                self.MAX_RECONNECT_DELAY,
            )

            try:
                await self.connect(self._symbol, self._kline_intervals)
                logger.info("Reconnected successfully")
            except Exception as e:
                logger.error(f"Reconnection attempt {reconnect_attempt} failed: {e}")
