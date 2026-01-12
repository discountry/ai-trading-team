"""Binance WebSocket stream client."""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

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
            "reconnect": [],  # Callback for reconnection events
        }
        # Reconnection state
        self._reconnect_delay = self.MIN_RECONNECT_DELAY
        self._reconnect_task: asyncio.Task | None = None
        self._symbol: str | None = None
        self._kline_interval: str = "1m"

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

    async def connect(self, symbol: str, kline_interval: str = "1m") -> None:
        """Connect to WebSocket streams.

        Args:
            symbol: Trading pair (e.g., "btcusdt")
            kline_interval: K-line interval (e.g., "1m", "5m")
        """
        # Store connection parameters for reconnection
        self._symbol = symbol
        self._kline_interval = kline_interval

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

            # Subscribe to kline stream
            kline_stream = await self._connection.kline_candlestick_streams(
                symbol=symbol_lower,
                interval=kline_interval,
            )
            kline_stream.on("message", self._handle_kline)
            logger.info(f"Subscribed to kline stream: {symbol_lower}@{kline_interval}")

            # Subscribe to partial depth stream for orderbook updates
            # Using partial_book_depth_streams instead of diff_book_depth_streams
            # because partial streams send complete top N levels every update,
            # avoiding the need for complex diff synchronization
            depth_stream = await self._connection.partial_book_depth_streams(
                symbol=symbol_lower,
                levels=10,  # Top 10 price levels on each side
            )
            depth_stream.on("message", self._handle_partial_depth)
            logger.info(f"Subscribed to partial depth stream: {symbol_lower}@depth10")

        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect WebSocket: {e}")
            raise

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

    def _handle_depth(self, message: Any) -> None:
        """Handle incoming depth (orderbook) diff data.

        NOTE: This is for diff_book_depth_streams (currently unused).
        We now use _handle_partial_depth for partial_book_depth_streams.
        """
        try:
            # Convert Pydantic model to dict first (like ticker/kline handlers)
            data = message.to_dict() if hasattr(message, "to_dict") else message

            bids = []
            asks = []

            # Get bids from 'b' key (WebSocket uses short names)
            for item in data.get("b", []):
                if isinstance(item, list | tuple) and len(item) >= 2:
                    bids.append([str(item[0]), str(item[1])])

            # Get asks from 'a' key
            for item in data.get("a", []):
                if isinstance(item, list | tuple) and len(item) >= 2:
                    asks.append([str(item[0]), str(item[1])])

            # Log first update to verify data structure
            if bids or asks:
                if bids:
                    logger.debug(f"Depth bids sample: {bids[0] if bids else 'none'}")
                if asks:
                    logger.debug(f"Depth asks sample: {asks[0] if asks else 'none'}")
                for callback in self._callbacks["depth"]:
                    callback(bids, asks)

        except Exception as e:
            logger.error(f"Error handling depth: {e}")

    def _handle_partial_depth(self, message: Any) -> None:
        """Handle incoming partial depth (orderbook) data.

        Partial depth stream sends complete top N levels, not diffs.
        This simplifies orderbook management since we don't need to track state.
        """
        try:
            # Convert Pydantic model to dict first
            data = message.to_dict() if hasattr(message, "to_dict") else message

            bids = []
            asks = []

            # Parse bids from 'b' or 'bids' key
            bids_data = data.get("b") or data.get("bids", [])
            for item in bids_data:
                # Handle different formats
                if hasattr(item, "root"):
                    # Pydantic model with root attribute
                    price, qty = item.root
                    bids.append([str(price), str(qty)])
                elif isinstance(item, list | tuple) and len(item) >= 2:
                    bids.append([str(item[0]), str(item[1])])

            # Parse asks from 'a' or 'asks' key
            asks_data = data.get("a") or data.get("asks", [])
            for item in asks_data:
                if hasattr(item, "root"):
                    price, qty = item.root
                    asks.append([str(price), str(qty)])
                elif isinstance(item, list | tuple) and len(item) >= 2:
                    asks.append([str(item[0]), str(item[1])])

            # Validate orderbook integrity
            if bids and asks:
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                if best_bid > 0 and best_ask > 0 and best_bid >= best_ask:
                    logger.warning(
                        f"Invalid orderbook: best_bid({best_bid}) >= best_ask({best_ask}), skipping"
                    )
                    return

                logger.debug(
                    f"Partial depth: {len(bids)} bids, {len(asks)} asks, "
                    f"best_bid={bids[0][0]}, best_ask={asks[0][0]}"
                )

            # Notify callbacks with full orderbook data
            for callback in self._callbacks["depth"]:
                callback(bids, asks)

        except Exception as e:
            logger.error(f"Error handling partial depth: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        self._connected = False
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
                    await self.connect(self._symbol, self._kline_interval)
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
                await self.connect(self._symbol, self._kline_interval)
                logger.info("Reconnected successfully")
            except Exception as e:
                logger.error(f"Reconnection attempt {reconnect_attempt} failed: {e}")
