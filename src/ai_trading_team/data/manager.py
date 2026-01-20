"""Data manager - coordinates REST and WebSocket data sources."""

import logging
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.data.binance.rest import BinanceRestClient
from ai_trading_team.data.binance.stream import BinanceStreamClient
from ai_trading_team.data.models import Kline, Ticker

logger = logging.getLogger(__name__)


class OrderBookManager:
    """Manages local orderbook state with WebSocket diff updates.

    Maintains a local copy of the orderbook by:
    1. Getting initial snapshot via REST API
    2. Applying incremental updates from WebSocket diff stream

    Uses Decimal for price keys to ensure consistent comparison.
    """

    def __init__(self, max_levels: int = 10) -> None:
        self._bids: dict[Decimal, Decimal] = {}  # price -> quantity
        self._asks: dict[Decimal, Decimal] = {}  # price -> quantity
        self._max_levels = max_levels
        self._last_update_id: int = 0
        self._initialized = False

    def set_snapshot(
        self, bids: list, asks: list, last_update_id: int = 0, *, log_info: bool = True
    ) -> None:
        """Set orderbook from REST snapshot or partial depth stream.

        Args:
            bids: List of [price, quantity] pairs
            asks: List of [price, quantity] pairs
            last_update_id: Last update ID from snapshot
            log_info: Whether to log info (set False for frequent WebSocket updates)
        """
        self._bids.clear()
        self._asks.clear()

        for bid in bids:
            price = Decimal(str(bid[0]))
            qty = Decimal(str(bid[1]))
            if qty > 0:
                self._bids[price] = qty

        for ask in asks:
            price = Decimal(str(ask[0]))
            qty = Decimal(str(ask[1]))
            if qty > 0:
                self._asks[price] = qty

        self._last_update_id = last_update_id
        self._initialized = True

        # Log sample to verify correct bid/ask separation (only on initial snapshot)
        if log_info and self._bids and self._asks:
            best_bid = max(self._bids.keys())
            best_ask = min(self._asks.keys())
            logger.info(
                f"OrderBook snapshot: {len(self._bids)} bids, {len(self._asks)} asks, "
                f"best_bid={best_bid}, best_ask={best_ask}, spread={best_ask - best_bid}"
            )
        elif log_info:
            logger.debug(f"OrderBook snapshot set: {len(self._bids)} bids, {len(self._asks)} asks")

    def apply_diff(self, bids: list, asks: list) -> None:
        """Apply diff update from WebSocket.

        Args:
            bids: List of [price, quantity] bid updates
            asks: List of [price, quantity] ask updates
        """
        if not self._initialized:
            return

        # Apply bid updates
        for bid in bids:
            price = Decimal(str(bid[0]))
            qty = Decimal(str(bid[1]))
            if qty == 0:
                # Remove price level
                self._bids.pop(price, None)
            else:
                # Update price level
                self._bids[price] = qty

        # Apply ask updates
        for ask in asks:
            price = Decimal(str(ask[0]))
            qty = Decimal(str(ask[1]))
            if qty == 0:
                # Remove price level
                self._asks.pop(price, None)
            else:
                # Update price level
                self._asks[price] = qty

    def get_orderbook(self) -> dict[str, list]:
        """Get current orderbook state as sorted lists.

        Returns:
            Dict with 'bids' and 'asks' as sorted [price, qty] lists
        """
        # Sort bids descending by price (highest first), take top N
        sorted_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[
            : self._max_levels
        ]

        # Sort asks ascending by price (lowest first), take top N
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])[: self._max_levels]

        # Convert to string format for JSON serialization
        return {
            "bids": [[str(p), str(q)] for p, q in sorted_bids],
            "asks": [[str(p), str(q)] for p, q in sorted_asks],
        }


class BinanceDataManager:
    """Binance data manager.

    Coordinates REST API and WebSocket streams,
    writes updates to the DataPool.
    """

    # Required intervals for signal system
    REQUIRED_INTERVALS = ["15m", "1h", "4h"]
    MIN_KLINES_PER_INTERVAL = 100  # Need enough for MA60 + buffer

    def __init__(
        self,
        data_pool: DataPool,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        """Initialize Binance data manager.

        Args:
            data_pool: Shared data pool for storing real-time data
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self._data_pool = data_pool
        self._rest_client = BinanceRestClient(api_key, api_secret)
        self._stream_client = BinanceStreamClient()
        self._running = False
        self._symbol = ""
        self._kline_intervals: list[str] = ["1m"]
        self._data_ready = False
        self._orderbook_manager = OrderBookManager(max_levels=10)

    async def start(
        self,
        symbol: str,
        kline_interval: str = "1m",
        kline_intervals: list[str] | None = None,
    ) -> None:
        """Start data collection for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            kline_interval: Primary Kline interval for streaming (e.g., "15m", "1h")
            kline_intervals: Optional list of intervals to subscribe via WebSocket
        """
        self._symbol = symbol
        if kline_intervals is None:
            intervals = [kline_interval]
        else:
            intervals = list(kline_intervals)
            if kline_interval not in intervals:
                intervals.insert(0, kline_interval)
        self._kline_intervals = list(dict.fromkeys(intervals))
        self._running = True

        # Initialize historical data first
        await self.initialize(symbol)

        # Set up callbacks
        self._stream_client.on_ticker(self._on_ticker)
        self._stream_client.on_kline(self._on_kline)
        self._stream_client.on_depth(self._on_depth)
        self._stream_client.on_liquidation(self._on_liquidation)

        # Connect to WebSocket streams
        await self._stream_client.connect(symbol, self._kline_intervals)
        logger.info(f"Started data collection for {symbol}")

    async def stop(self) -> None:
        """Stop data collection."""
        self._running = False
        await self._stream_client.close()
        logger.info("Stopped data collection")

    async def initialize(self, symbol: str) -> None:
        """Initialize historical data via REST API.

        Fetches klines for all required timeframes (15m, 1h, 4h)
        to ensure signal system has sufficient data.

        Args:
            symbol: Trading pair
        """
        logger.info(f"Initializing historical data for {symbol}")
        self._data_ready = False

        try:
            # Fetch symbol info (precision, filters)
            symbol_info = await self._rest_client.get_symbol_info(symbol)
            if symbol_info:
                self._data_pool.update_symbol_info(symbol_info)
                logger.info(
                    f"Symbol info: pricePrecision={symbol_info.get('pricePrecision')}, "
                    f"quantityPrecision={symbol_info.get('quantityPrecision')}"
                )

            # Fetch initial ticker
            ticker = await self._rest_client.get_ticker(symbol)
            self._data_pool.update_ticker(self._ticker_to_dict(ticker))

            # Fetch historical klines for all required intervals
            for interval in self.REQUIRED_INTERVALS:
                klines = await self._rest_client.get_klines(
                    symbol, interval, limit=self.MIN_KLINES_PER_INTERVAL
                )
                kline_dicts = [self._kline_to_dict(k) for k in klines]
                self._data_pool.update_klines(interval, kline_dicts)
                logger.info(f"Initialized {len(klines)} klines for {symbol} {interval}")

            # Also fetch any streaming intervals not in required list
            for interval in self._kline_intervals:
                if interval in self.REQUIRED_INTERVALS:
                    continue
                klines = await self._rest_client.get_klines(
                    symbol, interval, limit=self.MIN_KLINES_PER_INTERVAL
                )
                kline_dicts = [self._kline_to_dict(k) for k in klines]
                self._data_pool.update_klines(interval, kline_dicts)
                logger.info(f"Initialized {len(klines)} klines for {symbol} {interval}")

            # Fetch extended market data
            await self._fetch_extended_data(symbol)

            self._data_ready = True
            logger.info(
                f"Data initialization complete: {len(self.REQUIRED_INTERVALS)} timeframes loaded"
            )

        except Exception as e:
            logger.error(f"Failed to initialize data: {e}")
            raise

    async def _fetch_extended_data(self, symbol: str) -> None:
        """Fetch extended market data (funding rate, L/S ratio, open interest, orderbook).

        Args:
            symbol: Trading pair
        """
        try:
            # Fetch initial orderbook via REST and initialize the orderbook manager
            orderbook = await self._rest_client.get_orderbook(symbol, limit=20)
            bids = [[str(level.price), str(level.quantity)] for level in orderbook.bids]
            asks = [[str(level.price), str(level.quantity)] for level in orderbook.asks]

            # Initialize orderbook manager with snapshot
            self._orderbook_manager.set_snapshot(
                bids=bids, asks=asks, last_update_id=orderbook.last_update_id or 0
            )

            # Update data pool with initial orderbook
            self._data_pool.update_orderbook(self._orderbook_manager.get_orderbook())
            logger.info(f"Initialized orderbook for {symbol}: {len(bids)} bids, {len(asks)} asks")
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook: {e}")

        try:
            # Fetch funding rate
            funding = await self._rest_client.get_funding_rate(symbol)
            self._data_pool.update_funding_rate(
                {
                    "symbol": symbol,
                    "funding_rate": float(funding.funding_rate),
                    "funding_time": funding.funding_time.isoformat()
                    if funding.funding_time
                    else None,
                }
            )
            logger.info(f"Fetched funding rate for {symbol}: {funding.funding_rate}")
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate: {e}")

        try:
            # Fetch open interest
            oi = await self._rest_client.get_open_interest(symbol)
            self._data_pool.update_open_interest(
                {
                    "symbol": symbol,
                    "open_interest": float(oi.open_interest),
                }
            )
            logger.info(f"Fetched open interest for {symbol}: {oi.open_interest}")
        except Exception as e:
            logger.warning(f"Failed to fetch open interest: {e}")

        try:
            # Fetch long/short ratio
            ls = await self._rest_client.get_long_short_ratio(symbol)
            self._data_pool.update_long_short_ratio(
                {
                    "symbol": symbol,
                    "long_ratio": float(ls.long_ratio),
                    "short_ratio": float(ls.short_ratio),
                    "long_short_ratio": float(ls.long_short_ratio),
                }
            )
            logger.info(f"Fetched L/S ratio for {symbol}: {ls.long_short_ratio}")
        except Exception as e:
            logger.warning(f"Failed to fetch L/S ratio: {e}")

    def _on_ticker(self, ticker: Ticker) -> None:
        """Handle ticker update from WebSocket."""
        self._data_pool.update_ticker(self._ticker_to_dict(ticker))

    def _on_kline(self, interval: str, kline: Kline) -> None:
        """Handle kline update from WebSocket."""
        # Get existing klines and update/append
        existing = self._data_pool.get_klines(interval)
        kline_dict = self._kline_to_dict(kline)

        # Check if this is an update to the last kline or a new one
        if existing and existing[-1].get("open_time") == kline_dict.get("open_time"):
            # Update the last kline
            existing[-1] = kline_dict
        else:
            # Append new kline
            existing.append(kline_dict)
            # Keep only the last 500 klines
            if len(existing) > 500:
                existing = existing[-500:]

        self._data_pool.update_klines(interval, existing)

    def _on_depth(self, bids: list, asks: list) -> None:
        """Handle depth (orderbook) update from WebSocket.

        With partial_book_depth_streams, we receive complete orderbook snapshots,
        not diffs. So we can directly update the data pool without the OrderBookManager.
        """
        # For partial depth streams, we receive full orderbook snapshots
        # Use OrderBookManager.set_snapshot() to replace the entire orderbook
        # log_info=False to avoid flooding logs with every WebSocket update
        if not bids and not asks:
            logger.debug("Empty orderbook update received, skipping")
            return

        self._orderbook_manager.set_snapshot(bids, asks, log_info=False)

        # Update data pool with current orderbook state
        orderbook = self._orderbook_manager.get_orderbook()
        if orderbook.get("bids") or orderbook.get("asks"):
            self._data_pool.update_orderbook(orderbook)
        else:
            logger.warning("OrderBookManager returned empty orderbook after set_snapshot")

    def _on_liquidation(self, event: dict[str, Any]) -> None:
        """Handle liquidation event update from WebSocket."""
        if not isinstance(event, dict):
            return
        symbol = event.get("symbol") or event.get("s")
        order = event.get("o")
        if not symbol and isinstance(order, dict):
            symbol = order.get("s")
        if symbol and str(symbol).upper() != self._symbol.upper():
            return
        self._data_pool.add_liquidation(event)

    @staticmethod
    def _ticker_to_dict(ticker: Ticker) -> dict[str, Any]:
        """Convert Ticker to dict with serializable values."""
        result = asdict(ticker)
        for key, value in result.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
        return result

    @staticmethod
    def _kline_to_dict(kline: Kline) -> dict[str, Any]:
        """Convert Kline to dict with serializable values."""
        result = asdict(kline)
        for key, value in result.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
        return result

    @property
    def is_running(self) -> bool:
        """Check if data collection is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._stream_client.is_connected

    @property
    def is_data_ready(self) -> bool:
        """Check if all required historical data is loaded."""
        return self._data_ready
