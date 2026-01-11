"""Real-time data pool with thread-safe storage."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any

from ai_trading_team.core.types import EventType


@dataclass
class DataSnapshot:
    """Point-in-time snapshot of all data."""

    timestamp: datetime
    ticker: dict[str, Any] | None = None
    klines: dict[str, list[dict[str, Any]]] | None = None  # interval -> klines
    orderbook: dict[str, Any] | None = None
    indicators: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    orders: list[dict[str, Any]] | None = None
    account: dict[str, Any] | None = None

    # Extended market data
    funding_rate: dict[str, Any] | None = None
    long_short_ratio: dict[str, Any] | None = None
    open_interest: dict[str, Any] | None = None
    mark_price: dict[str, Any] | None = None
    liquidations: list[dict[str, Any]] | None = None

    # Recent operation history for agent context
    recent_operations: list[dict[str, Any]] | None = None

    # Symbol info (precision, etc.)
    symbol_info: dict[str, Any] | None = None


class DataPool:
    """Thread-safe real-time data storage.

    Central repository for all market and account data.
    Supports subscriptions for reactive updates.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._ticker: dict[str, Any] | None = None
        self._klines: dict[str, list[dict[str, Any]]] = {}  # interval -> klines
        self._orderbook: dict[str, Any] | None = None
        self._indicators: dict[str, Any] = {}
        self._position: dict[str, Any] | None = None
        self._orders: list[dict[str, Any]] = []
        self._account: dict[str, Any] | None = None
        self._subscribers: list[Callable[[EventType, Any], None]] = []

        # Extended market data
        self._funding_rate: dict[str, Any] | None = None
        self._long_short_ratio: dict[str, Any] | None = None
        self._open_interest: dict[str, Any] | None = None
        self._mark_price: dict[str, Any] | None = None
        self._liquidations: list[dict[str, Any]] = []

        # Recent operation history (last 10)
        self._recent_operations: list[dict[str, Any]] = []
        self._max_operations_history = 10

        # Symbol info (precision, filters, etc.)
        self._symbol_info: dict[str, Any] | None = None

    def subscribe(self, callback: Callable[[EventType, Any], None]) -> None:
        """Subscribe to data updates.

        Args:
            callback: Function called with (event_type, data) on updates
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[EventType, Any], None]) -> None:
        """Unsubscribe from data updates."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify(self, event_type: EventType, data: Any) -> None:
        """Notify all subscribers of an update."""
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            callback(event_type, data)

    def update_ticker(self, ticker: dict[str, Any]) -> None:
        """Update ticker data."""
        with self._lock:
            self._ticker = ticker
        self._notify(EventType.TICKER_UPDATE, ticker)

    def update_klines(self, interval: str, klines: list[dict[str, Any]]) -> None:
        """Update kline data for an interval."""
        with self._lock:
            self._klines[interval] = klines
        self._notify(EventType.KLINE_UPDATE, {"interval": interval, "klines": klines})

    def update_orderbook(self, orderbook: dict[str, Any]) -> None:
        """Update orderbook data."""
        with self._lock:
            self._orderbook = orderbook
        self._notify(EventType.ORDERBOOK_UPDATE, orderbook)

    def update_indicator(self, name: str, value: Any) -> None:
        """Update indicator value."""
        with self._lock:
            self._indicators[name] = value
        self._notify(EventType.INDICATOR_UPDATE, {"name": name, "value": value})

    def update_position(self, position: dict[str, Any] | None) -> None:
        """Update position data."""
        with self._lock:
            self._position = position
        self._notify(EventType.POSITION_UPDATED, position)

    def update_orders(self, orders: list[dict[str, Any]]) -> None:
        """Update open orders."""
        with self._lock:
            self._orders = orders

    def update_account(self, account: dict[str, Any]) -> None:
        """Update account data."""
        with self._lock:
            self._account = account

    def update_funding_rate(self, funding_rate: dict[str, Any]) -> None:
        """Update funding rate data."""
        with self._lock:
            self._funding_rate = funding_rate
        self._notify(EventType.FUNDING_RATE_UPDATE, funding_rate)

    def update_long_short_ratio(self, ratio: dict[str, Any]) -> None:
        """Update long/short ratio data."""
        with self._lock:
            self._long_short_ratio = ratio

    def update_open_interest(self, oi: dict[str, Any]) -> None:
        """Update open interest data."""
        with self._lock:
            self._open_interest = oi

    def update_mark_price(self, mark_price: dict[str, Any]) -> None:
        """Update mark price data."""
        with self._lock:
            self._mark_price = mark_price
        self._notify(EventType.MARK_PRICE_UPDATE, mark_price)

    def add_liquidation(self, liquidation: dict[str, Any]) -> None:
        """Add a liquidation event."""
        with self._lock:
            self._liquidations.append(liquidation)
            # Keep only last 100 liquidations
            if len(self._liquidations) > 100:
                self._liquidations = self._liquidations[-100:]
        self._notify(EventType.LIQUIDATION_UPDATE, liquidation)

    def add_operation(self, operation: dict[str, Any]) -> None:
        """Add an operation to history.

        Args:
            operation: Operation record with timestamp, action, result, etc.
        """
        with self._lock:
            self._recent_operations.append(operation)
            # Keep only last N operations
            if len(self._recent_operations) > self._max_operations_history:
                self._recent_operations = self._recent_operations[-self._max_operations_history:]

    def update_symbol_info(self, symbol_info: dict[str, Any]) -> None:
        """Update symbol info (precision, filters, etc.)."""
        with self._lock:
            self._symbol_info = symbol_info

    def get_snapshot(self) -> DataSnapshot:
        """Get a point-in-time snapshot of all data."""
        with self._lock:
            return DataSnapshot(
                timestamp=datetime.now(),
                ticker=self._ticker.copy() if self._ticker else None,
                klines={k: list(v) for k, v in self._klines.items()},
                orderbook=self._orderbook.copy() if self._orderbook else None,
                indicators=self._indicators.copy(),
                position=self._position.copy() if self._position else None,
                orders=list(self._orders),
                account=self._account.copy() if self._account else None,
                funding_rate=self._funding_rate.copy() if self._funding_rate else None,
                long_short_ratio=self._long_short_ratio.copy() if self._long_short_ratio else None,
                open_interest=self._open_interest.copy() if self._open_interest else None,
                mark_price=self._mark_price.copy() if self._mark_price else None,
                liquidations=list(self._liquidations) if self._liquidations else None,
                recent_operations=list(self._recent_operations),
                symbol_info=self._symbol_info.copy() if self._symbol_info else None,
            )

    @property
    def ticker(self) -> dict[str, Any] | None:
        """Get current ticker."""
        with self._lock:
            return self._ticker.copy() if self._ticker else None

    @property
    def indicators(self) -> dict[str, Any]:
        """Get current indicators."""
        with self._lock:
            return self._indicators.copy()

    def get_klines(self, interval: str) -> list[dict[str, Any]]:
        """Get klines for an interval."""
        with self._lock:
            return list(self._klines.get(interval, []))
