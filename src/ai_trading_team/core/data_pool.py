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

    # Trading statistics
    trading_stats: dict[str, Any] | None = None


@dataclass
class TradingStats:
    """Trading session statistics."""

    # Basic counters
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    total_fees: float = 0.0

    # Initial capital tracking
    initial_balance: float = 0.0
    current_equity: float = 0.0

    # Time tracking
    session_start: datetime | None = None
    last_trade_time: datetime | None = None

    # Per-trade details (last 20)
    trade_history: list[dict[str, Any]] | None = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.largest_loss == 0:
            return float("inf") if self.largest_win > 0 else 0.0
        gross_profit = sum(
            t.get("pnl", 0) for t in (self.trade_history or []) if t.get("pnl", 0) > 0
        )
        gross_loss = abs(
            sum(t.get("pnl", 0) for t in (self.trade_history or []) if t.get("pnl", 0) < 0)
        )
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        """Average winning trade P&L."""
        if self.winning_trades == 0:
            return 0.0
        total_wins = sum(t.get("pnl", 0) for t in (self.trade_history or []) if t.get("pnl", 0) > 0)
        return total_wins / self.winning_trades

    @property
    def avg_loss(self) -> float:
        """Average losing trade P&L."""
        if self.losing_trades == 0:
            return 0.0
        total_losses = sum(
            t.get("pnl", 0) for t in (self.trade_history or []) if t.get("pnl", 0) < 0
        )
        return total_losses / self.losing_trades

    @property
    def total_return_percent(self) -> float:
        """Total return as percentage of initial balance."""
        if self.initial_balance == 0:
            return 0.0
        return ((self.current_equity - self.initial_balance) / self.initial_balance) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2)
            if self.profit_factor != float("inf")
            else "∞",
            "total_return_percent": round(self.total_return_percent, 2),
            "initial_balance": round(self.initial_balance, 2),
            "current_equity": round(self.current_equity, 2),
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
        }


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

        # Trading statistics
        self._trading_stats = TradingStats()
        self._max_trade_history = 20

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
        stored = klines[-1000:] if len(klines) > 1000 else klines
        with self._lock:
            self._klines[interval] = stored
        self._notify(EventType.KLINE_UPDATE, {"interval": interval, "klines": stored})

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
                self._recent_operations = self._recent_operations[-self._max_operations_history :]

    def update_symbol_info(self, symbol_info: dict[str, Any]) -> None:
        """Update symbol info (precision, filters, etc.)."""
        with self._lock:
            self._symbol_info = symbol_info

    def init_trading_stats(self, initial_balance: float) -> None:
        """Initialize trading stats at session start.

        Args:
            initial_balance: Starting account balance in USDT
        """
        with self._lock:
            self._trading_stats = TradingStats(
                initial_balance=initial_balance,
                current_equity=initial_balance,
                session_start=datetime.now(),
                trade_history=[],
            )

    def record_trade(
        self,
        pnl: float,
        entry_price: float,
        exit_price: float,
        side: str,
        size: float,
        fees: float = 0.0,
    ) -> None:
        """Record a completed trade.

        Args:
            pnl: Realized P&L for this trade
            entry_price: Entry price
            exit_price: Exit price
            side: Trade side ('long' or 'short')
            size: Position size
            fees: Trading fees paid
        """
        with self._lock:
            stats = self._trading_stats
            now = datetime.now()

            # Update counters
            stats.total_trades += 1
            if pnl > 0:
                stats.winning_trades += 1
                stats.largest_win = max(stats.largest_win, pnl)
            elif pnl < 0:
                stats.losing_trades += 1
                stats.largest_loss = min(stats.largest_loss, pnl)

            # Update P&L
            stats.realized_pnl += pnl
            stats.total_fees += fees
            stats.last_trade_time = now

            # Add to trade history
            if stats.trade_history is None:
                stats.trade_history = []

            trade_record = {
                "timestamp": now.isoformat(),
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "fees": fees,
            }
            stats.trade_history.append(trade_record)

            # Keep only last N trades
            if len(stats.trade_history) > self._max_trade_history:
                stats.trade_history = stats.trade_history[-self._max_trade_history :]

    def update_equity(self, current_equity: float, unrealized_pnl: float = 0.0) -> None:
        """Update current account equity and unrealized P&L.

        Args:
            current_equity: Current total equity (balance + unrealized P&L)
            unrealized_pnl: Current unrealized P&L from open positions
        """
        with self._lock:
            self._trading_stats.current_equity = current_equity
            self._trading_stats.unrealized_pnl = unrealized_pnl

    @property
    def trading_stats(self) -> TradingStats:
        """Get current trading statistics."""
        with self._lock:
            return self._trading_stats

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
                trading_stats=self._trading_stats.to_dict(),
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
