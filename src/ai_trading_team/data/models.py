"""Data models for market data."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class Ticker:
    """Real-time ticker data."""

    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    high_24h: Decimal
    low_24h: Decimal
    volume_24h: Decimal
    price_change_percent: Decimal
    mark_price: Decimal | None = None
    index_price: Decimal | None = None
    funding_rate: Decimal | None = None
    timestamp: datetime | None = None


@dataclass
class Kline:
    """Candlestick/K-line data."""

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime | None = None
    quote_volume: Decimal | None = None
    trades: int | None = None


@dataclass
class OrderBookLevel:
    """Single level in order book."""

    price: Decimal
    quantity: Decimal


@dataclass
class OrderBook:
    """Order book snapshot."""

    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime | None = None
    last_update_id: int | None = None


@dataclass
class Trade:
    """Trade/transaction data."""

    trade_id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    is_buyer_maker: bool
    timestamp: datetime


@dataclass
class FundingRate:
    """Funding rate data."""

    symbol: str
    funding_rate: Decimal
    funding_time: datetime
    mark_price: Decimal | None = None


@dataclass
class OpenInterest:
    """Open interest data."""

    symbol: str
    open_interest: Decimal
    timestamp: datetime


@dataclass
class LongShortRatio:
    """Long/short ratio data."""

    symbol: str
    long_ratio: Decimal
    short_ratio: Decimal
    long_short_ratio: Decimal
    timestamp: datetime


@dataclass
class Liquidation:
    """Liquidation event data."""

    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    timestamp: datetime
