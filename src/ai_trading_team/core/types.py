"""Global type definitions."""

from enum import Enum, auto


class Side(str, Enum):
    """Position/order side."""

    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class TimeInForce(str, Enum):
    """Time in force for orders."""

    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing (Post Only)


class OrderStatus(str, Enum):
    """Order status."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(str, Enum):
    """Position status."""

    OPEN = "open"
    CLOSED = "closed"


class EventType(str, Enum):
    """Event types for inter-module communication."""

    # Data events
    TICKER_UPDATE = auto()
    KLINE_UPDATE = auto()
    ORDERBOOK_UPDATE = auto()
    TRADE_UPDATE = auto()
    FUNDING_RATE_UPDATE = auto()
    MARK_PRICE_UPDATE = auto()
    LIQUIDATION_UPDATE = auto()

    # Indicator events
    INDICATOR_UPDATE = auto()

    # Strategy events
    SIGNAL_GENERATED = auto()

    # Execution events
    ORDER_CREATED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELED = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    # Risk events
    RISK_TRIGGERED = auto()

    # Agent events
    AGENT_DECISION = auto()
