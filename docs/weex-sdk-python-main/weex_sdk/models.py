"""Data models and type definitions for Weex SDK."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from typing_extensions import TypedDict


# Account Models
class AccountInfo(TypedDict, total=False):
    """Account information model."""

    defaultFeeSetting: Dict[str, Any]
    feeSetting: List[Dict[str, Any]]
    modeSetting: List[Dict[str, Any]]
    leverageSetting: List[Dict[str, Any]]
    createOrderRateLimitPerMinute: int
    createOrderDelayMilliseconds: int
    createdTime: str
    updatedTime: str


class Collateral(TypedDict, total=False):
    """Collateral information model."""

    coin: str
    marginMode: str
    crossSymbol: Optional[str]
    isolatedPositionId: str
    amount: str
    pending_deposit_amount: str
    pending_withdraw_amount: str
    pending_transfer_in_amount: str
    pending_transfer_out_amount: str
    is_liquidating: bool
    legacy_amount: str
    cum_deposit_amount: str
    cum_withdraw_amount: str
    cum_transfer_in_amount: str
    cum_transfer_out_amount: str
    cum_margin_move_in_amount: str
    cum_margin_move_out_amount: str
    cum_position_open_long_amount: str
    cum_position_open_short_amount: str
    cum_position_close_long_amount: str
    cum_position_close_short_amount: str
    cum_position_fill_fee_amount: str
    cum_position_liquidate_fee_amount: str
    cum_position_funding_amount: str
    cum_order_fill_fee_income_amount: str
    cum_order_liquidate_fee_income_amount: str
    created_time: str
    updated_time: str


class Asset(TypedDict):
    """Asset information model."""

    coinName: str
    available: str
    frozen: str
    equity: str
    unrealizePnl: str


class Position(TypedDict, total=False):
    """Position information model."""

    id: str
    account_id: str
    coin_id: int
    contract_id: str
    symbol: str
    side: str
    margin_mode: str
    separated_mode: str
    separated_open_order_id: str
    leverage: str
    size: str
    open_value: str
    open_fee: str
    funding_fee: str
    marginSize: str
    isolated_margin: str
    is_auto_append_isolated_margin: bool
    cum_open_size: str
    cum_open_value: str
    cum_open_fee: str
    cum_close_size: str
    cum_close_value: str
    cum_close_fee: str
    cum_funding_fee: str
    cum_liquidate_fee: str
    created_match_sequence_id: str
    updated_match_sequence_id: str
    created_time: str
    updated_time: str
    contractVal: str
    unrealizePnl: str
    liquidatePrice: str


# Market Models
class Ticker(TypedDict, total=False):
    """Ticker information model."""

    symbol: str
    last: str
    best_ask: str
    best_bid: str
    high_24h: str
    low_24h: str
    volume_24h: str
    timestamp: str
    priceChangePercent: str
    base_volume: str
    markPrice: str
    indexPrice: str


class Contract(TypedDict, total=False):
    """Contract information model."""

    symbol: str
    underlying_index: str
    quote_currency: str
    coin: str
    contract_val: str
    delivery: List[str]
    size_increment: str
    tick_size: str
    forwardContractFlag: bool
    priceEndStep: int
    minLeverage: int
    maxLeverage: int
    buyLimitPriceRatio: str
    sellLimitPriceRatio: str
    makerFeeRate: str
    takerFeeRate: str
    minOrderSize: str
    maxOrderSize: str
    maxPositionSize: str
    marketOpenLimitSize: str


class DepthLevel(TypedDict):
    """Depth level model."""

    price: str
    size: str


class Depth(TypedDict):
    """Market depth model."""

    asks: List[List[str]]
    bids: List[List[str]]
    timestamp: str


class Trade(TypedDict, total=False):
    """Trade information model."""

    ticketId: str
    time: int
    price: str
    size: str
    value: str
    symbol: str
    isBestMatch: bool
    isBuyerMaker: bool
    contractVal: str


class Candle(TypedDict):
    """Candle/K-line data model."""

    # Array format: [timestamp, open, high, low, close, base_volume, quote_volume]
    pass  # Represented as List[str] in practice


# Order Models
class Order(TypedDict, total=False):
    """Order information model."""

    symbol: str
    size: str
    client_oid: Optional[str]
    createTime: str
    filled_qty: str
    fee: str
    order_id: str
    price: str
    price_avg: str
    status: str
    type: str
    order_type: str
    totalProfits: str
    contracts: int
    filledQtyContracts: int
    presetTakeProfitPrice: Optional[str]
    presetStopLossPrice: Optional[str]


class OrderFill(TypedDict, total=False):
    """Order fill/trade detail model."""

    tradeId: int
    orderId: int
    symbol: str
    marginMode: str
    separatedMode: str
    positionSide: str
    orderSide: str
    fillSize: str
    fillValue: str
    fillFee: str
    liquidateFee: str
    realizePnl: str
    direction: str
    liquidateType: str
    legacyOrdeDirection: str
    createdTime: int


# WebSocket Models
class WebSocketMessage(TypedDict, total=False):
    """WebSocket message model."""

    event: str
    channel: str
    type: Optional[str]
    data: Union[List[Any], Dict[str, Any]]
    msg: Optional[Dict[str, Any]]
    time: Optional[int]


@dataclass
class WebSocketSubscription:
    """WebSocket subscription information."""

    channel: str
    callback: Callable[[Dict[str, Any]], None]
    params: Optional[Dict[str, Any]] = None
