"""Weex SDK - A comprehensive Python SDK for Weex exchange API."""

from weex_sdk.api.account import AccountAPI, AsyncAccountAPI
from weex_sdk.api.ai import AIAPI, AsyncAIAPI
from weex_sdk.api.market import AsyncMarketAPI, MarketAPI
from weex_sdk.api.trade import AsyncTradeAPI, TradeAPI
from weex_sdk.client import AsyncWeexClient, WeexClient
from weex_sdk.exceptions import (
    WeexAPIError,
    WeexAuthenticationError,
    WeexNetworkError,
    WeexRateLimitError,
    WeexValidationError,
    WeexWebSocketError,
)
from weex_sdk.websocket import AsyncWeexWebSocket, WeexWebSocket

__version__ = "1.0.10"

__all__ = [
    # Clients
    "WeexClient",
    "AsyncWeexClient",
    "WeexWebSocket",
    "AsyncWeexWebSocket",
    # Exceptions
    "WeexAPIError",
    "WeexAuthenticationError",
    "WeexRateLimitError",
    "WeexNetworkError",
    "WeexWebSocketError",
    "WeexValidationError",
    # API modules
    "AccountAPI",
    "AsyncAccountAPI",
    "MarketAPI",
    "AsyncMarketAPI",
    "TradeAPI",
    "AsyncTradeAPI",
    "AIAPI",
    "AsyncAIAPI",
]
