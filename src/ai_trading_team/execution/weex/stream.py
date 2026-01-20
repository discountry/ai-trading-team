"""WEEX WebSocket stream for account updates."""

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from weex_sdk import AsyncWeexWebSocket

logger = logging.getLogger(__name__)


def _patch_websockets_headers() -> None:
    try:
        import websockets
    except ImportError:
        return

    if getattr(websockets.connect, "_weex_headers_patched", False):
        return

    try:
        sig = inspect.signature(websockets.connect)
        params = sig.parameters
    except (TypeError, ValueError):
        return

    original_connect = websockets.connect

    def connect_wrapper(*args: Any, **kwargs: Any) -> Any:
        if "additional_headers" in params:
            if "additional_headers" not in kwargs:
                if "extra_headers" in kwargs:
                    kwargs["additional_headers"] = kwargs.pop("extra_headers")
                elif "headers" in kwargs:
                    kwargs["additional_headers"] = kwargs.pop("headers")
            kwargs.pop("extra_headers", None)
            kwargs.pop("headers", None)
        elif "extra_headers" in params:
            if "extra_headers" not in kwargs and "headers" in kwargs:
                kwargs["extra_headers"] = kwargs.pop("headers")
            kwargs.pop("additional_headers", None)
        else:
            kwargs.pop("extra_headers", None)
            kwargs.pop("headers", None)
            kwargs.pop("additional_headers", None)
        return original_connect(*args, **kwargs)

    connect_wrapper._weex_headers_patched = True  # type: ignore[attr-defined]
    websockets.connect = connect_wrapper


class WEEXStream(ABC):
    """Abstract WEEX WebSocket stream.

    Handles real-time account, position, and order updates.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        ...

    @abstractmethod
    async def subscribe_account(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to account updates."""
        ...

    @abstractmethod
    async def subscribe_positions(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to position updates."""
        ...

    @abstractmethod
    async def subscribe_orders(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to order updates."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        ...


class WEEXPrivateStream(WEEXStream):
    """Private WEEX WebSocket stream for account, position, and order updates."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
    ) -> None:
        self._ws = AsyncWeexWebSocket(
            api_key=api_key,
            secret_key=api_secret,
            passphrase=passphrase,
            is_private=True,
        )

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        _patch_websockets_headers()
        await self._ws.connect()
        logger.info("WEEX private WebSocket connected")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        await self._ws.close()
        logger.info("WEEX private WebSocket disconnected")

    async def subscribe_account(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to account updates."""
        await self._ws.subscribe_account(callback)

    async def subscribe_positions(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to position updates."""
        await self._ws.subscribe_position(callback)

    async def subscribe_orders(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to order updates."""
        await self._ws.subscribe_order(callback)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return bool(self._ws.connected)
