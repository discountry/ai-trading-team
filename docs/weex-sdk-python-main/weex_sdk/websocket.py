"""WebSocket client for Weex API (sync and async)."""

import asyncio
import json
import threading
import time
from typing import Any, Callable, Dict, Optional

import websocket
import websockets

from weex_sdk.auth import RequestHeaders
from weex_sdk.exceptions import WeexNetworkError, WeexWebSocketError
from weex_sdk.logger import get_logger
from weex_sdk.models import WebSocketSubscription

logger = get_logger("websocket")

# WebSocket URLs
WS_PUBLIC_URL = "wss://ws-contract.weex.com/v2/ws/public"
WS_PRIVATE_URL = "wss://ws-contract.weex.com/v2/ws/private"


class WeexWebSocket:
    """Synchronous WebSocket client for Weex API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        is_private: bool = False,
        reconnect_attempts: int = 10,
        reconnect_delay: int = 1,
        max_reconnect_delay: int = 60,
    ) -> None:
        """Initialize WebSocket client.

        Args:
            api_key: API key (required for private channels)
            secret_key: Secret key (required for private channels)
            passphrase: API passphrase (required for private channels)
            is_private: Whether to use private channel (default: False)
            reconnect_attempts: Maximum reconnection attempts (default: 10)
            reconnect_delay: Initial reconnection delay in seconds (default: 1)
            max_reconnect_delay: Maximum reconnection delay in seconds (default: 60)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_private = is_private
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self.ws: Optional[websocket.WebSocketApp] = None
        self.subscriptions: Dict[str, WebSocketSubscription] = {}
        self.current_reconnect_attempts = 0
        self.connected = False
        self.should_reconnect = True
        self._lock = threading.Lock()

        if is_private and (not api_key or not secret_key or not passphrase):
            raise ValueError("API credentials required for private channels")

    def _get_url(self) -> str:
        """Get WebSocket URL based on channel type.

        Returns:
            WebSocket URL
        """
        return WS_PRIVATE_URL if self.is_private else WS_PUBLIC_URL

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """Get WebSocket headers for private channels.

        Returns:
            Headers dictionary or None for public channels
        """
        if not self.is_private:
            return {"User-Agent": "weex-sdk-python"}

        headers_builder = RequestHeaders(
            api_key=self.api_key or "",
            secret_key=self.secret_key or "",
            passphrase=self.passphrase or "",
        )
        return headers_builder.get_websocket_headers()

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages.

        Args:
            ws: WebSocket connection
            message: Received message
        """
        try:
            data = json.loads(message)
            event = data.get("event")

            # Handle ping/pong
            if event == "ping":
                self._handle_ping(data)
                return

            # Handle subscription confirmations
            if event == "subscribe" or event == "unsubscribe":
                logger.info(f"Subscription event: {event}, channel: {data.get('channel')}")
                return

            # Handle payload messages
            if event == "payload":
                channel = data.get("channel")
                if channel and channel in self.subscriptions:
                    subscription = self.subscriptions[channel]
                    try:
                        subscription.callback(data)
                    except Exception as e:
                        logger.error(f"Error in callback for channel {channel}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def _handle_ping(self, message: Dict[str, Any]) -> None:
        """Handle ping message and send pong response.

        Args:
            message: Ping message
        """
        pong = {
            "event": "pong",
            "time": message.get("time"),
        }
        self.send(json.dumps(pong))
        logger.debug("Sent pong response")

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket errors.

        Args:
            ws: WebSocket connection
            error: Error object
        """
        logger.error(f"WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket close.

        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False

        # Attempt reconnection if enabled
        if self.should_reconnect and self.current_reconnect_attempts < self.reconnect_attempts:
            self._reconnect()

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket open.

        Args:
            ws: WebSocket connection
        """
        logger.info("WebSocket connected")
        self.connected = True
        self.current_reconnect_attempts = 0

        # Resubscribe to all channels
        self._resubscribe()

    def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self.current_reconnect_attempts >= self.reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self.current_reconnect_attempts += 1
        delay = min(
            self.reconnect_delay * (2 ** (self.current_reconnect_attempts - 1)),
            self.max_reconnect_delay,
        )

        logger.info(f"Reconnecting in {delay} seconds (attempt {self.current_reconnect_attempts})")
        time.sleep(delay)

        try:
            self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    def _resubscribe(self) -> None:
        """Resubscribe to all previously subscribed channels."""
        if not self.subscriptions:
            return

        logger.info(f"Resubscribing to {len(self.subscriptions)} channels")
        for channel, subscription in self.subscriptions.items():
            try:
                self._send_subscribe(channel, subscription.params)
            except Exception as e:
                logger.error(f"Failed to resubscribe to {channel}: {e}")

    def _send_subscribe(self, channel: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send subscription message.

        Args:
            channel: Channel name
            params: Optional subscription parameters
        """
        message = {
            "event": "subscribe",
            "channel": channel,
        }
        if params:
            message.update(params)

        self.send(json.dumps(message))

    def _send_unsubscribe(self, channel: str) -> None:
        """Send unsubscription message.

        Args:
            channel: Channel name
        """
        message = {
            "event": "unsubscribe",
            "channel": channel,
        }
        self.send(json.dumps(message))

    def connect(self) -> None:
        """Connect to WebSocket server."""
        url = self._get_url()
        headers = self._get_headers()

        logger.info(f"Connecting to WebSocket: {url}")

        self.ws = websocket.WebSocketApp(
            url,
            header=headers if headers else None,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

        # Run in a separate thread
        self.ws.run_forever()

    def send(self, message: str) -> None:
        """Send message through WebSocket.

        Args:
            message: Message to send

        Raises:
            WeexWebSocketError: If not connected
        """
        if not self.ws or not self.connected:
            raise WeexWebSocketError("WebSocket not connected")

        try:
            self.ws.send(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise WeexWebSocketError(f"Failed to send message: {str(e)}") from e

    def subscribe_ticker(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to ticker channel.

        Args:
            symbol: Trading pair symbol
            callback: Callback function for ticker updates
        """
        channel = f"ticker.{symbol}"
        self.subscriptions[channel] = WebSocketSubscription(
            channel=channel,
            callback=callback,
        )
        self._send_subscribe(channel)

    def subscribe_kline(
        self,
        symbol: str,
        granularity: str,
        price_type: str = "LAST_PRICE",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Subscribe to K-line channel.

        Args:
            symbol: Trading pair symbol
            granularity: K-line interval (MINUTE_1, MINUTE_5, etc.)
            price_type: Price type (LAST_PRICE, MARK, INDEX)
            callback: Callback function for K-line updates
        """
        channel = f"kline.{price_type}.{symbol}.{granularity}"
        if callback:
            self.subscriptions[channel] = WebSocketSubscription(
                channel=channel,
                callback=callback,
            )
        self._send_subscribe(channel)

    def subscribe_depth(
        self,
        symbol: str,
        limit: int = 15,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Subscribe to depth channel.

        Args:
            symbol: Trading pair symbol
            limit: Depth limit (15 or 200)
            callback: Callback function for depth updates
        """
        channel = f"depth.{symbol}.{limit}"
        if callback:
            self.subscriptions[channel] = WebSocketSubscription(
                channel=channel,
                callback=callback,
            )
        self._send_subscribe(channel)

    def subscribe_trades(
        self,
        symbol: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Subscribe to trades channel.

        Args:
            symbol: Trading pair symbol
            callback: Callback function for trade updates
        """
        channel = f"trades.{symbol}"
        if callback:
            self.subscriptions[channel] = WebSocketSubscription(
                channel=channel,
                callback=callback,
            )
        self._send_subscribe(channel)

    def subscribe_account(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to account channel (private).

        Args:
            callback: Callback function for account updates

        Raises:
            ValueError: If not using private channel
        """
        if not self.is_private:
            raise ValueError("Account channel requires private WebSocket connection")

        channel = "account"
        self.subscriptions[channel] = WebSocketSubscription(
            channel=channel,
            callback=callback,
        )
        self._send_subscribe(channel)

    def subscribe_position(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to position channel (private).

        Args:
            callback: Callback function for position updates

        Raises:
            ValueError: If not using private channel
        """
        if not self.is_private:
            raise ValueError("Position channel requires private WebSocket connection")

        channel = "position"
        self.subscriptions[channel] = WebSocketSubscription(
            channel=channel,
            callback=callback,
        )
        self._send_subscribe(channel)

    def subscribe_order(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to order channel (private).

        Args:
            callback: Callback function for order updates

        Raises:
            ValueError: If not using private channel
        """
        if not self.is_private:
            raise ValueError("Order channel requires private WebSocket connection")

        channel = "orders"
        self.subscriptions[channel] = WebSocketSubscription(
            channel=channel,
            callback=callback,
        )
        self._send_subscribe(channel)

    def subscribe_trade(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to trade channel (private).

        Args:
            callback: Callback function for trade updates

        Raises:
            ValueError: If not using private channel
        """
        if not self.is_private:
            raise ValueError("Trade channel requires private WebSocket connection")

        channel = "trade"
        self.subscriptions[channel] = WebSocketSubscription(
            channel=channel,
            callback=callback,
        )
        self._send_subscribe(channel)

    def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel.

        Args:
            channel: Channel name
        """
        if channel in self.subscriptions:
            self._send_unsubscribe(channel)
            del self.subscriptions[channel]

    def close(self) -> None:
        """Close WebSocket connection."""
        self.should_reconnect = False
        if self.ws:
            self.ws.close()
        self.connected = False


class AsyncWeexWebSocket:
    """Asynchronous WebSocket client for Weex API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        is_private: bool = False,
        reconnect_attempts: int = 10,
        reconnect_delay: int = 1,
        max_reconnect_delay: int = 60,
    ) -> None:
        """Initialize async WebSocket client.

        Args:
            api_key: API key (required for private channels)
            secret_key: Secret key (required for private channels)
            passphrase: API passphrase (required for private channels)
            is_private: Whether to use private channel
            reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_private = is_private
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.subscriptions: Dict[str, WebSocketSubscription] = {}
        self.current_reconnect_attempts = 0
        self.connected = False
        self.should_reconnect = True
        self._receive_task: Optional[asyncio.Task] = None

        if is_private and (not api_key or not secret_key or not passphrase):
            raise ValueError("API credentials required for private channels")

    def _get_url(self) -> str:
        """Get WebSocket URL."""
        return WS_PRIVATE_URL if self.is_private else WS_PUBLIC_URL

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """Get WebSocket headers."""
        if not self.is_private:
            return {"User-Agent": "weex-sdk-python"}

        headers_builder = RequestHeaders(
            api_key=self.api_key or "",
            secret_key=self.secret_key or "",
            passphrase=self.passphrase or "",
        )
        return headers_builder.get_websocket_headers()

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            event = data.get("event")

            if event == "ping":
                await self._handle_ping(data)
                return

            if event == "subscribe" or event == "unsubscribe":
                logger.info(f"Subscription event: {event}, channel: {data.get('channel')}")
                return

            if event == "payload":
                channel = data.get("channel")
                if channel and channel in self.subscriptions:
                    subscription = self.subscriptions[channel]
                    try:
                        if asyncio.iscoroutinefunction(subscription.callback):
                            await subscription.callback(data)
                        else:
                            subscription.callback(data)
                    except Exception as e:
                        logger.error(f"Error in callback for channel {channel}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _handle_ping(self, message: Dict[str, Any]) -> None:
        """Handle ping and send pong."""
        pong = {
            "event": "pong",
            "time": message.get("time"),
        }
        await self.send(json.dumps(pong))
        logger.debug("Sent pong response")

    async def _receive_loop(self) -> None:
        """Receive messages loop."""
        try:
            while self.connected and self.ws:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    await self._handle_message(message)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    ping = {"event": "ping", "time": str(int(time.time() * 1000))}
                    await self.send(json.dumps(ping))
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
        finally:
            self.connected = False

    async def _reconnect(self) -> None:
        """Attempt reconnection with exponential backoff."""
        if self.current_reconnect_attempts >= self.reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self.current_reconnect_attempts += 1
        delay = min(
            self.reconnect_delay * (2 ** (self.current_reconnect_attempts - 1)),
            self.max_reconnect_delay,
        )

        logger.info(f"Reconnecting in {delay} seconds (attempt {self.current_reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def _resubscribe(self) -> None:
        """Resubscribe to all channels."""
        if not self.subscriptions:
            return

        logger.info(f"Resubscribing to {len(self.subscriptions)} channels")
        for channel, subscription in self.subscriptions.items():
            try:
                await self._send_subscribe(channel, subscription.params)
            except Exception as e:
                logger.error(f"Failed to resubscribe to {channel}: {e}")

    async def _send_subscribe(self, channel: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send subscription message."""
        message = {
            "event": "subscribe",
            "channel": channel,
        }
        if params:
            message.update(params)
        await self.send(json.dumps(message))

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        url = self._get_url()
        headers = self._get_headers()

        logger.info(f"Connecting to WebSocket: {url}")

        try:
            self.ws = await websockets.connect(
                url,
                extra_headers=headers if headers else None,
            )
            self.connected = True
            self.current_reconnect_attempts = 0

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Resubscribe
            await self._resubscribe()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            if self.should_reconnect:
                await self._reconnect()
            raise WeexNetworkError(f"WebSocket connection failed: {str(e)}") from e

    async def send(self, message: str) -> None:
        """Send message through WebSocket."""
        if not self.ws or not self.connected:
            raise WeexWebSocketError("WebSocket not connected")

        try:
            await self.ws.send(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise WeexWebSocketError(f"Failed to send message: {str(e)}") from e

    async def subscribe_ticker(
        self, symbol: str, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to ticker channel."""
        channel = f"ticker.{symbol}"
        self.subscriptions[channel] = WebSocketSubscription(channel=channel, callback=callback)
        await self._send_subscribe(channel)

    async def subscribe_account(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to account channel (private)."""
        if not self.is_private:
            raise ValueError("Account channel requires private WebSocket connection")
        channel = "account"
        self.subscriptions[channel] = WebSocketSubscription(channel=channel, callback=callback)
        await self._send_subscribe(channel)

    async def subscribe_position(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to position channel (private)."""
        if not self.is_private:
            raise ValueError("Position channel requires private WebSocket connection")
        channel = "position"
        self.subscriptions[channel] = WebSocketSubscription(channel=channel, callback=callback)
        await self._send_subscribe(channel)

    async def subscribe_order(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to order channel (private)."""
        if not self.is_private:
            raise ValueError("Order channel requires private WebSocket connection")
        channel = "orders"
        self.subscriptions[channel] = WebSocketSubscription(channel=channel, callback=callback)
        await self._send_subscribe(channel)

    async def close(self) -> None:
        """Close WebSocket connection."""
        self.should_reconnect = False
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
        if self.ws:
            await self.ws.close()
