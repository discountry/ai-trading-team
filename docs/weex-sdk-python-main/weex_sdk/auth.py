"""Authentication and signature generation for Weex API."""

import base64
import hashlib
import hmac
import time


class SignatureGenerator:
    """Generate signatures for Weex API requests."""

    @staticmethod
    def generate_signature(
        secret_key: str,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> str:
        """Generate HMAC SHA256 signature for HTTP requests.

        Args:
            secret_key: API secret key
            timestamp: Request timestamp (milliseconds)
            method: HTTP method (GET, POST)
            request_path: API endpoint path
            query_string: Query string (without '?')
            body: Request body (JSON string)

        Returns:
            Base64 encoded signature
        """
        # Build message string
        if query_string:
            message = f"{timestamp}{method.upper()}{request_path}?{query_string}{body}"
        else:
            message = f"{timestamp}{method.upper()}{request_path}{body}"

        # Generate HMAC SHA256 signature
        signature = hmac.new(
            secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        # Base64 encode
        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def generate_websocket_signature(
        secret_key: str,
        timestamp: str,
        request_path: str = "/v2/ws/private",
    ) -> str:
        """Generate signature for WebSocket private channel connection.

        Args:
            secret_key: API secret key
            timestamp: Request timestamp (milliseconds)
            request_path: WebSocket path (default: /v2/ws/private)

        Returns:
            Base64 encoded signature
        """
        message = f"{timestamp}{request_path}"

        # Generate HMAC SHA256 signature
        signature = hmac.new(
            secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        # Base64 encode
        return base64.b64encode(signature).decode("utf-8")


class RequestHeaders:
    """Build request headers for authenticated API requests."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        locale: str = "en-US",
    ) -> None:
        """Initialize request headers builder.

        Args:
            api_key: API key
            secret_key: Secret key for signature generation
            passphrase: API passphrase
            locale: Locale setting (default: en-US)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.locale = locale

    def get_headers(
        self,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> dict[str, str]:
        """Generate authenticated request headers.

        Args:
            method: HTTP method (GET, POST)
            request_path: API endpoint path
            query_string: Query string (without '?')
            body: Request body (JSON string)

        Returns:
            Dictionary of request headers
        """
        timestamp = str(int(time.time() * 1000))

        signature = SignatureGenerator.generate_signature(
            secret_key=self.secret_key,
            timestamp=timestamp,
            method=method,
            request_path=request_path,
            query_string=query_string,
            body=body,
        )

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": self.locale,
        }

    def get_websocket_headers(self) -> dict[str, str]:
        """Generate headers for WebSocket private channel connection.

        Returns:
            Dictionary of WebSocket headers
        """
        timestamp = str(int(time.time() * 1000))

        signature = SignatureGenerator.generate_websocket_signature(
            secret_key=self.secret_key,
            timestamp=timestamp,
        )

        return {
            "User-Agent": "weex-sdk-python",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
        }
