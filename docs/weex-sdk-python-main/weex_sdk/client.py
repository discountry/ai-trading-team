"""HTTP client for Weex API (sync and async)."""

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp
import requests

from weex_sdk.auth import RequestHeaders
from weex_sdk.exceptions import (
    WeexNetworkError,
    WeexRateLimitError,
    raise_exception_from_response,
)
from weex_sdk.logger import get_logger
from weex_sdk.utils.helpers import sanitize_log_data

if TYPE_CHECKING:
    from weex_sdk.api.account import AccountAPI, AsyncAccountAPI
    from weex_sdk.api.ai import AIAPI, AsyncAIAPI
    from weex_sdk.api.market import AsyncMarketAPI, MarketAPI
    from weex_sdk.api.trade import AsyncTradeAPI, TradeAPI

logger = get_logger("client")

# Base URL for Weex API
BASE_URL = "https://api-contract.weex.com"


class BaseClient:
    """Base client with common functionality."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: str = BASE_URL,
        locale: str = "en-US",
        timeout: int = 30,
    ) -> None:
        """Initialize base client.

        Args:
            api_key: API key
            secret_key: Secret key
            passphrase: API passphrase
            base_url: Base URL for API (default: production)
            locale: Locale setting (default: en-US)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = base_url.rstrip("/")
        self.locale = locale
        self.timeout = timeout
        self.headers_builder = RequestHeaders(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            locale=locale,
        )

    def _handle_response(self, response: Any) -> Dict[str, Any]:
        """Handle API response and raise exceptions on errors.

        Args:
            response: Response object (requests.Response or aiohttp.ClientResponse)

        Returns:
            Parsed JSON response

        Raises:
            WeexAPIException: On API errors
            WeexNetworkError: On network errors
        """
        # Extract status code and content
        if hasattr(response, "status"):
            status_code = response.status
        else:
            status_code = response.status_code

        try:
            if hasattr(response, "json"):
                data = response.json()
            else:
                data = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError):
            data = {"code": str(status_code), "msg": response.text or "Unknown error"}

        # Check for errors
        if status_code != 200:
            code = data.get("code", str(status_code))
            message = data.get("msg", f"HTTP {status_code} error")
            request_time = data.get("requestTime")

            logger.error(
                f"API error: [{code}] {message}",
                extra={"status_code": status_code, "response": sanitize_log_data(data)},
            )

            # Handle rate limiting
            if status_code == 429:
                retry_after = None
                if hasattr(response, "headers"):
                    retry_after_header = response.headers.get("Retry-After")
                    if retry_after_header:
                        try:
                            retry_after = int(retry_after_header)
                        except ValueError:
                            pass

                raise WeexRateLimitError(
                    message=message,
                    code=code,
                    request_time=request_time,
                    retry_after=retry_after,
                )

            # Raise appropriate exception
            raise_exception_from_response(code, message, request_time)

        return data


class WeexClient(BaseClient):
    """Synchronous HTTP client for Weex API."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: str = BASE_URL,
        locale: str = "en-US",
        timeout: int = 30,
    ) -> None:
        """Initialize synchronous client.

        Args:
            api_key: API key
            secret_key: Secret key
            passphrase: API passphrase
            base_url: Base URL for API
            locale: Locale setting
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, secret_key, passphrase, base_url, locale, timeout)
        self.session = requests.Session()

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make GET request.

        Args:
            path: API endpoint path
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            WeexAPIError: On API errors
            WeexNetworkError: On network errors
        """
        url = f"{self.base_url}{path}"
        query_string = ""
        if params:
            query_parts = []
            for key, value in params.items():
                if value is not None:
                    if isinstance(value, list):
                        for item in value:
                            query_parts.append(f"{key}={item}")
                    else:
                        query_parts.append(f"{key}={value}")
            query_string = "&".join(query_parts)

        headers = self.headers_builder.get_headers(
            method="GET",
            request_path=path,
            query_string=query_string,
        )

        logger.debug(
            f"GET {path}",
            extra={"params": params, "headers": sanitize_log_data(headers)},
        )

        try:
            response = self.session.get(
                url,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise WeexNetworkError(f"Network error: {str(e)}") from e

    def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make POST request.

        Args:
            path: API endpoint path
            data: Request body data

        Returns:
            Parsed JSON response

        Raises:
            WeexAPIError: On API errors
            WeexNetworkError: On network errors
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(data) if data else ""

        headers = self.headers_builder.get_headers(
            method="POST",
            request_path=path,
            query_string="",
            body=body,
        )

        logger.debug(
            f"POST {path}",
            extra={"data": sanitize_log_data(data or {}), "headers": sanitize_log_data(headers)},
        )

        try:
            response = self.session.post(
                url,
                headers=headers,
                data=body,
                timeout=self.timeout,
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise WeexNetworkError(f"Network error: {str(e)}") from e

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    # API modules
    @property
    def account(self) -> "AccountAPI":
        """Get Account API instance."""
        from weex_sdk.api.account import AccountAPI

        return AccountAPI(self)

    @property
    def market(self) -> "MarketAPI":
        """Get Market API instance."""
        from weex_sdk.api.market import MarketAPI

        return MarketAPI(self)

    @property
    def trade(self) -> "TradeAPI":
        """Get Trade API instance."""
        from weex_sdk.api.trade import TradeAPI

        return TradeAPI(self)

    @property
    def ai(self) -> "AIAPI":
        """Get AI API instance."""
        from weex_sdk.api.ai import AIAPI

        return AIAPI(self)


class AsyncWeexClient(BaseClient):
    """Asynchronous HTTP client for Weex API."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: str = BASE_URL,
        locale: str = "en-US",
        timeout: int = 30,
    ) -> None:
        """Initialize asynchronous client.

        Args:
            api_key: API key
            secret_key: Secret key
            passphrase: API passphrase
            base_url: Base URL for API
            locale: Locale setting
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, secret_key, passphrase, base_url, locale, timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncWeexClient":
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make async GET request.

        Args:
            path: API endpoint path
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            WeexAPIError: On API errors
            WeexNetworkError: On network errors
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}{path}"
        query_string = ""
        if params:
            query_parts = []
            for key, value in params.items():
                if value is not None:
                    if isinstance(value, list):
                        for item in value:
                            query_parts.append(f"{key}={item}")
                    else:
                        query_parts.append(f"{key}={value}")
            query_string = "&".join(query_parts)

        headers = self.headers_builder.get_headers(
            method="GET",
            request_path=path,
            query_string=query_string,
        )

        logger.debug(
            f"GET {path}",
            extra={"params": params, "headers": sanitize_log_data(headers)},
        )

        try:
            async with self.session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                # Read response text
                text = await response.text()

                # Create a mock response object for _handle_response
                class MockResponse:
                    def __init__(self, status: int, text: str, headers: Any) -> None:
                        self.status = status
                        self.text = text
                        self.headers = headers

                    def json(self) -> Dict[str, Any]:
                        return json.loads(self.text)

                mock_response = MockResponse(response.status, text, response.headers)
                return self._handle_response(mock_response)
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise WeexNetworkError(f"Network error: {str(e)}") from e

    async def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make async POST request.

        Args:
            path: API endpoint path
            data: Request body data

        Returns:
            Parsed JSON response

        Raises:
            WeexAPIError: On API errors
            WeexNetworkError: On network errors
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}{path}"
        body = json.dumps(data) if data else ""

        headers = self.headers_builder.get_headers(
            method="POST",
            request_path=path,
            query_string="",
            body=body,
        )

        logger.debug(
            f"POST {path}",
            extra={"data": sanitize_log_data(data or {}), "headers": sanitize_log_data(headers)},
        )

        try:
            async with self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                text = await response.text()

                class MockResponse:
                    def __init__(self, status: int, text: str, headers: Any) -> None:
                        self.status = status
                        self.text = text
                        self.headers = headers

                    def json(self) -> Dict[str, Any]:
                        return json.loads(self.text)

                mock_response = MockResponse(response.status, text, response.headers)
                return self._handle_response(mock_response)
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise WeexNetworkError(f"Network error: {str(e)}") from e

    async def close(self) -> None:
        """Close the session."""
        if self.session:
            await self.session.close()

    # API modules
    @property
    def account(self) -> "AsyncAccountAPI":
        """Get Async Account API instance."""
        from weex_sdk.api.account import AsyncAccountAPI

        return AsyncAccountAPI(self)

    @property
    def market(self) -> "AsyncMarketAPI":
        """Get Async Market API instance."""
        from weex_sdk.api.market import AsyncMarketAPI

        return AsyncMarketAPI(self)

    @property
    def trade(self) -> "AsyncTradeAPI":
        """Get Async Trade API instance."""
        from weex_sdk.api.trade import AsyncTradeAPI

        return AsyncTradeAPI(self)

    @property
    def ai(self) -> "AsyncAIAPI":
        """Get Async AI API instance."""
        from weex_sdk.api.ai import AsyncAIAPI

        return AsyncAIAPI(self)
