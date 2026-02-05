"""Custom exceptions for Weex SDK."""

from typing import Optional


class WeexAPIError(Exception):
    """Base exception for all Weex API errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        request_time: Optional[int] = None,
    ) -> None:
        """Initialize Weex API exception.

        Args:
            message: Error message
            code: Error code from API response
            request_time: Request timestamp
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.request_time = request_time

    def __str__(self) -> str:
        """Return string representation of exception."""
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class WeexAuthenticationError(WeexAPIError):
    """Authentication related errors (40001-40014, 40011-40012)."""

    pass


class WeexRateLimitError(WeexAPIError):
    """Rate limit error (429)."""

    def __init__(
        self,
        message: str = "Too many requests",
        code: Optional[str] = None,
        request_time: Optional[int] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message
            code: Error code
            request_time: Request timestamp
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message, code, request_time)
        self.retry_after = retry_after


class WeexNetworkError(WeexAPIError):
    """Network related errors."""

    pass


class WeexWebSocketError(WeexAPIError):
    """WebSocket related errors."""

    pass


class WeexValidationError(WeexAPIError):
    """Parameter validation errors (40017-40020)."""

    pass


# Error code mapping based on API documentation
ERROR_CODE_MAP: dict[str, type[WeexAPIError]] = {
    "40001": WeexAuthenticationError,  # Header "ACCESS_KEY" is required
    "40002": WeexAuthenticationError,  # Header "ACCESS_SIGN" is required
    "40003": WeexAuthenticationError,  # Header "ACCESS_TIMESTAMP" is required
    "40005": WeexAuthenticationError,  # Invalid ACCESS_TIMESTAMP
    "40006": WeexAuthenticationError,  # Invalid ACCESS_KEY
    "40007": WeexAuthenticationError,  # Invalid Content_Type
    "40008": WeexAuthenticationError,  # Request timestamp expired
    "40009": WeexAuthenticationError,  # API verification failed
    "40011": WeexAuthenticationError,  # Header "ACCESS_PASSPHRASE" is required
    "40012": WeexAuthenticationError,  # Incorrect API key/Passphrase
    "40013": WeexAuthenticationError,  # Account frozen
    "40014": WeexAuthenticationError,  # Invalid permissions
    "40015": WeexAPIError,  # System error
    "40017": WeexValidationError,  # Parameter validation failed
    "40018": WeexAuthenticationError,  # Invalid IP request
    "40019": WeexValidationError,  # Parameter cannot be empty
    "40020": WeexValidationError,  # Parameter is invalid
    "40022": WeexAuthenticationError,  # Insufficient permissions
    "40753": WeexAuthenticationError,  # API permission disabled
    "429": WeexRateLimitError,  # Too many requests
    "50003": WeexAPIError,  # Not have permission to trade this pair
    "50004": WeexAPIError,  # Not have permission to access this API
    "50005": WeexAPIError,  # Order does not exist
    "50007": WeexAPIError,  # Leverage cannot exceed the limit
}


def raise_exception_from_response(
    code: str,
    message: str,
    request_time: Optional[int] = None,
) -> None:
    """Raise appropriate exception based on error code.

    Args:
        code: Error code from API response
        message: Error message
        request_time: Request timestamp

    Raises:
        WeexAPIError: Appropriate exception based on error code
    """
    exception_class = ERROR_CODE_MAP.get(code, WeexAPIError)
    raise exception_class(message=message, code=code, request_time=request_time)
