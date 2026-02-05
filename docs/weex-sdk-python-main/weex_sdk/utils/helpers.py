"""Helper utilities for Weex SDK."""

import time
from typing import Any, Dict, List, Optional, Union


def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds.

    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def build_query_string(params: Dict[str, Any]) -> str:
    """Build query string from parameters dictionary.

    Args:
        params: Dictionary of query parameters

    Returns:
        Query string (without '?')
    """
    if not params:
        return ""

    # Filter out None values
    filtered_params = {k: v for k, v in params.items() if v is not None}

    if not filtered_params:
        return ""

    # Build query string
    query_parts = []
    for key, value in filtered_params.items():
        if isinstance(value, list):
            # Handle list values
            for item in value:
                query_parts.append(f"{key}={item}")
        else:
            query_parts.append(f"{key}={value}")

    return "&".join(query_parts)


def sanitize_log_data(
    data: Dict[str, Any], sensitive_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Sanitize sensitive data for logging.

    Args:
        data: Data dictionary to sanitize
        sensitive_keys: List of keys to mask (default: common sensitive keys)

    Returns:
        Sanitized data dictionary
    """
    if sensitive_keys is None:
        sensitive_keys = [
            "secret_key",
            "secretKey",
            "passphrase",
            "ACCESS-KEY",
            "ACCESS-SIGN",
            "ACCESS-PASSPHRASE",
        ]

    sanitized = data.copy()
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "***"

    return sanitized


def format_decimal(value: Union[str, float, int], precision: int = 8) -> str:
    """Format decimal value to string with specified precision.

    Args:
        value: Decimal value to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    try:
        float_value = float(value)
        return f"{float_value:.{precision}f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return str(value)
