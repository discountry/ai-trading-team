"""Parameter validation utilities."""

from typing import Union

from weex_sdk.exceptions import WeexValidationError


def validate_symbol(symbol: str) -> None:
    """Validate trading pair symbol format.

    Args:
        symbol: Trading pair symbol (e.g., 'cmt_btcusdt')

    Raises:
        WeexValidationError: If symbol format is invalid
    """
    if not symbol:
        raise WeexValidationError("Symbol cannot be empty")

    if not isinstance(symbol, str):
        raise WeexValidationError(f"Symbol must be a string, got {type(symbol)}")

    # Basic format check: should start with 'cmt_' and contain base/quote
    if not symbol.startswith("cmt_"):
        raise WeexValidationError(f"Symbol must start with 'cmt_', got: {symbol}")

    if len(symbol) < 8:  # Minimum: cmt_XX
        raise WeexValidationError(f"Symbol too short: {symbol}")


def validate_price(price: Union[str, float, int]) -> None:
    """Validate price value.

    Args:
        price: Price value

    Raises:
        WeexValidationError: If price is invalid
    """
    try:
        price_float = float(price)
        if price_float < 0:
            raise WeexValidationError(f"Price cannot be negative: {price}")
    except (ValueError, TypeError):
        raise WeexValidationError(f"Invalid price format: {price}")


def validate_size(size: Union[str, float, int]) -> None:
    """Validate order size.

    Args:
        size: Order size

    Raises:
        WeexValidationError: If size is invalid
    """
    try:
        size_float = float(size)
        if size_float <= 0:
            raise WeexValidationError(f"Size must be positive: {size}")
    except (ValueError, TypeError):
        raise WeexValidationError(f"Invalid size format: {size}")


def validate_order_type(order_type: Union[str, int]) -> None:
    """Validate order type.

    Args:
        order_type: Order type (0: Normal, 1: Post-Only, 2: FOK, 3: IOC)

    Raises:
        WeexValidationError: If order type is invalid
    """
    valid_types = ["0", "1", "2", "3"]
    order_type_str = str(order_type)
    if order_type_str not in valid_types:
        raise WeexValidationError(f"Invalid order type: {order_type}. Must be one of {valid_types}")


def validate_order_direction(direction: Union[str, int]) -> None:
    """Validate order direction/type.

    Args:
        direction: Order direction (1: Open long, 2: Open short, 3: Close long, 4: Close short)

    Raises:
        WeexValidationError: If direction is invalid
    """
    valid_directions = ["1", "2", "3", "4"]
    direction_str = str(direction)
    if direction_str not in valid_directions:
        raise WeexValidationError(
            f"Invalid order direction: {direction}. Must be one of {valid_directions}"
        )


def validate_margin_mode(margin_mode: Union[str, int]) -> None:
    """Validate margin mode.

    Args:
        margin_mode: Margin mode (1: Cross, 3: Isolated)

    Raises:
        WeexValidationError: If margin mode is invalid
    """
    valid_modes = ["1", "3", 1, 3]
    if margin_mode not in valid_modes:
        raise WeexValidationError(
            f"Invalid margin mode: {margin_mode}. Must be 1 (Cross) or 3 (Isolated)"
        )


def validate_client_oid(client_oid: str) -> None:
    """Validate client order ID.

    Args:
        client_oid: Client order ID

    Raises:
        WeexValidationError: If client order ID is invalid
    """
    if not client_oid:
        raise WeexValidationError("Client order ID cannot be empty")

    if not isinstance(client_oid, str):
        raise WeexValidationError(f"Client order ID must be a string, got {type(client_oid)}")

    if len(client_oid) > 40:
        raise WeexValidationError(
            f"Client order ID cannot exceed 40 characters, got {len(client_oid)}"
        )


def validate_limit(limit: int, min_value: int = 1, max_value: int = 100) -> None:
    """Validate limit parameter.

    Args:
        limit: Limit value
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Raises:
        WeexValidationError: If limit is out of range
    """
    if not isinstance(limit, int):
        raise WeexValidationError(f"Limit must be an integer, got {type(limit)}")

    if limit < min_value or limit > max_value:
        raise WeexValidationError(f"Limit must be between {min_value} and {max_value}, got {limit}")


def validate_timestamp(timestamp: int) -> None:
    """Validate timestamp (milliseconds).

    Args:
        timestamp: Timestamp in milliseconds

    Raises:
        WeexValidationError: If timestamp is invalid
    """
    if not isinstance(timestamp, int):
        raise WeexValidationError(f"Timestamp must be an integer, got {type(timestamp)}")

    # Check if timestamp is reasonable (between 2000-01-01 and 2100-01-01)
    min_ts = 946684800000  # 2000-01-01
    max_ts = 4102444800000  # 2100-01-01

    if timestamp < min_ts or timestamp > max_ts:
        raise WeexValidationError(f"Timestamp out of valid range: {timestamp}")
