"""Grid strategy specific exceptions."""

from weex_sdk.exceptions import WeexAPIError


class GridStrategyError(WeexAPIError):
    """Base exception for grid strategy."""

    pass


class InvalidGridDirectionError(GridStrategyError):
    """Raised when an invalid grid direction is specified."""

    pass
