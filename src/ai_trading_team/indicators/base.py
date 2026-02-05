"""Base indicator class."""

from abc import ABC, abstractmethod
from typing import TypedDict


class OHLCVData(TypedDict, total=False):
    """OHLCV market data for indicator calculation."""

    open: float
    high: float
    low: float
    close: float
    volume: float


class Indicator(ABC):
    """Abstract base class for technical indicators.

    Wraps talipp indicators with a unified interface.
    """

    def __init__(self, name: str, **params: int | float | str) -> None:
        self._name = name
        self._params = params
        self._value: float | dict[str, float] | None = None

    @property
    def name(self) -> str:
        """Indicator name."""
        return self._name

    @property
    def params(self) -> dict[str, int | float | str]:
        """Indicator parameters."""
        return self._params

    @property
    def value(self) -> float | dict[str, float] | None:
        """Current indicator value."""
        return self._value

    @abstractmethod
    def update(self, data: OHLCVData) -> float | dict[str, float] | None:
        """Update indicator with new data.

        Args:
            data: OHLCV data dict with keys: open, high, low, close, volume

        Returns:
            Updated indicator value
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state."""
        ...
