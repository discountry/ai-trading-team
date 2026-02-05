"""Technical indicators using talipp library."""

from typing import Protocol, runtime_checkable

from ai_trading_team.indicators.base import Indicator, OHLCVData

# Note: Actual talipp imports will be added when implementing
# from talipp.indicators import RSI, MACD, BB, etc.


@runtime_checkable
class _TalippIndicator(Protocol):
    """Protocol for talipp indicator interface."""

    def add(self, value: float) -> None: ...
    def __getitem__(self, index: int) -> float | None: ...
    def __bool__(self) -> bool: ...


class RSIIndicator(Indicator):
    """Relative Strength Index indicator."""

    def __init__(self, period: int = 14) -> None:
        super().__init__(f"RSI_{period}", period=period)
        self._period = period
        self._indicator: _TalippIndicator | None = None

    def update(self, data: OHLCVData) -> float | None:
        """Update RSI with new close price."""
        # TODO: Implementation will use talipp RSI
        # self._indicator.add(data["close"])
        # self._value = self._indicator[-1] if self._indicator else None
        _ = data  # Placeholder until implemented
        value = self._value
        if value is None:
            return None
        if isinstance(value, int | float):
            return float(value)
        return None

    def reset(self) -> None:
        """Reset RSI state."""
        self._indicator = None
        self._value = None


class MACDIndicator(Indicator):
    """Moving Average Convergence Divergence indicator."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        super().__init__(
            f"MACD_{fast_period}_{slow_period}_{signal_period}",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        self._indicator: _TalippIndicator | None = None

    def update(self, data: OHLCVData) -> dict[str, float] | None:
        """Update MACD with new close price.

        Returns dict with 'macd', 'signal', 'histogram' keys.
        """
        # TODO: Implementation will use talipp MACD
        _ = data  # Placeholder until implemented
        value = self._value
        if value is None:
            return None
        if isinstance(value, dict):
            # Ensure all values are float
            return {k: float(v) for k, v in value.items()}
        return None

    def reset(self) -> None:
        """Reset MACD state."""
        self._indicator = None
        self._value = None


class BollingerBandsIndicator(Indicator):
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        super().__init__(f"BB_{period}_{std_dev}", period=period, std_dev=std_dev)
        self._indicator: _TalippIndicator | None = None

    def update(self, data: OHLCVData) -> dict[str, float] | None:
        """Update BB with new close price.

        Returns dict with 'upper', 'middle', 'lower' keys.
        """
        # TODO: Implementation will use talipp BB
        _ = data  # Placeholder until implemented
        value = self._value
        if value is None:
            return None
        if isinstance(value, dict):
            # Ensure all values are float
            return {k: float(v) for k, v in value.items()}
        return None

    def reset(self) -> None:
        """Reset BB state."""
        self._indicator = None
        self._value = None


class ATRIndicator(Indicator):
    """Average True Range indicator."""

    def __init__(self, period: int = 14) -> None:
        super().__init__(f"ATR_{period}", period=period)
        self._indicator: _TalippIndicator | None = None

    def update(self, data: OHLCVData) -> float | None:
        """Update ATR with new OHLC data."""
        # TODO: Implementation will use talipp ATR
        _ = data  # Placeholder until implemented
        value = self._value
        if value is None:
            return None
        if isinstance(value, int | float):
            return float(value)
        return None

    def reset(self) -> None:
        """Reset ATR state."""
        self._indicator = None
        self._value = None
