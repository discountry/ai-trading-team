"""MACD cross strategy."""

from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy


class MACDCrossStrategy(Strategy):
    """Strategy that triggers on MACD crossovers."""

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        macd_indicator_name: str = "MACD_12_26_9",
    ) -> None:
        super().__init__("MACD_Cross", data_pool, signal_queue)
        self._macd_name = macd_indicator_name
        self._prev_histogram: float | None = None

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check for MACD crossover conditions."""
        if not snapshot.indicators:
            return None

        macd_value = snapshot.indicators.get(self._macd_name)
        if not isinstance(macd_value, dict):
            return None

        histogram = macd_value.get("histogram")
        if histogram is None:
            return None

        signal: StrategySignal | None = None

        # Check for crossover
        if self._prev_histogram is not None:
            # Bullish cross: histogram crosses from negative to positive
            if self._prev_histogram < 0 and histogram >= 0:
                signal = StrategySignal(
                    signal_type=SignalType.MACD_BULLISH_CROSS,
                    data={
                        "macd": macd_value.get("macd"),
                        "signal": macd_value.get("signal"),
                        "histogram": histogram,
                        "strategy": self.name,
                    },
                )
            # Bearish cross: histogram crosses from positive to negative
            elif self._prev_histogram > 0 and histogram <= 0:
                signal = StrategySignal(
                    signal_type=SignalType.MACD_BEARISH_CROSS,
                    data={
                        "macd": macd_value.get("macd"),
                        "signal": macd_value.get("signal"),
                        "histogram": histogram,
                        "strategy": self.name,
                    },
                )

        self._prev_histogram = histogram
        return signal

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "macd_indicator_name": self._macd_name,
        }
