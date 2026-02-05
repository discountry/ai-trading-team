"""RSI oversold/overbought strategy."""

from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy


class RSIOversoldStrategy(Strategy):
    """Strategy that triggers on RSI oversold/overbought conditions."""

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        rsi_indicator_name: str = "RSI_14",
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
    ) -> None:
        super().__init__("RSI_Oversold_Overbought", data_pool, signal_queue)
        self._rsi_name = rsi_indicator_name
        self._oversold = oversold_threshold
        self._overbought = overbought_threshold

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check for RSI oversold/overbought conditions."""
        if not snapshot.indicators:
            return None

        rsi_value = snapshot.indicators.get(self._rsi_name)
        if rsi_value is None:
            return None

        if rsi_value <= self._oversold:
            return StrategySignal(
                signal_type=SignalType.RSI_OVERSOLD,
                data={
                    "rsi_value": rsi_value,
                    "threshold": self._oversold,
                    "strategy": self.name,
                },
            )

        if rsi_value >= self._overbought:
            return StrategySignal(
                signal_type=SignalType.RSI_OVERBOUGHT,
                data={
                    "rsi_value": rsi_value,
                    "threshold": self._overbought,
                    "strategy": self.name,
                },
            )

        return None

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "rsi_indicator_name": self._rsi_name,
            "oversold_threshold": self._oversold,
            "overbought_threshold": self._overbought,
        }
