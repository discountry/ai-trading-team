"""Price level strategy."""

from decimal import Decimal
from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.strategy.base import Strategy


class PriceLevelStrategy(Strategy):
    """Strategy that triggers when price reaches support/resistance levels."""

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        support_levels: list[Decimal] | None = None,
        resistance_levels: list[Decimal] | None = None,
        tolerance_percent: float = 0.1,
    ) -> None:
        super().__init__("Price_Level", data_pool, signal_queue)
        self._support_levels = support_levels or []
        self._resistance_levels = resistance_levels or []
        self._tolerance = Decimal(str(tolerance_percent / 100))

    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check if price is near support/resistance levels."""
        if not snapshot.ticker:
            return None

        current_price = snapshot.ticker.get("last_price")
        if current_price is None:
            return None

        price = Decimal(str(current_price))

        # Check support levels
        for level in self._support_levels:
            tolerance = level * self._tolerance
            if abs(price - level) <= tolerance:
                return StrategySignal(
                    signal_type=SignalType.PRICE_SUPPORT,
                    data={
                        "current_price": float(price),
                        "support_level": float(level),
                        "tolerance_percent": float(self._tolerance * 100),
                        "strategy": self.name,
                    },
                )

        # Check resistance levels
        for level in self._resistance_levels:
            tolerance = level * self._tolerance
            if abs(price - level) <= tolerance:
                return StrategySignal(
                    signal_type=SignalType.PRICE_RESISTANCE,
                    data={
                        "current_price": float(price),
                        "resistance_level": float(level),
                        "tolerance_percent": float(self._tolerance * 100),
                        "strategy": self.name,
                    },
                )

        return None

    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "support_levels": [float(level) for level in self._support_levels],
            "resistance_levels": [float(level) for level in self._resistance_levels],
            "tolerance_percent": float(self._tolerance * 100),
        }

    def add_support_level(self, level: Decimal) -> None:
        """Add a support level."""
        if level not in self._support_levels:
            self._support_levels.append(level)

    def add_resistance_level(self, level: Decimal) -> None:
        """Add a resistance level."""
        if level not in self._resistance_levels:
            self._resistance_levels.append(level)
