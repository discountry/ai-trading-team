"""Base strategy class."""

from abc import ABC, abstractmethod
from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, StrategySignal


class Strategy(ABC):
    """Abstract base class for mechanical strategies.

    Monitors data pool and generates signals when conditions are met.
    """

    def __init__(
        self,
        name: str,
        data_pool: DataPool,
        signal_queue: SignalQueue,
    ) -> None:
        self._name = name
        self._data_pool = data_pool
        self._signal_queue = signal_queue
        self._enabled = True

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Whether strategy is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the strategy."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the strategy."""
        self._enabled = False

    @abstractmethod
    def check_conditions(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Check if strategy conditions are met.

        Args:
            snapshot: Current data snapshot

        Returns:
            Signal if conditions met, None otherwise
        """
        ...

    def evaluate(self) -> StrategySignal | None:
        """Evaluate strategy and emit signal if conditions met.

        Returns:
            Emitted signal or None
        """
        if not self._enabled:
            return None

        snapshot = self._data_pool.get_snapshot()
        signal = self.check_conditions(snapshot)

        if signal:
            self._signal_queue.push(signal)
            return signal

        return None

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        ...
