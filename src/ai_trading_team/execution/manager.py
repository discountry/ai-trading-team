"""Execution manager - coordinates exchange operations."""

from abc import ABC, abstractmethod

from ai_trading_team.agent.commands import AgentCommand
from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.execution.base import Exchange
from ai_trading_team.execution.models import Order


class ExecutionManager(ABC):
    """Abstract execution manager.

    Coordinates exchange operations and data synchronization.
    """

    def __init__(self, exchange: Exchange, data_pool: DataPool) -> None:
        self._exchange = exchange
        self._data_pool = data_pool

    @property
    def exchange(self) -> Exchange:
        """Get underlying exchange."""
        return self._exchange

    @abstractmethod
    async def start(self) -> None:
        """Start execution manager.

        Connects to exchange and starts data synchronization.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop execution manager."""
        ...

    @abstractmethod
    async def execute_command(self, command: AgentCommand) -> Order | None:
        """Execute an agent command.

        Args:
            command: Agent command to execute

        Returns:
            Resulting order if applicable, None for observe
        """
        ...

    @abstractmethod
    async def sync_account(self) -> None:
        """Synchronize account data from exchange."""
        ...

    @abstractmethod
    async def sync_positions(self) -> None:
        """Synchronize position data from exchange."""
        ...

    @abstractmethod
    async def sync_orders(self) -> None:
        """Synchronize order data from exchange."""
        ...
