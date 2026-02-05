"""Risk control monitor."""

from typing import Any, Protocol

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.risk.actions import RiskAction
from ai_trading_team.risk.rules import RiskRule


class ExchangeProtocol(Protocol):
    """Protocol for exchange interface used by RiskMonitor."""

    async def get_account(self) -> Any:
        """Get account information."""
        ...

    async def get_positions(self) -> list[Any]:
        """Get all open positions."""
        ...


class RiskMonitor:
    """Risk monitor that continuously evaluates risk rules.

    Independent from Agent, operates with higher priority.
    """

    def __init__(
        self,
        data_pool: DataPool,
        exchange: ExchangeProtocol,
    ) -> None:
        self._data_pool = data_pool
        self._exchange = exchange
        self._rules: list[RiskRule] = []
        self._enabled = True

    def add_rule(self, rule: RiskRule) -> None:
        """Add a risk rule."""
        self._rules.append(rule)
        # Sort by priority (highest first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> None:
        """Remove a risk rule by name."""
        self._rules = [r for r in self._rules if r.name != name]

    def enable(self) -> None:
        """Enable risk monitoring."""
        self._enabled = True

    def disable(self) -> None:
        """Disable risk monitoring."""
        self._enabled = False

    def reset_rules(self) -> None:
        """Reset all stateful risk rules.

        Called when a position is closed to reset tracking state
        for rules like DynamicTakeProfitRule and TrailingStopRule.
        """
        for rule in self._rules:
            if hasattr(rule, "reset"):
                rule.reset()

    async def evaluate(self) -> RiskAction | None:
        """Evaluate all risk rules.

        Returns:
            First triggered action (highest priority), or None
        """
        if not self._enabled:
            return None

        snapshot = self._data_pool.get_snapshot()
        account = await self._exchange.get_account()
        positions = await self._exchange.get_positions()

        # Check rules for each position
        for rule in self._rules:
            if not rule.enabled:
                continue

            # Only evaluate with position for position-dependent rules
            for position in positions:
                action = rule.evaluate(snapshot, position, account)
                if action:
                    return action

        # Only check with None if no positions exist (for non-position rules)
        if not positions:
            for rule in self._rules:
                if not rule.enabled:
                    continue
                action = rule.evaluate(snapshot, None, account)
                if action:
                    return action

        return None

    @property
    def rules(self) -> list[RiskRule]:
        """Get all registered rules."""
        return list(self._rules)
