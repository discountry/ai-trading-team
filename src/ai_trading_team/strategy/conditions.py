"""Strategy condition definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class ConditionOperator(str, Enum):
    """Comparison operators for conditions."""

    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


@dataclass
class Condition(ABC):
    """Abstract condition base class."""

    @abstractmethod
    def evaluate(self, data: dict[str, Any]) -> bool:
        """Evaluate the condition.

        Args:
            data: Data dict containing values to check

        Returns:
            True if condition is met
        """
        ...


@dataclass
class PriceCondition(Condition):
    """Price-based condition."""

    operator: ConditionOperator
    target_price: Decimal
    price_field: str = "last_price"

    def evaluate(self, data: dict[str, Any]) -> bool:
        """Check if price condition is met."""
        current_price = data.get(self.price_field)
        if current_price is None:
            return False

        current = Decimal(str(current_price))

        match self.operator:
            case ConditionOperator.GT:
                return current > self.target_price
            case ConditionOperator.GTE:
                return current >= self.target_price
            case ConditionOperator.LT:
                return current < self.target_price
            case ConditionOperator.LTE:
                return current <= self.target_price
            case ConditionOperator.EQ:
                return current == self.target_price
            case ConditionOperator.NEQ:
                return current != self.target_price
            case _:
                return False


@dataclass
class IndicatorCondition(Condition):
    """Indicator-based condition."""

    indicator_name: str
    operator: ConditionOperator
    threshold: float
    indicator_field: str | None = None  # For multi-value indicators like MACD

    def evaluate(self, data: dict[str, Any]) -> bool:
        """Check if indicator condition is met."""
        indicators = data.get("indicators", {})
        value = indicators.get(self.indicator_name)

        if value is None:
            return False

        # Handle multi-value indicators
        if self.indicator_field and isinstance(value, dict):
            value = value.get(self.indicator_field)
            if value is None:
                return False

        match self.operator:
            case ConditionOperator.GT:
                return bool(value > self.threshold)
            case ConditionOperator.GTE:
                return bool(value >= self.threshold)
            case ConditionOperator.LT:
                return bool(value < self.threshold)
            case ConditionOperator.LTE:
                return bool(value <= self.threshold)
            case ConditionOperator.EQ:
                return bool(value == self.threshold)
            case ConditionOperator.NEQ:
                return bool(value != self.threshold)
            case _:
                return False
