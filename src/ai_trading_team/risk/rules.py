"""Risk control rules.

According to STRATEGY.md:
- Force stop-loss at 25% of margin loss (no agent needed)
- Dynamic take-profit signals at 10% profit increments (agent decides)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.execution.models import Account, Position
from ai_trading_team.risk.actions import RiskAction


class RuleType(str, Enum):
    """Types of risk rules."""

    STOP_LOSS = "stop_loss"
    FORCE_STOP_LOSS = "force_stop_loss"  # 25% - no agent needed
    TAKE_PROFIT = "take_profit"
    DYNAMIC_TAKE_PROFIT = "dynamic_take_profit"  # 10% increments
    MAX_DRAWDOWN = "max_drawdown"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_LEVERAGE = "max_leverage"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class RiskRule(ABC):
    """Abstract risk rule base class."""

    name: str
    enabled: bool = True
    priority: int = 0  # Higher = checked first

    @abstractmethod
    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Evaluate the risk rule.

        Args:
            snapshot: Current market data snapshot
            position: Current position (if any)
            account: Current account state

        Returns:
            Risk action if triggered, None otherwise
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get rule configuration."""
        ...


@dataclass
class StopLossRule(RiskRule):
    """Fixed stop loss rule."""

    stop_loss_percent: Decimal = Decimal("1.0")  # Synced with prompts: 1% SL

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position has exceeded stop loss threshold."""
        if not position:
            return None

        # Guard against zero margin
        if position.margin <= 0:
            return None

        # Calculate P&L percentage
        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        if pnl_percent <= -self.stop_loss_percent:
            return RiskAction(
                action_type="close",
                symbol=position.symbol,
                reason=f"Stop loss triggered: PnL {pnl_percent:.2f}% <= -{self.stop_loss_percent}%",
                priority=self.priority,
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"stop_loss_percent": float(self.stop_loss_percent)}


@dataclass
class TakeProfitRule(RiskRule):
    """Fixed take profit rule."""

    take_profit_percent: Decimal = Decimal("5.0")  # Synced with prompts: 5% TP

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position has reached take profit threshold."""
        if not position:
            return None

        # Guard against zero margin
        if position.margin <= 0:
            return None

        # Calculate P&L percentage
        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        if pnl_percent >= self.take_profit_percent:
            return RiskAction(
                action_type="close",
                symbol=position.symbol,
                reason=f"Take profit triggered: PnL {pnl_percent:.2f}% >= {self.take_profit_percent}%",
                priority=self.priority,
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"take_profit_percent": float(self.take_profit_percent)}


@dataclass
class MaxDrawdownRule(RiskRule):
    """Maximum account drawdown rule."""

    max_drawdown_percent: Decimal = Decimal("10.0")
    peak_equity: Decimal = Decimal("0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if account drawdown exceeds threshold."""
        current_equity = account.total_equity

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return None

        if self.peak_equity <= 0:
            return None

        drawdown_percent = ((self.peak_equity - current_equity) / self.peak_equity) * 100

        if drawdown_percent >= self.max_drawdown_percent:
            return RiskAction(
                action_type="close_all",
                symbol="*",
                reason=f"Max drawdown triggered: {drawdown_percent:.2f}% >= {self.max_drawdown_percent}%",
                priority=100,  # Highest priority
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"max_drawdown_percent": float(self.max_drawdown_percent)}


@dataclass
class ForceStopLossRule(RiskRule):
    """Force stop-loss rule at 25% margin loss.

    This rule executes immediately without agent confirmation.
    Per STRATEGY.md: "每次亏损超过保证金的 25% 以上时强制止损"
    """

    name: str = "force_stop_loss"
    force_stop_loss_percent: Decimal = Decimal("25.0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position has exceeded 25% margin loss.

        This is a FORCE action - bypasses agent.
        """
        if not position:
            return None

        if position.margin <= 0:
            return None

        # Calculate P&L percentage of margin
        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        if pnl_percent <= -self.force_stop_loss_percent:
            return RiskAction(
                action_type="close",
                symbol=position.symbol,
                reason=(
                    f"FORCE STOP-LOSS: Loss {pnl_percent:.2f}% >= {self.force_stop_loss_percent}% of margin. "
                    f"This action is executed immediately without agent confirmation."
                ),
                priority=100,  # Highest priority - immediate execution
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"force_stop_loss_percent": float(self.force_stop_loss_percent)}


@dataclass
class DynamicTakeProfitRule(RiskRule):
    """Dynamic take-profit rule that signals at 10% profit increments.

    This rule generates SIGNALS for agent to set/move stop loss orders.
    Per STRATEGY.md: "每当收益增加保证金的 10% 以上时，推送信号给 agent"

    Triggers at: 10%, 20%, 30%, 40%, 50%, ... of margin profit.
    Default action: Agent sets/moves stop loss order based on market conditions.
    """

    name: str = "dynamic_take_profit"
    profit_threshold_percent: Decimal = Decimal("10.0")

    # Track last threshold that triggered a signal
    _last_signaled_threshold: Decimal = Decimal("0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position profit crossed a 10% threshold.

        Triggers at each 10% increment: 10%, 20%, 30%, 40%, etc.
        Returns a "signal" action type for agent to set stop loss.
        """
        if not position:
            self._last_signaled_threshold = Decimal("0")
            return None

        if position.margin <= 0:
            return None

        # Calculate P&L percentage of margin
        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        # Only trigger on profit
        if pnl_percent <= 0:
            return None

        # Calculate which threshold level we're at (10, 20, 30, ...)
        current_threshold_level = (
            pnl_percent // self.profit_threshold_percent
        ) * self.profit_threshold_percent

        # Check if we crossed a new threshold
        if current_threshold_level > self._last_signaled_threshold:
            # Update to current level
            self._last_signaled_threshold = current_threshold_level

            # This is a signal for agent to set/move stop loss
            return RiskAction(
                action_type="move_stop_loss",  # Changed from "close" to "move_stop_loss"
                symbol=position.symbol,
                reason=(
                    f"PROFIT THRESHOLD {int(current_threshold_level)}%: "
                    f"Profit reached {pnl_percent:.2f}% of margin. "
                    f"Please set/move stop loss order based on current market conditions."
                ),
                priority=50,  # Lower priority - agent decides stop loss price
                data={
                    "current_pnl_percent": float(pnl_percent),
                    "threshold_level": int(current_threshold_level),
                    "entry_price": float(position.entry_price),
                    "current_margin": float(position.margin),
                    "position_side": position.side.value,
                },
            )

        return None

    def reset(self) -> None:
        """Reset the threshold tracker when position closes."""
        self._last_signaled_threshold = Decimal("0")

    def get_config(self) -> dict[str, Any]:
        return {
            "profit_threshold_percent": float(self.profit_threshold_percent),
            "last_signaled_threshold": float(self._last_signaled_threshold),
        }


@dataclass
class TrailingStopRule(RiskRule):
    """Trailing stop rule to protect profits.

    When position is in profit, set a trailing stop that moves
    with the highest profit reached.
    """

    name: str = "trailing_stop"
    activation_profit_percent: Decimal = Decimal("10.0")  # Activate at 10% profit
    trail_distance_percent: Decimal = Decimal("5.0")  # Trail by 5%

    # Track highest profit percentage
    _highest_profit_percent: Decimal = Decimal("0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if trailing stop should trigger."""
        if not position:
            self._highest_profit_percent = Decimal("0")
            return None

        if position.margin <= 0:
            return None

        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        # Update highest profit
        if pnl_percent > self._highest_profit_percent:
            self._highest_profit_percent = pnl_percent

        # Only activate trailing stop if we reached activation threshold
        if self._highest_profit_percent < self.activation_profit_percent:
            return None

        # Calculate trailing stop level
        stop_level = self._highest_profit_percent - self.trail_distance_percent

        # Trigger if profit dropped below trailing stop
        if pnl_percent <= stop_level and stop_level > 0:
            return RiskAction(
                action_type="close",
                symbol=position.symbol,
                reason=(
                    f"TRAILING STOP: Profit dropped from {self._highest_profit_percent:.2f}% "
                    f"to {pnl_percent:.2f}% (stop at {stop_level:.2f}%)"
                ),
                priority=80,
            )

        return None

    def reset(self) -> None:
        """Reset when position closes."""
        self._highest_profit_percent = Decimal("0")

    def get_config(self) -> dict[str, Any]:
        return {
            "activation_profit_percent": float(self.activation_profit_percent),
            "trail_distance_percent": float(self.trail_distance_percent),
            "highest_profit_percent": float(self._highest_profit_percent),
        }


@dataclass
class MaxPositionSizeRule(RiskRule):
    """Maximum position size rule.

    Enforces the 750 USDT maximum margin limit for total position size.
    Per prompts: "总仓位占用保证金 ≤ 750 USDT"

    This is a PRE-TRADE validation rule that should be checked before
    opening or adding to positions.
    """

    name: str = "max_position_size"
    max_margin_usdt: Decimal = Decimal("750.0")  # Maximum total margin allowed

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position margin exceeds maximum allowed.

        This rule is primarily for pre-trade validation, but also monitors
        existing positions in case they grow beyond limits.
        """
        if not position:
            return None

        # Check if current position margin exceeds max
        if position.margin > float(self.max_margin_usdt):
            return RiskAction(
                action_type="reduce",
                symbol=position.symbol,
                reason=(
                    f"MAX POSITION SIZE EXCEEDED: Current margin {position.margin:.2f} USDT "
                    f"> max allowed {self.max_margin_usdt} USDT. Reduce position size."
                ),
                priority=90,  # High priority
                data={
                    "current_margin": float(position.margin),
                    "max_margin": float(self.max_margin_usdt),
                    "excess_margin": float(position.margin) - float(self.max_margin_usdt),
                },
            )

        return None

    def validate_new_order(
        self,
        current_margin: float,
        new_order_margin: float,
    ) -> tuple[bool, str]:
        """Validate if a new order would exceed position size limits.

        Args:
            current_margin: Current position margin in USDT
            new_order_margin: Margin required for the new order

        Returns:
            Tuple of (is_valid, reason)
        """
        total_margin = current_margin + new_order_margin

        if total_margin > float(self.max_margin_usdt):
            return (
                False,
                f"Order rejected: Total margin {total_margin:.2f} USDT would exceed "
                f"max {self.max_margin_usdt} USDT (current: {current_margin:.2f}, "
                f"new: {new_order_margin:.2f})",
            )

        return (True, "")

    def get_config(self) -> dict[str, Any]:
        return {"max_margin_usdt": float(self.max_margin_usdt)}
