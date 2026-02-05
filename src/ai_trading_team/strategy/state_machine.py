"""Strategy State Machine for trading lifecycle management.

This module implements a comprehensive state machine that manages the trading lifecycle,
ensuring stable and predictable behavior throughout the trading session.

State Diagram:
    IDLE -> ANALYZING (on signal) -> WAITING_ENTRY (agent decision) -> IN_POSITION (trade executed)
    IN_POSITION -> MONITORING (profit/loss check) -> WAITING_EXIT (exit signal) -> COOLDOWN -> IDLE
    Any State -> RISK_OVERRIDE (on 25% loss) -> COOLDOWN -> IDLE
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from ai_trading_team.core.types import Side

logger = logging.getLogger(__name__)


class StrategyState(str, Enum):
    """Trading strategy states."""

    IDLE = "idle"  # No position, waiting for entry signals
    ANALYZING = "analyzing"  # Signal received, gathering context
    WAITING_ENTRY = "waiting_entry"  # Waiting for agent entry decision
    IN_POSITION = "in_position"  # Position open, monitoring
    PROFIT_SIGNAL = "profit_signal"  # Profit threshold reached, asking agent
    WAITING_EXIT = "waiting_exit"  # Waiting for agent exit decision
    RISK_OVERRIDE = "risk_override"  # Risk control triggered, force closing
    COOLDOWN = "cooldown"  # Post-trade cooldown period
    ERROR = "error"  # Error state, needs manual intervention


class StateTransition(str, Enum):
    """State transition triggers."""

    ENTRY_SIGNAL = "entry_signal"  # Entry signal from strategy factors
    CONTEXT_READY = "context_ready"  # Market context gathered
    AGENT_OPEN = "agent_open"  # Agent decided to open position
    AGENT_OBSERVE = "agent_observe"  # Agent decided to observe
    ORDER_PLACED = "order_placed"  # Order placed, awaiting fill
    ORDER_FILLED = "order_filled"  # Order executed successfully
    PROFIT_THRESHOLD = "profit_threshold"  # 10% profit increase
    AGENT_CLOSE = "agent_close"  # Agent decided to close
    AGENT_HOLD = "agent_hold"  # Agent decided to hold position
    RISK_TRIGGERED = "risk_triggered"  # 25% loss - force close
    POSITION_CLOSED = "position_closed"  # Position closed
    COOLDOWN_EXPIRED = "cooldown_expired"  # Cooldown period ended
    ORDER_FAILED = "order_failed"  # Order execution failed
    TIMEOUT = "timeout"  # State timeout
    RESET = "reset"  # Manual reset


@dataclass
class PositionContext:
    """Current position context for state tracking."""

    symbol: str = ""
    side: Side | None = None
    entry_price: Decimal = Decimal("0")
    size: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")
    leverage: int = 1
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_percent: Decimal = Decimal("0")
    entry_time: datetime | None = None
    highest_pnl_percent: Decimal = Decimal("0")  # For trailing stop
    last_profit_signal_threshold: Decimal = Decimal("0")  # Last 10% threshold crossed
    stop_loss_price: float | None = None  # Current stop loss target price


@dataclass
class StateContext:
    """Context data for state machine."""

    current_state: StrategyState = StrategyState.IDLE
    previous_state: StrategyState | None = None
    state_entered_at: datetime = field(default_factory=datetime.now)
    position: PositionContext = field(default_factory=PositionContext)

    # Signal that triggered current analysis
    pending_signal_type: str | None = None
    pending_signal_data: dict[str, Any] = field(default_factory=dict)

    # Cooldown settings
    cooldown_until: datetime | None = None
    cooldown_duration_seconds: int = 60  # 1 minute default

    # Signal debounce - prevent sending same signal repeatedly
    last_signal_type: str | None = None
    last_signal_time: datetime | None = None
    signal_debounce_seconds: int = 300  # 5 minutes between same signal type

    # Timeouts (in seconds)
    analyzing_timeout: int = 70
    waiting_entry_timeout: int = 60
    waiting_exit_timeout: int = 70

    # Trade counters
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    last_trade_result: str | None = None

    # Error tracking
    error_count: int = 0
    last_error: str | None = None


class StrategyStateMachine:
    """State machine for trading strategy lifecycle.

    Ensures predictable behavior and prevents conflicting operations.
    """

    def __init__(
        self,
        symbol: str,
        cooldown_seconds: int = 60,
        force_stop_loss_percent: Decimal = Decimal("25"),
        profit_signal_threshold: Decimal = Decimal("10"),
    ) -> None:
        """Initialize state machine.

        Args:
            symbol: Trading symbol
            cooldown_seconds: Cooldown period after closing position
            force_stop_loss_percent: Loss percentage to trigger forced stop-loss (25%)
            profit_signal_threshold: Profit percentage increment to signal agent (10%)
        """
        self._symbol = symbol
        self._context = StateContext(cooldown_duration_seconds=cooldown_seconds)
        self._force_stop_loss_percent = force_stop_loss_percent
        self._profit_signal_threshold = profit_signal_threshold

        # Valid state transitions
        self._valid_transitions: dict[
            StrategyState, list[tuple[StateTransition, StrategyState]]
        ] = {
            StrategyState.IDLE: [
                (StateTransition.ENTRY_SIGNAL, StrategyState.ANALYZING),
                (StateTransition.AGENT_OPEN, StrategyState.IN_POSITION),
            ],
            StrategyState.ANALYZING: [
                (StateTransition.CONTEXT_READY, StrategyState.WAITING_ENTRY),
                (StateTransition.AGENT_OPEN, StrategyState.IN_POSITION),
                (StateTransition.ORDER_FILLED, StrategyState.IN_POSITION),
                (StateTransition.AGENT_OBSERVE, StrategyState.IDLE),
                (StateTransition.ORDER_PLACED, StrategyState.WAITING_ENTRY),
                (StateTransition.ORDER_FAILED, StrategyState.IDLE),
                (StateTransition.TIMEOUT, StrategyState.IDLE),
            ],
            StrategyState.WAITING_ENTRY: [
                (StateTransition.AGENT_OPEN, StrategyState.IN_POSITION),
                (StateTransition.ORDER_FILLED, StrategyState.IN_POSITION),
                (StateTransition.AGENT_OBSERVE, StrategyState.IDLE),
                (StateTransition.ORDER_FAILED, StrategyState.IDLE),
                (StateTransition.TIMEOUT, StrategyState.IDLE),
            ],
            StrategyState.IN_POSITION: [
                (StateTransition.PROFIT_THRESHOLD, StrategyState.PROFIT_SIGNAL),
                (StateTransition.RISK_TRIGGERED, StrategyState.RISK_OVERRIDE),
                (StateTransition.AGENT_CLOSE, StrategyState.WAITING_EXIT),
                (StateTransition.POSITION_CLOSED, StrategyState.COOLDOWN),
            ],
            StrategyState.PROFIT_SIGNAL: [
                (StateTransition.AGENT_CLOSE, StrategyState.WAITING_EXIT),
                (StateTransition.AGENT_HOLD, StrategyState.IN_POSITION),
                (StateTransition.RISK_TRIGGERED, StrategyState.RISK_OVERRIDE),
                (StateTransition.TIMEOUT, StrategyState.IN_POSITION),
                (StateTransition.POSITION_CLOSED, StrategyState.COOLDOWN),
            ],
            StrategyState.WAITING_EXIT: [
                (StateTransition.POSITION_CLOSED, StrategyState.COOLDOWN),
                (StateTransition.ORDER_FAILED, StrategyState.IN_POSITION),
                (StateTransition.RISK_TRIGGERED, StrategyState.RISK_OVERRIDE),
                (StateTransition.TIMEOUT, StrategyState.IN_POSITION),
            ],
            StrategyState.RISK_OVERRIDE: [
                (StateTransition.POSITION_CLOSED, StrategyState.COOLDOWN),
                (StateTransition.TIMEOUT, StrategyState.IN_POSITION),
            ],
            StrategyState.COOLDOWN: [
                (StateTransition.COOLDOWN_EXPIRED, StrategyState.IDLE),
                (StateTransition.AGENT_OPEN, StrategyState.IN_POSITION),
            ],
            StrategyState.ERROR: [
                (StateTransition.RESET, StrategyState.IDLE),
            ],
        }

    @property
    def state(self) -> StrategyState:
        """Current state."""
        return self._context.current_state

    @property
    def context(self) -> StateContext:
        """State context."""
        return self._context

    @property
    def is_idle(self) -> bool:
        """Check if ready for new signals."""
        return self._context.current_state == StrategyState.IDLE

    @property
    def has_position(self) -> bool:
        """Check if currently in a position."""
        return self._context.current_state in (
            StrategyState.IN_POSITION,
            StrategyState.PROFIT_SIGNAL,
            StrategyState.WAITING_EXIT,
            StrategyState.RISK_OVERRIDE,
        )

    @property
    def is_analyzing(self) -> bool:
        """Check if currently analyzing a signal."""
        return self._context.current_state in (
            StrategyState.ANALYZING,
            StrategyState.WAITING_ENTRY,
        )

    def should_debounce_signal(self, signal_type: str) -> bool:
        """Check if a signal should be debounced (too soon after same signal).

        Args:
            signal_type: Type of signal to check

        Returns:
            True if signal should be skipped (debounced)
        """
        if self._context.last_signal_type != signal_type:
            return False

        if self._context.last_signal_time is None:
            return False

        elapsed = (datetime.now() - self._context.last_signal_time).total_seconds()
        return elapsed < self._context.signal_debounce_seconds

    def record_signal(self, signal_type: str) -> None:
        """Record that a signal was sent (for debounce tracking).

        Args:
            signal_type: Type of signal that was sent
        """
        self._context.last_signal_type = signal_type
        self._context.last_signal_time = datetime.now()

    def can_transition(self, trigger: StateTransition) -> bool:
        """Check if a transition is valid from current state.

        Args:
            trigger: Transition trigger

        Returns:
            True if transition is valid
        """
        valid = self._valid_transitions.get(self._context.current_state, [])
        return any(t == trigger for t, _ in valid)

    def transition(self, trigger: StateTransition, data: dict[str, Any] | None = None) -> bool:
        """Attempt a state transition.

        Args:
            trigger: Transition trigger
            data: Optional data for the transition

        Returns:
            True if transition succeeded
        """
        data = data or {}
        valid = self._valid_transitions.get(self._context.current_state, [])

        for transition_trigger, next_state in valid:
            if transition_trigger == trigger:
                self._execute_transition(next_state, trigger, data)
                return True

        logger.warning(
            f"Invalid transition: {trigger.value} from state {self._context.current_state.value}"
        )
        return False

    def _execute_transition(
        self,
        next_state: StrategyState,
        trigger: StateTransition,
        data: dict[str, Any],
    ) -> None:
        """Execute a state transition.

        Args:
            next_state: Target state
            trigger: Transition trigger
            data: Transition data
        """
        prev_state = self._context.current_state
        self._context.previous_state = prev_state
        self._context.current_state = next_state
        self._context.state_entered_at = datetime.now()

        logger.info(
            f"State transition: {prev_state.value} -> {next_state.value} (trigger: {trigger.value})"
        )

        # Handle state-specific initialization
        self._on_enter_state(next_state, data)

    def _on_enter_state(self, state: StrategyState, data: dict[str, Any]) -> None:
        """Handle state entry.

        Args:
            state: New state
            data: Transition data
        """
        if state == StrategyState.ANALYZING:
            self._context.pending_signal_type = data.get("signal_type")
            self._context.pending_signal_data = data.get("signal_data", {})

        elif state == StrategyState.IN_POSITION:
            pos = data.get("position")
            if pos:
                self._context.position = pos
                self._context.position.entry_time = datetime.now()
                self._context.position.highest_pnl_percent = Decimal("0")
                self._context.position.last_profit_signal_threshold = Decimal("0")

        elif state == StrategyState.COOLDOWN:
            self._context.cooldown_until = datetime.now() + timedelta(
                seconds=self._context.cooldown_duration_seconds
            )
            self._context.trades_today += 1

            # Record trade result
            if self._context.position.unrealized_pnl >= 0:
                self._context.wins_today += 1
                self._context.last_trade_result = "win"
            else:
                self._context.losses_today += 1
                self._context.last_trade_result = "loss"

            # Clear position context
            self._context.position = PositionContext()

        elif state == StrategyState.IDLE:
            self._context.pending_signal_type = None
            self._context.pending_signal_data = {}

    def update_position(
        self,
        unrealized_pnl: Decimal,
        unrealized_pnl_percent: Decimal,
    ) -> StateTransition | None:
        """Update position P&L and check for state triggers.

        Args:
            unrealized_pnl: Current unrealized P&L
            unrealized_pnl_percent: P&L as percentage of margin

        Returns:
            Triggered state transition or None
        """
        if not self.has_position:
            return None

        self._context.position.unrealized_pnl = unrealized_pnl
        self._context.position.unrealized_pnl_percent = unrealized_pnl_percent

        # Track highest P&L for trailing stop
        if unrealized_pnl_percent > self._context.position.highest_pnl_percent:
            self._context.position.highest_pnl_percent = unrealized_pnl_percent

        # Check for force stop-loss (25% of margin)
        if unrealized_pnl_percent <= -self._force_stop_loss_percent:
            logger.warning(
                f"Force stop-loss triggered: PnL {unrealized_pnl_percent}% <= "
                f"-{self._force_stop_loss_percent}%"
            )
            return StateTransition.RISK_TRIGGERED

        # Check for profit threshold signal (every 10% increase)
        if self._context.current_state == StrategyState.IN_POSITION:
            next_threshold = (
                self._context.position.last_profit_signal_threshold + self._profit_signal_threshold
            )
            if unrealized_pnl_percent >= next_threshold:
                self._context.position.last_profit_signal_threshold = (
                    unrealized_pnl_percent // self._profit_signal_threshold
                ) * self._profit_signal_threshold
                logger.info(
                    f"Profit threshold reached: PnL {unrealized_pnl_percent}% >= {next_threshold}%"
                )
                return StateTransition.PROFIT_THRESHOLD

        return None

    def update_position_metrics(
        self,
        unrealized_pnl: Decimal,
        unrealized_pnl_percent: Decimal,
    ) -> None:
        """Update position metrics without triggering transitions."""
        if not self.has_position:
            return

        self._context.position.unrealized_pnl = unrealized_pnl
        self._context.position.unrealized_pnl_percent = unrealized_pnl_percent
        if unrealized_pnl_percent > self._context.position.highest_pnl_percent:
            self._context.position.highest_pnl_percent = unrealized_pnl_percent

    def check_timeout(self) -> bool:
        """Check if current state has timed out.

        Returns:
            True if timed out and transition executed
        """
        now = datetime.now()
        elapsed = (now - self._context.state_entered_at).total_seconds()

        if self._context.current_state == StrategyState.ANALYZING:
            if elapsed > self._context.analyzing_timeout:
                return self.transition(StateTransition.TIMEOUT)

        elif self._context.current_state == StrategyState.WAITING_ENTRY:
            if elapsed > self._context.waiting_entry_timeout:
                return self.transition(StateTransition.TIMEOUT)

        elif self._context.current_state == StrategyState.WAITING_EXIT:
            if elapsed > self._context.waiting_exit_timeout:
                return self.transition(StateTransition.TIMEOUT)

        elif self._context.current_state == StrategyState.PROFIT_SIGNAL:
            # Auto-return to IN_POSITION if agent doesn't respond
            if elapsed > self._context.waiting_exit_timeout:
                return self.transition(StateTransition.TIMEOUT)

        elif self._context.current_state == StrategyState.RISK_OVERRIDE:
            if elapsed > self._context.waiting_exit_timeout:
                return self.transition(StateTransition.TIMEOUT)

        elif (
            self._context.current_state == StrategyState.COOLDOWN
            and self._context.cooldown_until
            and now >= self._context.cooldown_until
        ):
            return self.transition(StateTransition.COOLDOWN_EXPIRED)

        return False

    def force_reset(self, reason: str = "Manual reset") -> None:
        """Force reset to IDLE state.

        Args:
            reason: Reason for reset
        """
        logger.warning(f"Force resetting state machine: {reason}")
        self._context.current_state = StrategyState.IDLE
        self._context.previous_state = None
        self._context.state_entered_at = datetime.now()
        self._context.pending_signal_type = None
        self._context.pending_signal_data = {}
        self._context.position = PositionContext()
        self._context.cooldown_until = None

    def get_state_info(self) -> dict[str, Any]:
        """Get current state information for logging/debugging.

        Returns:
            State information dictionary
        """
        return {
            "current_state": self._context.current_state.value,
            "previous_state": (
                self._context.previous_state.value if self._context.previous_state else None
            ),
            "state_entered_at": self._context.state_entered_at.isoformat(),
            "time_in_state_seconds": (
                datetime.now() - self._context.state_entered_at
            ).total_seconds(),
            "has_position": self.has_position,
            "position_side": (
                self._context.position.side.value if self._context.position.side else None
            ),
            "position_pnl_percent": float(self._context.position.unrealized_pnl_percent),
            "trades_today": self._context.trades_today,
            "win_rate": (
                self._context.wins_today / self._context.trades_today
                if self._context.trades_today > 0
                else 0
            ),
            "pending_signal": self._context.pending_signal_type,
        }
